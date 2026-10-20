"""Attention sparsification as a custom attention function, plus the entropy recorder.

transformers >= 4.48 lets a model use any registered attention function:

    from transformers import AttentionInterface
    AttentionInterface.register("sparse", sparse_attention_forward)
    model.config._attn_implementation = "sparse"      # read by every attention module at forward time

The function receives (module, query, key, value, attention_mask, scaling,
dropout, **kwargs) with query [b, h, q, d] and key/value [b, kv_h, k, d] and must
return (attn_output [b, q, h, d], attn_weights).  We re-implement eager
attention and *mask out* keys on the (layer, head) pairs selected by the global
``SPARSE`` config.  Masking in eager attention measures the accuracy effect of
sparsification exactly but gives no speed-up: on a T4 (no FlashAttention) every
latency/energy number for this arm is simulated by the latency model and is
labelled as such.  The sparsity is applied at every step, prefill included.

Methods (``keep_frac`` = fraction of currently valid keys kept, per query):
  none               full attention (must be bit-exact with eager; control)
  sliding_window     the most recent keep_frac * n keys
  sink_window        the first ``sink_tokens`` keys + a window
  topk_heavy_hitter  the top-k keys by attention score for each query (per-query oracle
                     version of H2O's accumulated-attention heavy hitters)
  leverage           the top-k keys by ||k||^2, a query-independent "universal set" in the
                     spirit of LevAttention's leverage scores (proxy: squared key norm)

Entropy recorder (``record_entropy=True``, run at FP16 with method none): accumulates,
per (layer, head), the mean attention entropy and the mass on the top 10% of keys
over all decode positions, used to pick the 6-layer subset and to test H2.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Union


@dataclass
class SparseConfig:
    method: str = "none"
    keep_frac: float = 1.0
    layers: Optional[Union[int, list]] = None   # None = all layers
    heads: Optional[Union[int, list]] = None    # None = all heads
    sink_tokens: int = 4
    record_entropy: bool = False
    stats: Dict = field(default_factory=dict)   # (layer, head) -> {"ent_sum", "mass_sum", "n"}

    def layer_set(self) -> Optional[Set[int]]:
        if self.layers is None:
            return None
        return {int(self.layers)} if isinstance(self.layers, int) else {int(x) for x in self.layers}

    def head_set(self) -> Optional[Set[int]]:
        if self.heads is None:
            return None
        return {int(self.heads)} if isinstance(self.heads, int) else {int(x) for x in self.heads}

    @property
    def active(self) -> bool:
        return self.method != "none" and self.keep_frac < 1.0

    def summary(self) -> Dict:
        out = {}
        for (l, h), s in self.stats.items():
            out[f"{l}.{h}"] = {"layer": l, "head": h, "entropy": s["ent_sum"] / max(1, s["n"]), "topk_mass": s["mass_sum"] / max(1, s["n"]), "n": s["n"]}
        return out


SPARSE = SparseConfig()  # global, set by the runner before each cell


def set_sparse(cfg: SparseConfig) -> None:
    global SPARSE
    SPARSE = cfg


# ------------------------------------------------------------------ numpy reference for the masks (unit-tested)

def keep_mask_np(scores: "np.ndarray", valid: "np.ndarray", method: str, keep_frac: float, key_norm_sq: Optional["np.ndarray"] = None, sink_tokens: int = 4, q_positions: Optional["np.ndarray"] = None) -> "np.ndarray":
    """scores [q, k] (already causal-masked with -inf), valid [q, k] bool.  Returns bool keep [q, k]."""
    import numpy as np

    q, k = scores.shape
    keep = valid.copy()
    n_valid = valid.sum(axis=1)
    n_keep = np.maximum(1, np.ceil(keep_frac * n_valid)).astype(int)
    pos = np.arange(k)
    if method in ("sliding_window", "sink_window"):
        for i in range(q):
            last = np.where(valid[i])[0]
            if len(last) == 0:
                continue
            cur = last[-1]
            lo = cur - n_keep[i] + 1
            win = (pos >= lo) & (pos <= cur) & valid[i]
            if method == "sink_window":
                win |= (pos < sink_tokens) & valid[i]
            keep[i] = win
    elif method == "topk_heavy_hitter":
        for i in range(q):
            sc = np.where(valid[i], scores[i], -np.inf)
            order = np.argsort(-sc)[: n_keep[i]]
            m = np.zeros(k, dtype=bool)
            m[order] = True
            keep[i] = m & valid[i]
    elif method == "leverage":
        assert key_norm_sq is not None
        for i in range(q):
            sc = np.where(valid[i], key_norm_sq, -np.inf)
            order = np.argsort(-sc)[: n_keep[i]]
            m = np.zeros(k, dtype=bool)
            m[order] = True
            keep[i] = m & valid[i]
    elif method == "none":
        pass
    else:
        raise ValueError(method)
    return keep


# ------------------------------------------------------------------ torch attention function

def _repeat_kv(x, n_rep: int):
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def sparse_attention_forward(module, query, key, value, attention_mask, scaling: float, dropout: float = 0.0, **kwargs):
    import torch

    cfg = SPARSE
    n_rep = getattr(module, "num_key_value_groups", query.shape[1] // key.shape[1])
    key_states = _repeat_kv(key, n_rep)
    value_states = _repeat_kv(value, n_rep)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling  # [b, h, q, k]
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    layer_idx = getattr(module, "layer_idx", -1)
    lset, hset = cfg.layer_set(), cfg.head_set()
    layer_on = cfg.active and (lset is None or layer_idx in lset)

    if layer_on:
        b, h, q, k = scores.shape
        # HF masks add finfo.min (not -inf) to causal/padded positions; treat anything near it as masked
        valid = torch.isfinite(scores) & (scores > torch.finfo(scores.dtype).min * 0.5)
        n_valid = valid.sum(-1, keepdim=True)
        n_keep = torch.clamp((cfg.keep_frac * n_valid.float()).ceil().long(), min=1)
        if cfg.method in ("sliding_window", "sink_window"):
            pos = torch.arange(k, device=scores.device).view(1, 1, 1, k)
            cur = torch.where(valid, pos, torch.full_like(pos, -1).expand_as(valid)).amax(-1, keepdim=True)
            keep = (pos > cur - n_keep) & (pos <= cur) & valid
            if cfg.method == "sink_window":
                keep |= (pos < cfg.sink_tokens) & valid
        elif cfg.method == "topk_heavy_hitter":
            ranks = scores.masked_fill(~valid, float("-inf")).argsort(dim=-1, descending=True).argsort(dim=-1)
            keep = (ranks < n_keep) & valid
        elif cfg.method == "leverage":
            norm = (key_states.float() ** 2).sum(-1).unsqueeze(2).expand(b, h, q, k)  # [b, h, q, k]
            ranks = norm.masked_fill(~valid, float("-inf")).argsort(dim=-1, descending=True).argsort(dim=-1)
            keep = (ranks < n_keep) & valid
        else:
            raise ValueError(cfg.method)
        if hset is not None:
            head_mask = torch.zeros(h, dtype=torch.bool, device=scores.device)
            head_mask[list(hset)] = True
            keep = torch.where(head_mask.view(1, h, 1, 1), keep, valid)
        scores = scores.masked_fill(~keep, float("-inf"))

    attn = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn = torch.nn.functional.dropout(attn, p=dropout)

    if cfg.record_entropy:
        with torch.no_grad():
            a = attn.float()
            ent = -(a * torch.log(a.clamp_min(1e-12))).sum(-1)  # [b, h, q]
            n_valid = (torch.isfinite(scores) & (scores > torch.finfo(scores.dtype).min * 0.5)).sum(-1).clamp_min(1)  # [b, h, q]
            top_n = torch.clamp((0.1 * n_valid.float()).ceil().long(), min=1)
            srt = a.sort(dim=-1, descending=True).values
            cum = srt.cumsum(-1)
            mass = cum.gather(-1, (top_n - 1).unsqueeze(-1)).squeeze(-1)  # [b, h, q]
            # only decode positions (q == 1) and the last prefill position carry useful statistics
            sel = slice(-1, None)
            e = ent[:, :, sel].mean(dim=(0, 2))
            m = mass[:, :, sel].mean(dim=(0, 2))
            for hh in range(a.shape[1]):
                st = cfg.stats.setdefault((layer_idx, hh), {"ent_sum": 0.0, "mass_sum": 0.0, "n": 0})
                st["ent_sum"] += float(e[hh])
                st["mass_sum"] += float(m[hh])
                st["n"] += 1

    out = torch.matmul(attn, value_states).transpose(1, 2).contiguous()
    return out, attn


def register() -> None:
    """Register once per process; safe to call repeatedly."""
    from transformers import AttentionInterface

    try:
        AttentionInterface.register("sparse", sparse_attention_forward)
    except Exception:  # already registered
        pass


def apply_sparse(model, cfg: SparseConfig) -> None:
    """Point every attention module at our function (or back to eager) and set the global config."""
    set_sparse(cfg)
    impl = "sparse" if (cfg.active or cfg.record_entropy) else "eager"
    if impl == "sparse":
        register()
    model.config._attn_implementation = impl
    for m in model.modules():
        if hasattr(m, "config") and hasattr(m.config, "_attn_implementation"):
            m.config._attn_implementation = impl


def choose_layer_subset(stats: Dict, n_layers_total: int, n_pick: int = 6) -> list:
    """Pick layers spanning low to high mean entropy (quantiles), for the per-head sweep."""
    import numpy as np

    per_layer: Dict[int, list] = {}
    for key, s in stats.items():
        per_layer.setdefault(int(s["layer"]), []).append(float(s["entropy"]))
    if not per_layer:
        return list(np.linspace(0, n_layers_total - 1, n_pick).round().astype(int))
    layers = sorted(per_layer)
    means = np.array([np.mean(per_layer[l]) for l in layers])
    order = np.argsort(means)
    picks = np.linspace(0, len(order) - 1, n_pick).round().astype(int)
    return sorted(int(layers[order[p]]) for p in picks)
