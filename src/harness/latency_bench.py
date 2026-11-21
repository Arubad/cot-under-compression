"""Controlled decode-latency micro-benchmark and the cost model built on it (theory/03_cost_model_H3.md §2).

Why a benchmark and not a fit to the GSM8K runs: at GSM8K lengths the KV term is ~1-10 % of the
bytes a decode step moves (GQA, 2 KV heads), so per-token time barely varies with sequence length
and a regression on in-run timings cannot identify the bandwidth (SMOKE-04 gave 0.27 GB/s and a
negative intercept).  Here the cache is filled with random K/V (values do not affect timing), so
B * t can be pushed to where the KV term is large, without any prefill.

Model (per decode step, all B sequences at context length t):

    step_s(B, t) = alpha + gamma * B + beta * B * t

alpha  : batch-independent cost (weight reads + launch overhead)
gamma  : per-sequence cost independent of context (LM head rows, activations)
beta   : per-sequence-per-cached-token cost (KV reads + attention)  -> effective KV bandwidth
         kv_bytes_per_token / beta.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np


def _fill_cache(model, B: int, t: int):
    import torch
    from transformers import DynamicCache

    cfg = model.config
    n_kv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    p = next(model.parameters())
    cache = DynamicCache()
    for layer in range(cfg.num_hidden_layers):
        k = torch.randn(B, n_kv, t, hd, device=p.device, dtype=p.dtype if p.is_floating_point() else torch.float16)
        v = torch.randn_like(k)
        cache.update(k, v, layer)
    return cache


def bench_point(model, B: int, t: int, n_steps: int = 24, warmup: int = 4) -> float:
    """Median seconds per decode step at batch B with t cached tokens."""
    import torch

    dev = next(model.parameters()).device
    cache = _fill_cache(model, B, t)
    ids = torch.randint(0, model.config.vocab_size, (B, 1), device=dev)
    times: List[float] = []
    with torch.no_grad():
        for i in range(warmup + n_steps):
            cur = t + i
            mask = torch.ones(B, cur + 1, dtype=torch.long, device=dev)
            pos = torch.full((B, 1), cur, dtype=torch.long, device=dev)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            cache = out.past_key_values
            ids = out.logits[:, -1:].argmax(-1)
            if i >= warmup:
                times.append(dt)
    del cache
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return float(np.median(times))


def default_grid(max_bt: int = 65536) -> List[tuple]:
    """(B, t) points spanning the KV-negligible and KV-dominant regimes, capped at B*t <= max_bt."""
    pts = []
    for B in (1, 4, 8, 16):
        for t in (256, 1024, 2048, 4096, 8192):
            if B * t <= max_bt:
                pts.append((B, t))
    return pts


def fit_step_model(points: Sequence[Dict]) -> Dict:
    """Least squares for step_s = alpha + gamma*B + beta*B*t."""
    B = np.array([p["B"] for p in points], dtype=float)
    t = np.array([p["t"] for p in points], dtype=float)
    y = np.array([p["step_s"] for p in points], dtype=float)
    X = np.stack([np.ones_like(B), B, B * t], axis=1)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    ss_res, ss_tot = float(((y - pred) ** 2).sum()), float(((y - y.mean()) ** 2).sum())
    return {"alpha_s": float(coef[0]), "gamma_s": float(coef[1]), "beta_s": float(coef[2]), "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"), "max_rel_resid": float(np.max(np.abs(y - pred) / y))}


def run_bench(model, watts: float, grid: Optional[Sequence[tuple]] = None, n_steps: int = 24) -> Dict:
    from ..interventions.kv_quant import kv_bytes
    from ..interventions.weight_quant import weight_bytes

    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    elem_bits = 8 * next(model.parameters()).element_size()
    points = []
    for B, t in grid or default_grid():
        s = bench_point(model, B, t, n_steps=n_steps)
        points.append({"B": B, "t": t, "step_s": s})
        print(f"[latency_bench] B={B:>2} t={t:>5}  {s * 1e3:7.2f} ms/step", flush=True)
    fit = fit_step_model(points)
    kv_tok = kv_bytes(cfg.num_hidden_layers, cfg.num_key_value_heads, hd, 1, elem_bits)
    w = weight_bytes(model)
    fit.update({
        "points": points,
        "w_bytes": w,
        "kv_bytes_per_token": kv_tok,
        "ref_kv_bits": elem_bits,
        "ref_weight_bits": elem_bits,
        "watts": float(watts),
        "eff_weight_bw_bytes_per_s": w / fit["alpha_s"] if fit["alpha_s"] > 0 else float("nan"),
        "eff_kv_bw_bytes_per_s": kv_tok / fit["beta_s"] if fit["beta_s"] > 0 else float("nan"),
        "source": "latency_bench",
    })
    return fit


def simulated_energy_per_problem(lm: Dict, batch_size: int, mean_prompt: float, mean_gen: float, kv_bits: int = 16, keep_frac_eff: float = 1.0) -> float:
    """Energy per problem for a decode of ``mean_gen`` tokens after ``mean_prompt``, as if the KV cache were
    really stored at ``kv_bits`` (vs the reference element width) and only ``keep_frac_eff`` of it read.

    Batch-shared cost alpha is split over the batch; the KV term scales with bytes actually read.
    Weight-quantised configurations are measured, never simulated, so weights stay at the reference."""
    if lm.get("source") != "latency_bench":
        return float("nan")
    ref_bits = float(lm.get("ref_kv_bits", 32))
    s_kv = (min(float(kv_bits), ref_bits) / ref_bits) if kv_bits < 16 else 1.0
    a, g, b, w = lm["alpha_s"], lm["gamma_s"], lm["beta_s"], lm["watts"]
    total_s = 0.0
    for i in range(int(round(mean_gen))):
        total_s += a / batch_size + g + b * s_kv * keep_frac_eff * (mean_prompt + i)
    return float(total_s * w)
