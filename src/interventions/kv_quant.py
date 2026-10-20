"""KV-cache quantisation.

Primary backend: ``FakeQuantCache``, a ``DynamicCache`` subclass that quantises
and immediately dequantises every key/value block it stores ("fake quant").  It
reproduces the *accuracy* effect of a real low-bit cache exactly, under our
control (bits, per-token vs per-channel grouping, sink protection, prefill
policy), while speed and memory stay those of fp16: the cost side comes from
the latency model and the HF backend (below).

Schemes (x has shape [batch, heads, tokens, head_dim]):
  per_token   one (scale, zero-point) per token per head, over head_dim
  per_channel one (scale, zero-point) per channel per head, over a block of
              ``group_size`` tokens (KIVI-style); decode tokens are kept in fp16
              in a residual tail until a block completes, then re-quantised in place.
Quantiser: asymmetric uniform with 2^bits levels (``mode="asym"``, the KIVI /
KVQuant convention) or symmetric with scale = max|x| / (2^(bits-1) - 1).
``sink_tokens`` > 0 keeps the first positions in fp16 (StreamingLLM sinks).
``quantise_prefill=False`` (default): the prefill forward sees fp16 keys/values
and the cache is quantised *after* prefill, the realistic deployment setting.

Alternative backend: transformers' built-in ``QuantizedCache`` via
``generate(cache_implementation="quantized", cache_config={"backend": "quanto"|"hqq", "nbits": b})``
(``backend: hf_quanto`` / ``hf_hqq`` in the config).  Whether quanto/hqq run on a
T4 must be checked in the smoke test and logged.

At bits >= 16 the cache is a plain DynamicCache and must be bit-exact (control).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np


# ------------------------------------------------------------------ numpy reference (unit-tested)

def fake_quant_np(x: np.ndarray, bits: int, axis: int, mode: str = "asym") -> np.ndarray:
    """Quantise-dequantise ``x`` with one scale per slice along ``axis`` (reduction over ``axis``)."""
    if bits >= 16:
        return x
    x = np.asarray(x, dtype=np.float32)
    if mode == "sym":
        qmax = 2 ** (bits - 1) - 1
        scale = np.max(np.abs(x), axis=axis, keepdims=True) / qmax
        scale = np.where(scale == 0, 1.0, scale)
        q = np.clip(np.round(x / scale), -qmax - 1, qmax)
        return (q * scale).astype(np.float32)
    levels = 2**bits - 1
    lo = np.min(x, axis=axis, keepdims=True)
    hi = np.max(x, axis=axis, keepdims=True)
    scale = (hi - lo) / levels
    scale = np.where(scale == 0, 1.0, scale)
    zp = np.round(-lo / scale)
    q = np.clip(np.round(x / scale) + zp, 0, levels)
    return ((q - zp) * scale).astype(np.float32)


def fake_quant_torch(x, bits: int, axis: int, mode: str = "asym"):
    """Same formula as ``fake_quant_np`` on a torch tensor (computed in fp32, returned in x.dtype)."""
    import torch

    if bits >= 16:
        return x
    xf = x.float()
    if mode == "sym":
        qmax = 2 ** (bits - 1) - 1
        scale = xf.abs().amax(dim=axis, keepdim=True) / qmax
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(xf / scale), -qmax - 1, qmax)
        return (q * scale).to(x.dtype)
    levels = 2**bits - 1
    lo = xf.amin(dim=axis, keepdim=True)
    hi = xf.amax(dim=axis, keepdim=True)
    scale = (hi - lo) / levels
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    zp = torch.round(-lo / scale)
    q = torch.clamp(torch.round(xf / scale) + zp, 0, levels)
    return ((q - zp) * scale).to(x.dtype)


def quantise_block_np(k: np.ndarray, bits: int, scheme: str, mode: str = "asym") -> np.ndarray:
    """k: [..., tokens, head_dim].  per_token reduces over head_dim; per_channel over tokens."""
    return fake_quant_np(k, bits, axis=-1 if scheme == "per_token" else -2, mode=mode)


# ------------------------------------------------------------------ torch cache

def _make_cache_class():
    import torch
    from transformers import DynamicCache

    class FakeQuantCache(DynamicCache):
        def __init__(self, bits: int = 4, scheme: str = "per_token", group_size: int = 64, sink_tokens: int = 0, quantise_prefill: bool = False, mode: str = "asym"):
            super().__init__()
            self.bits, self.scheme, self.group_size, self.sink_tokens, self.quantise_prefill, self.mode = bits, scheme, group_size, sink_tokens, quantise_prefill, mode
            self._n_quantised: Dict[int, int] = {}  # per layer: tokens already quantised (per_channel residual bookkeeping)

        # --- storage access that works across transformers versions (key_cache lists or CacheLayer objects)
        def _get(self, layer_idx: int):
            if hasattr(self, "layers") and len(self.layers) > layer_idx:
                return self.layers[layer_idx].keys, self.layers[layer_idx].values
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def _set(self, layer_idx: int, k, v):
            if hasattr(self, "layers") and len(self.layers) > layer_idx:
                self.layers[layer_idx].keys, self.layers[layer_idx].values = k, v
            else:
                self.key_cache[layer_idx], self.value_cache[layer_idx] = k, v

        def _q(self, x):
            axis = -1 if self.scheme == "per_token" else -2
            return fake_quant_torch(x, self.bits, axis, self.mode)

        def _quantise_range(self, x, start: int, end: int):
            """Return x with positions [start, end) quantised (respecting sinks)."""
            s = max(start, self.sink_tokens)
            if s >= end:
                return x
            out = x.clone()
            out[:, :, s:end] = self._q(x[:, :, s:end])
            return out

        def update(self, key_states, value_states, layer_idx: int, cache_kwargs=None):
            if self.bits >= 16:
                return super().update(key_states, value_states, layer_idx, cache_kwargs)
            is_prefill = key_states.shape[-2] > 1 and self.get_seq_length(layer_idx) == 0
            k_full, v_full = super().update(key_states, value_states, layer_idx, cache_kwargs)
            T = k_full.shape[-2]
            done = self._n_quantised.get(layer_idx, 0)
            if self.scheme == "per_token":
                new_end = T
            else:  # per_channel: quantise completed blocks only; keep the tail in fp16
                new_end = done + ((T - done) // self.group_size) * self.group_size
            if new_end > done:
                k_full = self._quantise_range(k_full, done, new_end)
                v_full = self._quantise_range(v_full, done, new_end)
                self._set(layer_idx, k_full, v_full)
                self._n_quantised[layer_idx] = new_end
            if is_prefill and not self.quantise_prefill:
                # prefill attention sees fp16; the cache is already quantised for later steps
                return key_states, value_states
            return k_full, v_full

    return FakeQuantCache


def make_cache_factory(bits: int = 16, scheme: str = "per_token", group_size: int = 64, sink_tokens: int = 0, quantise_prefill: bool = False, backend: str = "fake", mode: str = "asym") -> Optional[Union[Callable, Dict]]:
    """None at full precision; a callable returning a fresh FakeQuantCache; or generate() kwargs for the HF backend."""
    if bits >= 16:
        return None
    if backend.startswith("hf_"):
        return {"cache_implementation": "quantized", "cache_config": {"backend": backend[3:], "nbits": int(bits), "q_group_size": int(group_size), "residual_length": int(group_size)}}
    cls = _make_cache_class()
    return lambda: cls(bits=bits, scheme=scheme, group_size=group_size, sink_tokens=sink_tokens, quantise_prefill=quantise_prefill, mode=mode)


def kv_bytes(n_layers: int, n_kv_heads: int, head_dim: int, tokens: int, bits: int) -> float:
    """Bytes of KV cache for ``tokens`` positions (for the latency model)."""
    return 2 * n_layers * n_kv_heads * head_dim * tokens * bits / 8.0
