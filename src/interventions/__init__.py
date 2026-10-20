"""The three efficiency knobs: weight precision, KV-cache precision, attention sparsity.

``build_intervention(cfg)`` returns ``(quant_config, cache_factory, sparse_config)``
for one cell; ``apply_sparse(model, sparse_config)`` switches the attention
implementation.  At full precision / keep_frac = 1.0 every path must reproduce
the plain FP16 run token-for-token (harness/control.py checks this).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .kv_quant import make_cache_factory
from .sparsify import SparseConfig
from .weight_quant import make_quant_config


def build_intervention(icfg: Dict) -> Tuple[Optional[object], Optional[object], SparseConfig]:
    quant = make_quant_config(int(icfg.get("weight_bits", 16)), icfg.get("weight_method", "bnb"))
    cache_factory = make_cache_factory(
        bits=int(icfg.get("kv_bits", 16)),
        scheme=icfg.get("kv_scheme", "per_token"),
        group_size=int(icfg.get("kv_group_size", 64)),
        sink_tokens=int(icfg.get("kv_sink_tokens", 0)),
        quantise_prefill=bool(icfg.get("kv_quantise_prefill", False)),
        backend=icfg.get("kv_backend", "fake"),
    )
    sp = SparseConfig(
        method=icfg.get("sparsifier", "none"),
        keep_frac=float(icfg.get("keep_frac", 1.0)),
        layers=icfg.get("layer"),
        heads=icfg.get("head"),
        sink_tokens=int(icfg.get("sparse_sink_tokens", 4)),
        record_entropy=bool(icfg.get("record_entropy", False)),
    )
    return quant, cache_factory, sp
