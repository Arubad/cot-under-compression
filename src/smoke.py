"""Tiny end-to-end run: 30 GSM8K problems, 64 new tokens, every code path.

baseline (fp16 cache + zero-change control + pass@2 difficulty + head stats + latency model)
-> int4 per-token KV (fake-quant cache) -> int8 weights (bitsandbytes, skipped if it fails to import)
-> top-k 50% on all layers -> single-head sparsification -> analyze.
Run this first on Kaggle (a few minutes on a T4).  Nothing it produces is a result.
It also records which KV backends work on this GPU (fake always; hf_quanto / hf_hqq if installed).
"""
from __future__ import annotations

import json
import os
import tempfile

from .common.config import resolve


def smoke_config(model: str, tmp: str) -> dict:
    return {
        "experiment_id": "SMOKE_I1_baseline",
        "seed": 0,
        "model": {"name": model, "dtype": "auto", "attn_implementation": "eager", "n_layers": 28, "n_heads": 12},
        "task": {"dataset": "gsm8k", "subset": None, "n": 30},
        "generation": {"max_new_tokens": 64, "batch_size": 8, "topk": 8},
        "baseline": {"control_problems": 12, "difficulty": {"k": 2, "temperature": 0.7}},
        "intervention": {"weight_bits": 16, "kv_bits": 16, "kv_scheme": "per_token", "sparsifier": "none", "keep_frac": 1.0},
        "metrics": {"bootstrap_resamples": 200},
        "cost": {"poll_hz": 10, "repeats": 1, "fallback_w_active": 70.0},
        "logging": {"output_dir": os.path.join(tmp, "results")},
    }


def run_smoke(model: str = "Qwen/Qwen2.5-1.5B-Instruct") -> None:
    from . import run as R
    from .analysis.figures import analyze

    tmp = tempfile.mkdtemp(prefix="cot_smoke_")
    os.environ["RESULTS_ROOT"] = os.path.join(tmp, "results")
    cfg = smoke_config(model, tmp)
    cfg_path = os.path.join(tmp, "smoke.yaml")
    import yaml

    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    class A:  # argparse stand-in
        config = cfg_path

    R.cmd_baseline(A)
    cells = [
        ("SMOKE_I3_kv_sweep", {"kv_bits": 4, "kv_scheme": "per_token"}),
        ("SMOKE_I3_kv_sweep", {"kv_bits": 4, "kv_scheme": "per_channel"}),
        ("SMOKE_I4_sparse_sweep", {"sparsifier": "topk_heavy_hitter", "keep_frac": 0.5}),
        ("SMOKE_I5_perhead_sweep", {"sparsifier": "leverage", "keep_frac": 0.5, "layer": 3, "head": 0}),
        ("SMOKE_I2_weight_sweep", {"weight_bits": 8}),
    ]
    status = {}
    for exp, cell in cells:
        c = resolve({**cfg, "experiment_id": exp}, cell)
        try:
            m = R.run_cell(c)
            status[json.dumps(cell)] = {"acc": m["acc_mean"], "median_first_div": m["median_first_div"]}
        except Exception as e:  # bitsandbytes may be missing; report, do not hide
            status[json.dumps(cell)] = f"FAILED: {type(e).__name__}: {e}"
    # optional HF quantized-cache backends
    for backend in ("hf_quanto", "hf_hqq"):
        c = resolve({**cfg, "experiment_id": "SMOKE_I3_kv_sweep"}, {"kv_bits": 4, "kv_backend": backend})
        try:
            m = R.run_cell(c)
            status[backend] = {"acc": m["acc_mean"]}
        except Exception as e:
            status[backend] = f"unavailable: {type(e).__name__}: {str(e)[:120]}"
    analyze(os.environ["RESULTS_ROOT"])
    print(json.dumps(status, indent=2))
    with open(os.path.join(os.environ["RESULTS_ROOT"], "smoke_status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print("SMOKE OK ->", tmp)
