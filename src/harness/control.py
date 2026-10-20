"""The expected-to-fail control: the intervention code paths at full precision must be bit-exact.

Runs the same problems three times — plain eager FP16, through FakeQuantCache
with bits=16, and through the sparse attention function with method=none — and
requires identical generated token sequences on every problem.  A mismatch
means the harness is wrong, and no sweep result may be reported until it passes.
``max_abs_logprob_diff`` tolerates fp16 reduction-order noise in the recorded
logprobs; token sequences get no tolerance.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..interventions.kv_quant import _make_cache_class
from ..interventions.sparsify import SparseConfig, apply_sparse


def _tokens(records: List[Dict]) -> Dict[str, List[int]]:
    return {r["id"]: r["gen_tokens"] for r in records}


def zero_change_control(model, tokenizer, problems: List[Dict], task: str, max_new_tokens: int, batch_size: int, max_abs_logprob_diff: float = 1e-2) -> Dict:
    from .generate import run_generation

    apply_sparse(model, SparseConfig(method="none"))
    base = run_generation(model, tokenizer, problems, task, max_new_tokens, batch_size, desc="control: eager fp16")

    cls = _make_cache_class()
    via_cache = run_generation(model, tokenizer, problems, task, max_new_tokens, batch_size, cache_factory=lambda: cls(bits=16), desc="control: cache path @16 bit")

    apply_sparse(model, SparseConfig(method="none", record_entropy=True))  # exercises the custom attention function
    via_sparse = run_generation(model, tokenizer, problems, task, max_new_tokens, batch_size, desc="control: sparse fn, keep=1.0")
    apply_sparse(model, SparseConfig(method="none"))

    tb, tc, ts = _tokens(base), _tokens(via_cache), _tokens(via_sparse)
    mism_cache = [i for i in tb if tb[i] != tc.get(i)]
    mism_sparse = [i for i in tb if tb[i] != ts.get(i)]
    lp_diff = []
    for b, s in zip(base, via_sparse):
        n = min(len(b["topk_logprobs"]), len(s["topk_logprobs"]))
        if n:
            lp_diff.append(float(np.max(np.abs(np.array(b["topk_logprobs"][:n]) - np.array(s["topk_logprobs"][:n])))))
    result = {
        "n_problems": len(base),
        "cache_path_mismatches": len(mism_cache),
        "sparse_path_mismatches": len(mism_sparse),
        "max_abs_logprob_diff_sparse_vs_eager": float(max(lp_diff)) if lp_diff else 0.0,
        "accuracy_fp16": float(np.mean([r["correct"] for r in base])),
        "nan_in_logprobs": bool(any(np.isnan(r["topk_logprobs"]).any() for r in base if r["topk_logprobs"])),
    }
    result["passed"] = result["cache_path_mismatches"] == 0 and result["sparse_path_mismatches"] == 0 and result["max_abs_logprob_diff_sparse_vs_eager"] <= max_abs_logprob_diff and not result["nan_in_logprobs"]
    if not result["passed"]:
        result["note"] = "HARNESS BROKEN: fix before any sweep. If nan_in_logprobs, set model.dtype: float32 (T4 has no bf16)."
    return result
