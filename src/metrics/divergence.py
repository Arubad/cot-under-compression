"""First-divergence token and per-token KL against the cached FP16 trajectory (numpy only)."""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def first_divergence(tokens: Sequence[int], ref_tokens: Sequence[int]) -> int:
    """Index of the first position where the two greedy sequences differ; len(min) if one is a prefix of the other
    (and -1 if they are identical and of equal length)."""
    n = min(len(tokens), len(ref_tokens))
    for i in range(n):
        if tokens[i] != ref_tokens[i]:
            return i
    return -1 if len(tokens) == len(ref_tokens) else n


def topk_kl(ref_ids: np.ndarray, ref_lp: np.ndarray, cfg_ids: np.ndarray, cfg_lp: np.ndarray) -> float:
    """KL(ref || cfg) restricted to the union of the two top-k supports, with the remaining mass lumped into one bucket.

    Exact KL needs full distributions; this is the standard truncated estimate and is
    only compared *across* configurations, never reported as an absolute number."""
    ref = dict(zip(ref_ids.tolist(), ref_lp.tolist()))
    cfg = dict(zip(cfg_ids.tolist(), cfg_lp.tolist()))
    support = set(ref) | set(cfg)
    r_tail = max(1e-12, 1.0 - sum(np.exp(v) for v in ref.values()))
    c_tail = max(1e-12, 1.0 - sum(np.exp(v) for v in cfg.values()))
    r_min, c_min = np.log(r_tail / max(1, len(support))), np.log(c_tail / max(1, len(support)))
    kl = 0.0
    for t in support:
        lr = ref.get(t, r_min)
        lc = cfg.get(t, c_min)
        kl += np.exp(lr) * (lr - lc)
    kl += r_tail * (np.log(r_tail) - np.log(c_tail))
    return float(max(0.0, kl))


def trajectory_divergence(rec: Dict, ref: Dict) -> Dict[str, float]:
    """Per-problem: first divergence, mean top-k KL *before* divergence (the quantised model's drift while still on track),
    and the argmax agreement rate over the shared prefix."""
    fd = first_divergence(rec["gen_tokens"], ref["gen_tokens"])
    n_shared = fd if fd >= 0 else min(len(rec["gen_tokens"]), len(ref["gen_tokens"]))
    kls = []
    for t in range(min(n_shared, len(rec["topk_ids"]), len(ref["topk_ids"]))):
        kls.append(topk_kl(np.array(ref["topk_ids"][t]), np.array(ref["topk_logprobs"][t]), np.array(rec["topk_ids"][t]), np.array(rec["topk_logprobs"][t])))
    n = min(len(rec["gen_tokens"]), len(ref["gen_tokens"]))
    agree = float(np.mean([a == b for a, b in zip(rec["gen_tokens"][:n], ref["gen_tokens"][:n])])) if n else float("nan")
    return {"first_divergence": fd, "first_divergence_or_len": n_shared if fd < 0 else fd, "mean_kl_before_divergence": float(np.mean(kls)) if kls else 0.0, "argmax_agreement": agree, "diverged": fd >= 0}


def divergence_summary(records: List[Dict], ref_records: List[Dict]) -> Dict:
    from ..common.stats import bootstrap_ci

    by_id = {r["id"]: r for r in ref_records}
    rows = [trajectory_divergence(r, by_id[r["id"]]) for r in records if r["id"] in by_id]
    fds = np.array([r["first_divergence_or_len"] for r in rows], dtype=float)
    return {
        "n": len(rows),
        "frac_diverged": float(np.mean([r["diverged"] for r in rows])) if rows else float("nan"),
        "median_first_divergence": bootstrap_ci(fds, stat=np.median) if rows else {},
        "mean_kl_before_divergence": bootstrap_ci([r["mean_kl_before_divergence"] for r in rows]) if rows else {},
        "argmax_agreement": float(np.nanmean([r["argmax_agreement"] for r in rows])) if rows else float("nan"),
        "per_problem": rows,
    }
