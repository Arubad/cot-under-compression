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


# ---------------------------------------------------------------- censoring-aware statistics (amendment 2026-11-21, Q3)
# A problem whose trajectory never leaves the reference is right-censored at its length: T_div > n.
# Ignoring that (the v0.2 median of min(T_div, n)) biases the median low exactly for the mild
# interventions (INT8 KV), which is where the monotonicity test needs it most (theory/01 §5).

def km_curve(times: Sequence[float], events: Sequence[bool]):
    """Kaplan-Meier survival S(t) at the distinct event times.  Returns (t, S)."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=bool)
    ut = np.unique(t[e])
    S, s = [], 1.0
    for u in ut:
        at_risk = np.sum(t >= u)
        d = np.sum((t == u) & e)
        s *= 1.0 - d / at_risk
        S.append(s)
    return ut, np.asarray(S)


def km_median(times: Sequence[float], events: Sequence[bool]) -> float:
    """Smallest t with S(t) <= 0.5; +inf if survival never drops to 0.5 (median censored)."""
    ut, S = km_curve(times, events)
    hit = np.where(S <= 0.5)[0]
    return float(ut[hit[0]]) if len(hit) else float("inf")


def km_median_ci(times: Sequence[float], events: Sequence[bool], n_boot: int = 1000, seed: int = 0) -> Dict[str, float]:
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=bool)
    n = len(t)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    boots = np.sort([km_median(t[i], e[i]) for i in rng.integers(0, n, size=(n_boot, n))])
    # order statistics, not interpolated quantiles: a censored (+inf) bootstrap median must stay +inf
    lo, hi = boots[int(np.floor(0.025 * (n_boot - 1)))], boots[int(np.ceil(0.975 * (n_boot - 1)))]
    return {"mean": km_median(t, e), "lo": float(lo), "hi": float(hi), "n": int(n), "frac_censored": float(1 - e.mean())}


def nelson_aalen_binned(times: Sequence[float], events: Sequence[bool], bin_width: int = 32, t_max: int = 512) -> Dict[str, list]:
    """Hazard per token in bins of ``bin_width``: events in the bin / token-exposure in the bin."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=bool)
    edges = np.arange(0, t_max + bin_width, bin_width)
    haz, ev, expo = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        exposure = float(np.clip(np.minimum(t, hi) - lo, 0, None).sum())
        d = int(np.sum(e & (t >= lo) & (t < hi)))
        haz.append(d / exposure if exposure > 0 else float("nan"))
        ev.append(d)
        expo.append(exposure)
    return {"bin_start": edges[:-1].tolist(), "hazard_per_token": haz, "events": ev, "exposure_tokens": expo}


def exponential_rate(times: Sequence[float], events: Sequence[bool]) -> float:
    """Censoring-aware constant-hazard MLE: events / total exposure (tokens)."""
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=bool)
    expo = float(np.sum(t + e))  # an event at index T means T+1 tokens were exposed
    return float(e.sum() / expo) if expo > 0 else float("nan")


def fit_sigma_exponent(rates: Dict[int, float]) -> Dict[str, float]:
    """Fit log h = c + beta log sigma_b with sigma_b ∝ 1/(2^b - 1) (same quantiser, theory/01 §1).
    beta ≈ 1 supports the near-tie (first-order) mechanism, beta ≈ 2 a second-order one.  Secondary."""
    bits = sorted(b for b, r in rates.items() if np.isfinite(r) and r > 0)
    if len(bits) < 2:
        return {"beta": float("nan"), "n_points": len(bits)}
    x = np.log([1.0 / (2**b - 1) for b in bits])
    y = np.log([rates[b] for b in bits])
    A = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return {"beta": float(coef[1]), "intercept": float(coef[0]), "n_points": len(bits), "bits": bits}


def divergence_summary(records: List[Dict], ref_records: List[Dict]) -> Dict:
    from ..common.stats import bootstrap_ci

    by_id = {r["id"]: r for r in ref_records}
    rows = [trajectory_divergence(r, by_id[r["id"]]) for r in records if r["id"] in by_id]
    fds = np.array([r["first_divergence_or_len"] for r in rows], dtype=float)
    ev = np.array([r["diverged"] for r in rows], dtype=bool)
    return {
        "km_median_first_divergence": km_median_ci(fds, ev) if rows else {},
        "nelson_aalen": nelson_aalen_binned(fds, ev) if rows else {},
        "exponential_hazard_per_token": exponential_rate(fds, ev) if rows else float("nan"),
        "n": len(rows),
        "frac_diverged": float(np.mean([r["diverged"] for r in rows])) if rows else float("nan"),
        "median_first_divergence": bootstrap_ci(fds, stat=np.median) if rows else {},
        "mean_kl_before_divergence": bootstrap_ci([r["mean_kl_before_divergence"] for r in rows]) if rows else {},
        "argmax_agreement": float(np.nanmean([r["argmax_agreement"] for r in rows])) if rows else float("nan"),
        "per_problem": rows,
    }
