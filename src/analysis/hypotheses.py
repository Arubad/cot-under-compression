"""Pre-registered pass/fail tests applied mechanically to the collected metrics (thresholds from PREREGISTRATION.md v0.2).

Each metrics.json (one per cell) carries: accuracy (dict with by_tertile / by_stratum_tertile), divergence
(median_first_divergence CI, frac_diverged), cost (energy_j_per_problem, simulated flag), the cell knobs in _meta.cell.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..common.stats import ci_overlaps, spearman
from ..metrics.accuracy import tertile_gap_ratio, within_stratum_gap_ratios


def _cells(rows: List[Dict], **match) -> List[Dict]:
    out = []
    for r in rows:
        c = r.get("_meta", {}).get("cell", {})
        if all(c.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


def test_h1(kv_rows: List[Dict], weight_rows: List[Dict], scheme: str = "per_token") -> Dict:
    """H1: median first-divergence decreases monotonically INT8 -> INT4 -> INT2 for KV with non-overlapping CIs;
    longest-tertile gap >= 2x shortest within each difficulty stratum (INT4 KV); INT4-weight gap ratio < 1.5."""
    kv = [r for r in kv_rows if r.get("_meta", {}).get("cell", {}).get("kv_scheme", scheme) == scheme]
    by_bits = {int(r["_meta"]["cell"]["kv_bits"]): r for r in kv}
    bits = sorted(by_bits, reverse=True)  # 8, 4, 2
    med = {b: by_bits[b]["divergence"]["median_first_divergence"] for b in bits if by_bits[b].get("divergence")}
    mono = all(med[bits[i]]["mean"] > med[bits[i + 1]]["mean"] and not ci_overlaps(med[bits[i]], med[bits[i + 1]]) for i in range(len(bits) - 1)) if len(med) >= 2 else False
    within_ci = all(ci_overlaps(med[bits[i]], med[bits[i + 1]]) for i in range(len(bits) - 1)) if len(med) >= 2 else False
    kv4 = by_bits.get(4)
    tert_ratio = tertile_gap_ratio(kv4["accuracy"]) if kv4 else float("nan")
    strata = within_stratum_gap_ratios(kv4["accuracy"]) if kv4 else {}
    strata_ok = bool(strata) and all(v >= 2.0 for v in strata.values() if np.isfinite(v))
    tert_within_ci = False
    if kv4:
        bt = kv4["accuracy"].get("by_tertile", {})
        if 0 in bt and 2 in bt and "gap_to_fp16" in bt[0]:
            tert_within_ci = ci_overlaps(bt[0]["gap_to_fp16"], bt[2]["gap_to_fp16"])
    w4 = next((r for r in weight_rows if int(r["_meta"]["cell"].get("weight_bits", 16)) == 4), None)
    w_ratio = tertile_gap_ratio(w4["accuracy"]) if w4 else float("nan")
    w_ok = np.isfinite(w_ratio) and w_ratio < 1.5
    if within_ci or tert_within_ci:
        verdict = "refuted"
    elif mono and strata_ok and w_ok:
        verdict = "confirmed"
    else:
        verdict = "under-powered" if (kv4 is None or w4 is None or len(med) < 3) else "inconclusive"
    return {"verdict": verdict, "median_first_divergence_by_bits": med, "monotone_non_overlapping": mono, "kv4_tertile_gap_ratio": tert_ratio, "kv4_within_stratum_ratios": strata, "weight4_tertile_gap_ratio": w_ratio}


def test_h2(perhead_rows: List[Dict], head_stats: pd.DataFrame, fp16_accuracy: float) -> Dict:
    """H2: Spearman(entropy, accuracy drop under 50% single-head sparsification) >= 0.6 and <= -0.5 for top-k mass; refuted if |rho| < 0.3."""
    rows = []
    st = head_stats.set_index(["layer", "head"]) if not head_stats.empty else None
    for r in perhead_rows:
        c = r["_meta"]["cell"]
        if "layer" not in c or "head" not in c or st is None:
            continue
        key = (int(c["layer"]), int(c["head"]))
        if key not in st.index:
            continue
        rows.append({"layer": key[0], "head": key[1], "drop": fp16_accuracy - r["accuracy"]["accuracy"]["mean"], "entropy": float(st.loc[key, "entropy"]), "topk_mass": float(st.loc[key, "topk_mass"])})
    df = pd.DataFrame(rows)
    if len(df) < 5:
        return {"verdict": "inconclusive", "n_heads": int(len(df))}
    rho_e = spearman(df["entropy"].values, df["drop"].values)
    rho_m = spearman(df["topk_mass"].values, df["drop"].values)
    if abs(rho_e) < 0.3:
        verdict = "refuted"
    elif rho_e >= 0.6 and rho_m <= -0.5:
        verdict = "confirmed"
    else:
        verdict = "under-powered"
    return {"verdict": verdict, "n_heads": int(len(df)), "spearman_entropy_drop": rho_e, "spearman_topk_mass_drop": rho_m, "table": df.to_dict("records")}


def test_h3(frontier_verdict: Dict, selector_verdict: Dict) -> Dict:
    """H3: best energy-per-correct config is not the most aggressive AND selector gains >= 1 point at equal or lower energy."""
    not_aggr = not frontier_verdict.get("best_is_most_aggressive", True)
    g = selector_verdict.get("gain_points", float("nan"))
    if frontier_verdict.get("best_is_most_aggressive", False) or (np.isfinite(g) and g < 0.5):
        verdict = "refuted"
    elif not_aggr and g >= 1.0:
        verdict = "confirmed"
    else:
        verdict = "under-powered"
    return {"verdict": verdict, "frontier": frontier_verdict, "selector": selector_verdict}
