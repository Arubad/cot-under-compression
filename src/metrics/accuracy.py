"""Accuracy with paired bootstrap over problems, overall and by tertile within difficulty stratum."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..common.stats import bootstrap_ci, paired_bootstrap_diff


def accuracy_table(records: List[Dict], meta: pd.DataFrame, fp16_records: Optional[List[Dict]] = None, n_boot: int = 1000) -> Dict:
    """``meta``: id, tertile, stratum.  Returns overall accuracy, per-tertile, per-stratum x tertile, and the gap to FP16 (paired)."""
    df = pd.DataFrame([{"id": r["id"], "correct": float(r["correct"]), "n_gen": r["n_gen"]} for r in records]).merge(meta, on="id", how="left")
    out: Dict = {"n": int(len(df)), "accuracy": bootstrap_ci(df["correct"].values, n_boot=n_boot), "mean_gen_tokens": float(df["n_gen"].mean())}
    if fp16_records is not None:
        f = pd.DataFrame([{"id": r["id"], "fp16_correct": float(r["correct"]), "fp16_n_gen": r["n_gen"]} for r in fp16_records])
        df = df.merge(f, on="id", how="left")
        out["gap_to_fp16"] = paired_bootstrap_diff(df["fp16_correct"].values, df["correct"].values, n_boot=n_boot)
        out["token_inflation"] = float(df["n_gen"].mean() / max(1e-9, df["fp16_n_gen"].mean()))
    out["by_tertile"] = {}
    for t, g in df.groupby("tertile"):
        d = {"accuracy": bootstrap_ci(g["correct"].values, n_boot=n_boot)}
        if "fp16_correct" in g:
            d["gap_to_fp16"] = paired_bootstrap_diff(g["fp16_correct"].values, g["correct"].values, n_boot=n_boot)
        out["by_tertile"][int(t)] = d
    if "stratum" in df:
        out["by_stratum_tertile"] = {}
        for (s, t), g in df.groupby(["stratum", "tertile"]):
            d = {"n": int(len(g)), "accuracy": float(g["correct"].mean())}
            if "fp16_correct" in g:
                d["gap_to_fp16"] = paired_bootstrap_diff(g["fp16_correct"].values, g["correct"].values, n_boot=n_boot)
            out["by_stratum_tertile"][f"{s}.{int(t)}"] = d
    return out


def tertile_gap_ratio(acc: Dict) -> float:
    """gap(longest tertile) / gap(shortest tertile); inf if the short gap is <= 0."""
    bt = acc.get("by_tertile", {})
    if 0 not in bt or 2 not in bt or "gap_to_fp16" not in bt[0]:
        return float("nan")
    g0, g2 = bt[0]["gap_to_fp16"]["mean"], bt[2]["gap_to_fp16"]["mean"]
    return float(g2 / g0) if g0 > 0 else (float("inf") if g2 > 0 else float("nan"))


def within_stratum_gap_ratios(acc: Dict) -> Dict[str, float]:
    bst = acc.get("by_stratum_tertile", {})
    out = {}
    for s in ("easy", "medium", "hard"):
        a, b = bst.get(f"{s}.0"), bst.get(f"{s}.2")
        if a and b and "gap_to_fp16" in a:
            g0, g2 = a["gap_to_fp16"]["mean"], b["gap_to_fp16"]["mean"]
            out[s] = float(g2 / g0) if g0 > 0 else (float("inf") if g2 > 0 else float("nan"))
    return out
