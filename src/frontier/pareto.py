"""Pareto sets, energy per correct answer, token inflation and the decode-latency model (numpy only).

A configuration row: name, accuracy, energy_j_per_problem (measured or simulated),
mean_gen_tokens, simulated (bool), plus the knobs (weight_bits, kv_bits, sparsifier, keep_frac).
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from ..common.stats import linear_fit


def energy_per_correct(energy_per_problem: float, accuracy: float) -> float:
    return float(energy_per_problem / accuracy) if accuracy > 0 else float("inf")


def pareto_front(df: pd.DataFrame, cost_col: str, gain_col: str) -> List[int]:
    """Indices of rows not dominated (lower cost and higher gain by any other row)."""
    cost, gain = df[cost_col].values, df[gain_col].values
    keep = []
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if j != i and cost[j] <= cost[i] and gain[j] >= gain[i] and (cost[j] < cost[i] or gain[j] > gain[i]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def aggressiveness(row: Dict) -> float:
    """Higher = more aggressive (lower bits, fewer keys kept).  Used only to name 'the most aggressive configuration'."""
    wb, kb, kf = float(row.get("weight_bits", 16)), float(row.get("kv_bits", 16)), float(row.get("keep_frac", 1.0))
    return (16 - wb) / 16 + (16 - kb) / 16 + (1 - kf)


def frontier_table(rows: Sequence[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    df["energy_j_per_correct"] = [energy_per_correct(e, a) for e, a in zip(df["energy_j_per_problem"], df["accuracy"])]
    df["aggressiveness"] = [aggressiveness(r) for r in rows]
    df["pareto_energy_vs_acc"] = False
    df.loc[pareto_front(df, "energy_j_per_problem", "accuracy"), "pareto_energy_vs_acc"] = True
    best = int(df["energy_j_per_correct"].idxmin())
    df["best_energy_per_correct"] = False
    df.loc[best, "best_energy_per_correct"] = True
    return df


def h3_frontier_verdict(df: pd.DataFrame) -> Dict:
    """Is the best energy-per-correct configuration the most aggressive one?  (First clause of H3.)"""
    best = df[df["best_energy_per_correct"]].iloc[0]
    most_aggr = df.loc[df["aggressiveness"].idxmax()]
    return {"best_config": str(best["name"]), "most_aggressive_config": str(most_aggr["name"]), "best_is_most_aggressive": bool(best["name"] == most_aggr["name"]), "best_energy_j_per_correct": float(best["energy_j_per_correct"]), "most_aggressive_energy_j_per_correct": float(most_aggr["energy_j_per_correct"]), "best_is_simulated": bool(best.get("simulated", False))}


# ---------------------------------------------------------------- latency model

def fit_latency_model(seq_lens: Sequence[float], per_token_s: Sequence[float], w_bytes: float, kv_bytes_per_token: float) -> Dict:
    """Fit per-token decode latency ~ (W_bytes + kv_bytes_per_token * t) / BW + c.

    With x = W_bytes + kv_bytes_per_token * t, a linear fit gives slope = 1/BW and intercept c."""
    x = np.array([w_bytes + kv_bytes_per_token * t for t in seq_lens], dtype=float)
    fit = linear_fit(x, np.asarray(per_token_s, dtype=float))
    bw = 1.0 / fit["slope"] if fit["slope"] > 0 else float("nan")
    threshold_t = w_bytes / kv_bytes_per_token if kv_bytes_per_token > 0 else float("inf")
    return {"bandwidth_bytes_per_s": bw, "overhead_s": fit["intercept"], "r2": fit["r2"], "kv_exceeds_weights_at_tokens": float(threshold_t)}


def simulated_energy(n_tokens: Sequence[float], w_bytes: float, kv_bytes_per_token: float, bw: float, overhead_s: float, watts: float) -> float:
    """Energy for one decode of n tokens under the latency model (used for the sparse arm and for KV-bit variants
    where the fake-quant cache does not change real memory traffic)."""
    total = 0.0
    for t in range(int(np.mean(n_tokens))):
        total += (w_bytes + kv_bytes_per_token * t) / bw + overhead_s
    return float(total * watts)
