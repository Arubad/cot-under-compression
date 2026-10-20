"""Difficulty-aware per-problem configuration selector (sklearn + numpy).

Input: a table with one row per (problem, config): features (q_tokens, n_numbers, q_chars, first_entropy),
correct (0/1), energy_j (per problem under that config).  Policy: for each problem, choose the cheapest
configuration whose predicted probability of being correct exceeds tau; else fall back to FP16.
Evaluated with 5-fold cross-validation over problems against (a) the best fixed configuration at equal or
lower energy and (b) the oracle that knows which configs are correct.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

FEATURES = ["q_tokens", "n_numbers", "q_chars", "first_entropy"]


def _folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def _fit_predict(Xtr, ytr, Xte):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        if len(set(ytr.tolist())) < 2:
            return np.full(len(Xte), float(ytr.mean()))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, 1]
    except ImportError:  # numpy fallback: logistic regression by gradient descent
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.c_[np.ones(len(Xtr)), (Xtr - mu) / sd]
        w = np.zeros(A.shape[1])
        for _ in range(500):
            p = 1 / (1 + np.exp(-A @ w))
            w -= 0.1 * (A.T @ (p - ytr) / len(A) + 1e-3 * w)
        B = np.c_[np.ones(len(Xte)), (Xte - mu) / sd]
        return 1 / (1 + np.exp(-B @ w))


def evaluate_selector(table: pd.DataFrame, fp16_name: str = "fp16", tau: float = 0.8, k_folds: int = 5, seed: int = 0) -> Dict:
    """``table`` columns: id, config, correct, energy_j, FEATURES.  Returns accuracy/energy of selector, best fixed, oracle."""
    configs = sorted(table["config"].unique().tolist())
    wide_c = table.pivot(index="id", columns="config", values="correct")
    wide_e = table.pivot(index="id", columns="config", values="energy_j")
    feats = table.drop_duplicates("id").set_index("id")[FEATURES].reindex(wide_c.index).fillna(0.0)
    ids = wide_c.index.tolist()
    n = len(ids)
    mean_e = wide_e.mean()
    cheap_order = [c for c in mean_e.sort_values().index if c != fp16_name]  # cheapest first

    chosen_correct = np.zeros(n)
    chosen_energy = np.zeros(n)
    chosen_cfg = np.empty(n, dtype=object)
    X = feats.values.astype(float)
    for te in _folds(n, k_folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        probs = {c: _fit_predict(X[tr], wide_c[c].values[tr].astype(float), X[te]) for c in cheap_order}
        for j, i in enumerate(te):
            pick = fp16_name
            for c in cheap_order:
                if probs[c][j] >= tau:
                    pick = c
                    break
            chosen_cfg[i] = pick
            chosen_correct[i] = wide_c.loc[ids[i], pick]
            chosen_energy[i] = wide_e.loc[ids[i], pick]

    sel_acc, sel_e = float(chosen_correct.mean()), float(chosen_energy.mean())
    fixed = {c: {"accuracy": float(wide_c[c].mean()), "energy_j": float(wide_e[c].mean())} for c in configs}
    # best fixed config at equal or lower energy than the selector
    cands = {c: v for c, v in fixed.items() if v["energy_j"] <= sel_e + 1e-9}
    best_fixed = max(cands.items(), key=lambda kv: kv[1]["accuracy"]) if cands else max(fixed.items(), key=lambda kv: kv[1]["accuracy"])
    # oracle: cheapest correct config per problem, else cheapest overall
    orc_c, orc_e = [], []
    for i in ids:
        ok = [c for c in cheap_order + [fp16_name] if wide_c.loc[i, c] == 1]
        c = min(ok, key=lambda c: wide_e.loc[i, c]) if ok else min(configs, key=lambda c: wide_e.loc[i, c])
        orc_c.append(wide_c.loc[i, c])
        orc_e.append(wide_e.loc[i, c])
    return {
        "tau": tau,
        "selector": {"accuracy": sel_acc, "energy_j": sel_e, "config_shares": pd.Series(chosen_cfg).value_counts(normalize=True).to_dict()},
        "best_fixed_at_equal_or_lower_energy": {"config": best_fixed[0], **best_fixed[1]},
        "oracle": {"accuracy": float(np.mean(orc_c)), "energy_j": float(np.mean(orc_e))},
        "fixed": fixed,
        "gain_points_vs_best_fixed": 100 * (sel_acc - best_fixed[1]["accuracy"]),
    }


def h3_selector_verdict(res: Dict) -> Dict:
    g = res["gain_points_vs_best_fixed"]
    return {"gain_points": g, "verdict": "confirmed" if g >= 1.0 else ("refuted" if g < 0.5 else "under-powered")}
