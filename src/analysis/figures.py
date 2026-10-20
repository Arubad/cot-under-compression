"""Collect metrics from results/ and draw the paper's figures.

  fig1_first_divergence.png     H1: median first-divergence token vs KV bits (per scheme) and vs weight bits
  fig2_tertile_gaps.png         H1: accuracy gap to FP16 by tertile within difficulty stratum (INT4 KV vs INT4 weights)
  fig3_perhead_entropy.png      H2: per-head accuracy drop vs entropy / top-k mass
  fig4_frontier.png             H3: energy per problem vs accuracy, energy per correct answer; simulated points hollow
  hypothesis_report.json        verdicts
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd

from ..frontier.pareto import frontier_table, h3_frontier_verdict
from ..frontier.selector import evaluate_selector, h3_selector_verdict
from .hypotheses import test_h1, test_h2, test_h3


def load_all_metrics(results_root: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for path in glob.glob(os.path.join(results_root, "*", "*", "metrics.json")):
        with open(path) as f:
            m = json.load(f)
        exp = m.get("_meta", {}).get("experiment_id") or path.split(os.sep)[-3]
        out.setdefault(exp, []).append(m)
    return out


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_first_divergence(kv: List[Dict], weights: List[Dict], out: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for scheme in sorted({r["_meta"]["cell"].get("kv_scheme", "per_token") for r in kv}):
        rows = sorted([r for r in kv if r["_meta"]["cell"].get("kv_scheme", "per_token") == scheme], key=lambda r: -int(r["_meta"]["cell"]["kv_bits"]))
        x = [int(r["_meta"]["cell"]["kv_bits"]) for r in rows]
        m = [r["divergence"]["median_first_divergence"]["mean"] for r in rows]
        lo = [m_ - r["divergence"]["median_first_divergence"]["lo"] for m_, r in zip(m, rows)]
        hi = [r["divergence"]["median_first_divergence"]["hi"] - m_ for m_, r in zip(m, rows)]
        ax[0].errorbar(x, m, yerr=[lo, hi], marker="o", label=f"KV {scheme}")
    ax[0].set_xlabel("KV bits"); ax[0].set_ylabel("median first-divergence token"); ax[0].invert_xaxis(); ax[0].legend(fontsize=8)
    rows = sorted(weights, key=lambda r: -int(r["_meta"]["cell"].get("weight_bits", 16)))
    if rows:
        x = [int(r["_meta"]["cell"]["weight_bits"]) for r in rows]
        m = [r["divergence"]["median_first_divergence"]["mean"] for r in rows]
        ax[1].plot(x, m, marker="s", color="C3")
    ax[1].set_xlabel("weight bits"); ax[1].set_ylabel("median first-divergence token"); ax[1].invert_xaxis()
    fig.tight_layout(); fig.savefig(out, dpi=150)


def fig_tertile_gaps(kv4: Dict, w4: Dict, out: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    width = 0.35
    for k, (name, r) in enumerate((("INT4 KV", kv4), ("INT4 weights", w4))):
        if not r:
            continue
        bt = r["accuracy"]["by_tertile"]
        xs = np.arange(3) + (k - 0.5) * width
        ys = [bt[t]["gap_to_fp16"]["mean"] for t in (0, 1, 2) if t in bt]
        err = [[bt[t]["gap_to_fp16"]["mean"] - bt[t]["gap_to_fp16"]["lo"] for t in (0, 1, 2) if t in bt], [bt[t]["gap_to_fp16"]["hi"] - bt[t]["gap_to_fp16"]["mean"] for t in (0, 1, 2) if t in bt]]
        ax.bar(xs[: len(ys)], ys, width, yerr=err, label=name)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(["short", "medium", "long"]); ax.set_ylabel("accuracy gap to FP16"); ax.legend()
    fig.tight_layout(); fig.savefig(out, dpi=150)


def fig_perhead(h2: Dict, out: str) -> None:
    if "table" not in h2:
        return
    plt = _plt()
    df = pd.DataFrame(h2["table"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].scatter(df["entropy"], df["drop"], c=df["layer"], cmap="viridis"); ax[0].set_xlabel("head attention entropy"); ax[0].set_ylabel("accuracy drop at 50% keep"); ax[0].set_title(f"rho={h2['spearman_entropy_drop']:.2f}")
    ax[1].scatter(df["topk_mass"], df["drop"], c=df["layer"], cmap="viridis"); ax[1].set_xlabel("top-10% key mass"); ax[1].set_title(f"rho={h2['spearman_topk_mass_drop']:.2f}")
    fig.tight_layout(); fig.savefig(out, dpi=150)


def fig_frontier(df: pd.DataFrame, out: str) -> None:
    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for _, r in df.iterrows():
        mk = dict(marker="o", facecolors="none" if r.get("simulated", False) else "C0", edgecolors="C0")
        ax[0].scatter(r["energy_j_per_problem"], r["accuracy"], **mk)
        ax[0].annotate(r["name"], (r["energy_j_per_problem"], r["accuracy"]), fontsize=6)
        ax[1].scatter(r["aggressiveness"], r["energy_j_per_correct"], **mk)
        ax[1].annotate(r["name"], (r["aggressiveness"], r["energy_j_per_correct"]), fontsize=6)
    pf = df[df["pareto_energy_vs_acc"]].sort_values("energy_j_per_problem")
    ax[0].plot(pf["energy_j_per_problem"], pf["accuracy"], "--", color="gray")
    ax[0].set_xlabel("energy per problem (J); hollow = simulated"); ax[0].set_ylabel("accuracy")
    ax[1].set_xlabel("aggressiveness"); ax[1].set_ylabel("energy per correct answer (J)")
    fig.tight_layout(); fig.savefig(out, dpi=150)


def analyze(results_root: str) -> Dict:
    figdir = os.path.join(results_root, "figures")
    os.makedirs(figdir, exist_ok=True)
    allm = load_all_metrics(results_root)
    kv = allm.get("I3_kv_sweep", [])
    weights = allm.get("I2_weight_sweep", [])
    sparse = allm.get("I4_sparse_sweep", [])
    perhead = allm.get("I5_perhead_sweep", [])
    base = allm.get("I1_baseline", [])
    report: Dict = {}
    if kv:
        fig_first_divergence(kv, weights, os.path.join(figdir, "fig1_first_divergence.png"))
        kv4 = next((r for r in kv if int(r["_meta"]["cell"]["kv_bits"]) == 4 and r["_meta"]["cell"].get("kv_scheme", "per_token") == "per_token"), None)
        w4 = next((r for r in weights if int(r["_meta"]["cell"].get("weight_bits", 16)) == 4), None)
        if kv4 or w4:
            fig_tertile_gaps(kv4, w4, os.path.join(figdir, "fig2_tertile_gaps.png"))
        report["H1"] = test_h1(kv, weights)
    stats_path = os.path.join(results_root, "cache", "head_stats.csv")
    if perhead and os.path.exists(stats_path) and base:
        fp16_acc = base[0]["accuracy"]["accuracy"]["mean"]
        report["H2"] = test_h2(perhead, pd.read_csv(stats_path), fp16_acc)
        fig_perhead(report["H2"], os.path.join(figdir, "fig3_perhead_entropy.png"))
    # frontier over every cell that has cost numbers
    rows = []
    for exp, lst in allm.items():
        for r in lst:
            if "cost" in r and "accuracy" in r:
                c = r["_meta"]["cell"]
                rows.append({"name": f"{exp}:{r['_meta'].get('cell_name', c)}", "accuracy": r["accuracy"]["accuracy"]["mean"], "energy_j_per_problem": r["cost"]["energy_j_per_problem"], "mean_gen_tokens": r["accuracy"]["mean_gen_tokens"], "simulated": r["cost"].get("simulated", False), **{k: c.get(k) for k in ("weight_bits", "kv_bits", "kv_scheme", "sparsifier", "keep_frac")}})
    if rows:
        ft = frontier_table(rows)
        ft.to_csv(os.path.join(figdir, "frontier_table.csv"), index=False)
        fig_frontier(ft, os.path.join(figdir, "fig4_frontier.png"))
        fv = h3_frontier_verdict(ft)
        sel_path = os.path.join(results_root, "cache", "selector_table.parquet")
        sv = {"gain_points": float("nan")}
        if os.path.exists(sel_path):
            sv = h3_selector_verdict(evaluate_selector(pd.read_parquet(sel_path)))
        report["H3"] = test_h3(fv, sv)
    ctrl = os.path.join(results_root, "cache", "control.json")
    if os.path.exists(ctrl):
        with open(ctrl) as f:
            report["zero_change_control"] = json.load(f)
    with open(os.path.join(figdir, "hypothesis_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(json.dumps({k: (v.get("verdict", v.get("passed")) if isinstance(v, dict) else v) for k, v in report.items()}, indent=2, default=str))
    return report
