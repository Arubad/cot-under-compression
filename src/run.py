"""Command-line entry point.  Every notebook is a thin wrapper around these commands.

  python -m src.run make_subset --config configs/baseline.yaml          # tertile boundaries + 300-problem stratified subset (once)
  python -m src.run baseline    --config configs/baseline.yaml          # I1: FP16 cache, zero-change control, difficulty labels, head stats, power
  python -m src.run control     --config configs/baseline.yaml          # the zero-change control alone (re-run after every harness change)
  python -m src.run sweep       --config configs/kv_sweep.yaml [--minimum] [--skip-existing]     # I2 / I3 / I4 / I6 cells
  python -m src.run perhead     --config configs/perhead_sweep.yaml [--minimum]                  # I5: every head of the 6-layer subset
  python -m src.run frontier    --config configs/frontier.yaml          # I7: selector table + energy repeats on headline configs
  python -m src.run analyze     [--results results]                     # pre-registered tests + figures
  python -m src.run smoke                                               # 30 problems, 64 tokens, every code path
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .common.config import cell_name, config_hash, iter_cells, load_yaml, parse_cell, resolve, run_dir
from .common.explog import finish_entry, start_entry
from .common.kaggle_env import hf_cache_dir, import_previous_results, resolve_dtype, results_root, torch_device
from .common.seed import seed_everything


# ----------------------------------------------------------------------------- shared helpers

def _cache_dir() -> str:
    d = os.path.join(results_root(), "cache")
    os.makedirs(d, exist_ok=True)
    return d


def _fp16_path(task: str) -> str:
    return os.path.join(_cache_dir(), f"fp16_{task}.parquet")


def load_fp16_records(task: str) -> List[Dict]:
    p = _fp16_path(task)
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p} missing: run `python -m src.run baseline` first")
    return pd.read_parquet(p).to_dict("records")


def _problems(cfg: Dict):
    from .harness.data import load_task

    t = cfg["task"]
    subset = t.get("subset") if t["dataset"] == "gsm8k" else None
    return load_task(t["dataset"], subset, t.get("n"))


def _meta_frame(cfg: Dict, problems) -> pd.DataFrame:
    """id, tertile, stratum for the problems of this task."""
    from .harness.data import SUBSET_PATH, assign_tertile, load_tertiles

    ids = [p["id"] for p in problems]
    tert = {}
    if cfg["task"]["dataset"] == "gsm8k" and os.path.exists(SUBSET_PATH):
        with open(SUBSET_PATH) as f:
            tert = json.load(f).get("tertile", {})
    df = pd.DataFrame({"id": ids, "tertile": [int(tert.get(i, 1)) for i in ids]})
    dpath = os.path.join(_cache_dir(), f"difficulty_{cfg['task']['dataset']}.parquet")
    if os.path.exists(dpath):
        df = df.merge(pd.read_parquet(dpath)[["id", "stratum", "n_correct"]], on="id", how="left")
    else:
        df["stratum"] = "unknown"
    return df


def _load_model(cfg: Dict, quant_config=None):
    from .harness.generate import load_model

    device = torch_device()
    dtype = resolve_dtype(cfg["model"].get("dtype", "auto"))
    return load_model(cfg["model"]["name"], dtype, device, cfg["model"].get("attn_implementation", "eager"), quant_config)


def _expected(cfg) -> str:
    c = cfg.get("_cell", {})
    e = cfg.get("expected", {})
    if isinstance(e, dict):
        for k, v in c.items():
            if f"{k}={v}" in e:
                return e[f"{k}={v}"]
        return e.get("default", "")
    return str(e)


def _latency_model(cfg: Dict) -> Optional[Dict]:
    p = os.path.join(_cache_dir(), "latency_model.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _fit_latency_model(records: List[Dict], model, cfg: Dict, watts: float) -> Dict:
    from .frontier.pareto import fit_latency_model
    from .interventions.kv_quant import kv_bytes
    from .interventions.weight_quant import weight_bytes

    mc = model.config
    n_layers, n_kv, head_dim = mc.num_hidden_layers, mc.num_key_value_heads, mc.hidden_size // mc.num_attention_heads
    kv_per_tok = kv_bytes(n_layers, n_kv, head_dim, 1, 16)
    wb = weight_bytes(model)
    # per-batch timing: per-token seconds vs mean sequence length of that batch
    by_batch: Dict[float, List] = {}
    for r in records:
        by_batch.setdefault(r["wall_s_batch"], []).append(r)
    seq, per_tok = [], []
    for wall, rs in by_batch.items():
        steps = max(r["n_gen"] for r in rs)
        if steps < 8:
            continue
        seq.append(float(np.mean([r["prompt_tokens"] + r["n_gen"] / 2 for r in rs])))
        per_tok.append(wall / steps / len(rs))  # per sequence-token (batch amortised)
    fit = fit_latency_model(seq, per_tok, wb, kv_per_tok) if len(seq) >= 3 else {"bandwidth_bytes_per_s": float("nan"), "overhead_s": float("nan"), "r2": float("nan"), "kv_exceeds_weights_at_tokens": float("nan")}
    fit.update({"w_bytes": wb, "kv_bytes_per_token_fp16": kv_per_tok, "watts": watts, "n_layers": n_layers, "n_kv_heads": n_kv, "head_dim": head_dim})
    with open(os.path.join(_cache_dir(), "latency_model.json"), "w") as f:
        json.dump(fit, f, indent=2)
    return fit


def _simulated_energy(cfg: Dict, lm: Dict, mean_prompt: float, mean_gen: float) -> float:
    """Energy per problem from the latency model, with KV bytes scaled by kv bits and keep fraction and weight bytes by weight bits."""
    i = cfg["intervention"]
    wb = lm["w_bytes"] * int(i.get("weight_bits", 16)) / 16.0
    kvb = lm["kv_bytes_per_token_fp16"] * int(i.get("kv_bits", 16)) / 16.0 * float(i.get("keep_frac", 1.0))
    bw, c, w = lm["bandwidth_bytes_per_s"], lm["overhead_s"], lm["watts"]
    if not np.isfinite(bw) or bw <= 0:
        return float("nan")
    total = 0.0
    for t in range(int(mean_gen)):
        total += (wb + kvb * (mean_prompt + t)) / bw + c
    return float(total * w)


# ----------------------------------------------------------------------------- one cell

def run_cell(cfg: Dict, skip_existing: bool = False) -> Optional[Dict]:
    from .harness.generate import run_generation
    from .harness.power import PowerMeter
    from .interventions import build_intervention
    from .interventions.sparsify import apply_sparse
    from .metrics.accuracy import accuracy_table
    from .metrics.divergence import divergence_summary

    d = run_dir(cfg)
    if skip_existing and os.path.exists(os.path.join(d, "metrics.json")):
        print("skip existing", d)
        return None
    seed_everything(int(cfg.get("seed", 0)))
    meta = start_entry(cfg, d, _expected(cfg))
    task = cfg["task"]["dataset"]
    problems = _problems(cfg)
    pmeta = _meta_frame(cfg, problems)
    fp16 = load_fp16_records(task)
    quant, cache_factory, sp = build_intervention(cfg["intervention"])
    tok, model = _load_model(cfg, quant)
    apply_sparse(model, sp)
    g = cfg["generation"]
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    with PowerMeter(poll_hz=float(cfg["cost"]["poll_hz"]), fallback_w_active=float(cfg["cost"].get("fallback_w_active", 70.0))) as pm:
        records = run_generation(model, tok, problems, task, int(g["max_new_tokens"]), int(g["batch_size"]), topk=int(g.get("topk", 16)), cache_factory=cache_factory, desc=cell_name(cfg.get("_cell", {})))
    apply_sparse(model, type(sp)(method="none"))
    acc = accuracy_table(records, pmeta, fp16, n_boot=int(cfg["metrics"]["bootstrap_resamples"]))
    div = divergence_summary(records, fp16)
    per_problem = div.pop("per_problem")
    n_tok = sum(r["n_gen"] for r in records)
    pw = pm.summary()
    cost = {"measured_energy_j_per_problem": pw["energy_j"] / len(records), "wall_s": pw["wall_s"], "tokens_per_s": n_tok / max(1e-9, pw["wall_s"]), "mean_w": pw["mean_w"], "energy_source": pw["energy_source"], "peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else float("nan")}
    i = cfg["intervention"]
    simulated = (int(i.get("kv_bits", 16)) < 16 and i.get("kv_backend", "fake") == "fake") or (i.get("sparsifier", "none") != "none" and float(i.get("keep_frac", 1.0)) < 1.0)
    lm = _latency_model(cfg)
    if simulated and lm:
        cost["energy_j_per_problem"] = _simulated_energy(cfg, lm, float(np.mean([r["prompt_tokens"] for r in records])), float(np.mean([r["n_gen"] for r in records])))
        cost["simulated"] = True
    else:
        cost["energy_j_per_problem"] = cost["measured_energy_j_per_problem"]
        cost["simulated"] = False
    pd.DataFrame(records).drop(columns=["topk_ids", "topk_logprobs"]).assign(**{k: [r[k] for r in per_problem] for k in ("first_divergence", "mean_kl_before_divergence", "argmax_agreement")}).to_parquet(os.path.join(d, "generations.parquet"), index=False)
    metrics = {"accuracy": acc, "divergence": div, "cost": cost, "n_problems": len(records), "acc_mean": acc["accuracy"]["mean"], "gap_to_fp16": acc.get("gap_to_fp16", {}).get("mean"), "median_first_div": div["median_first_divergence"].get("mean"), "energy_j_per_problem": cost["energy_j_per_problem"], "simulated_energy": cost["simulated"]}
    meta["cell_name"] = cell_name(cfg.get("_cell", {}))
    finish_entry(meta, metrics)
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return metrics


# ----------------------------------------------------------------------------- commands

def cmd_make_subset(args):
    from .harness.data import make_subset

    cfg = load_yaml(args.config)
    print(make_subset(cfg["model"]["name"], int(cfg["task"].get("subset_size", 300)), int(cfg.get("seed", 0))))


def cmd_control(args, cfg: Optional[Dict] = None, model_tok=None):
    from .harness.control import zero_change_control

    cfg = cfg or load_yaml(args.config)
    problems = _problems(cfg)[: int(cfg["baseline"].get("control_problems", 50))]
    tok, model = model_tok or _load_model(cfg)
    g = cfg["generation"]
    res = zero_change_control(model, tok, problems, cfg["task"]["dataset"], int(g["max_new_tokens"]), int(g["batch_size"]))
    with open(os.path.join(_cache_dir(), "control.json"), "w") as f:
        json.dump(res, f, indent=2)
    print("ZERO-CHANGE CONTROL:", json.dumps(res, indent=2))
    if not res["passed"]:
        raise SystemExit("zero-change control FAILED; no sweep may run until it passes")
    return res


def cmd_baseline(args):
    """I1: FP16 greedy cache + control + difficulty labels + head statistics + latency model."""
    from .harness.difficulty import cheap_features, pass_at_k_labels
    from .harness.generate import run_generation
    from .harness.power import PowerMeter
    from .interventions.sparsify import SparseConfig, apply_sparse
    from .metrics.accuracy import accuracy_table
    from .metrics.entropy import stats_frame

    cfg = resolve(load_yaml(args.config), {})
    cfg["logging"]["output_dir"] = results_root(cfg["logging"].get("output_dir", "results"))
    task = cfg["task"]["dataset"]
    problems = _problems(cfg)
    tok, model = _load_model(cfg)
    g = cfg["generation"]
    cmd_control(args, cfg, (tok, model))
    d = run_dir(cfg)
    meta = start_entry(cfg, d, _expected(cfg))

    # FP16 greedy cache with head statistics recorded in the same pass
    apply_sparse(model, SparseConfig(method="none", record_entropy=True))
    with PowerMeter(poll_hz=float(cfg["cost"]["poll_hz"]), fallback_w_active=float(cfg["cost"].get("fallback_w_active", 70.0))) as pm:
        fp16 = run_generation(model, tok, problems, task, int(g["max_new_tokens"]), int(g["batch_size"]), topk=int(g.get("topk", 16)), desc="I1 fp16 greedy")
    from .interventions import sparsify as _sp

    stats = stats_frame(_sp.SPARSE.summary())
    apply_sparse(model, SparseConfig(method="none"))
    stats.to_csv(os.path.join(_cache_dir(), "head_stats.csv"), index=False)
    pd.DataFrame(fp16).to_parquet(_fp16_path(task), index=False)
    pw = pm.summary()
    lm = _fit_latency_model(fp16, model, cfg, pw["mean_w"] if np.isfinite(pw["mean_w"]) else float(cfg["cost"].get("fallback_w_active", 70.0)))

    # difficulty labels (pass@k) and cheap features
    dl = cfg["baseline"].get("difficulty", {})
    diff = pass_at_k_labels(model, tok, problems, task, k=int(dl.get("k", 4)), temperature=float(dl.get("temperature", 0.7)), max_new_tokens=int(g["max_new_tokens"]), batch_size=int(g["batch_size"]), seed=int(cfg.get("seed", 0)))
    feats = cheap_features(problems, fp16, tok)
    diff.merge(feats, on="id").to_parquet(os.path.join(_cache_dir(), f"difficulty_{task}.parquet"), index=False)

    pmeta = _meta_frame(cfg, problems)
    acc = accuracy_table(fp16, pmeta, None, n_boot=int(cfg["metrics"]["bootstrap_resamples"]))
    cost = {"energy_j_per_problem": pw["energy_j"] / len(fp16), "wall_s": pw["wall_s"], "tokens_per_s": sum(r["n_gen"] for r in fp16) / max(1e-9, pw["wall_s"]), "mean_w": pw["mean_w"], "energy_source": pw["energy_source"], "simulated": False}
    metrics = {"accuracy": acc, "cost": cost, "latency_model": lm, "difficulty_counts": diff["stratum"].value_counts().to_dict(), "n_heads_recorded": int(len(stats)), "acc_mean": acc["accuracy"]["mean"], "energy_j_per_problem": cost["energy_j_per_problem"]}
    meta["cell_name"] = "fp16"
    finish_entry(meta, metrics)
    print(json.dumps({k: v for k, v in metrics.items() if not isinstance(v, dict)}, indent=2))


def _prepare(base: Dict) -> Dict:
    base["logging"]["output_dir"] = results_root(base["logging"].get("output_dir", "results"))
    return base


def cmd_sweep(args):
    base = _prepare(load_yaml(args.config))
    for cell in iter_cells(base, minimum=args.minimum):
        cfg = resolve(base, cell)
        print(f"\n===== cell {cell_name(cell)}  hash {config_hash(cfg)} =====")
        run_cell(cfg, skip_existing=args.skip_existing)


def cmd_perhead(args):
    """I5: 50% single-head sparsification on every head of a 6-layer subset spanning low and high entropy."""
    from .interventions.sparsify import choose_layer_subset

    base = _prepare(load_yaml(args.config))
    stats_path = os.path.join(_cache_dir(), "head_stats.csv")
    stats = pd.read_csv(stats_path) if os.path.exists(stats_path) else pd.DataFrame()
    n_layers = int(base["model"].get("n_layers", 28))
    layers = base["perhead"].get("layers") or choose_layer_subset({i: r for i, r in stats.to_dict("index").items()}, n_layers, int(base["perhead"].get("n_layers", 6)))
    heads = base["perhead"].get("heads") or list(range(int(base["model"].get("n_heads", 12))))
    if args.minimum:
        heads = heads[: int(base["perhead"].get("heads_minimum", 4))]
    print("per-head sweep layers:", layers, "heads:", heads)
    for method in base["perhead"]["methods"]:
        for l in layers:
            for h in heads:
                cell = {"sparsifier": method, "keep_frac": float(base["perhead"].get("keep_frac", 0.5)), "layer": int(l), "head": int(h)}
                cfg = resolve(base, cell)
                print(f"\n===== cell {cell_name(cell)} =====")
                run_cell(cfg, skip_existing=args.skip_existing)


def cmd_frontier(args):
    """I7: build the selector table from every finished cell and run energy repeats on the headline configs."""
    from .analysis.figures import load_all_metrics
    from .harness.power import median_with_spread

    base = _prepare(load_yaml(args.config))
    rr = results_root()
    allm = load_all_metrics(rr)
    task = base["task"]["dataset"]
    diff = pd.read_parquet(os.path.join(_cache_dir(), f"difficulty_{task}.parquet"))
    rows = []
    for exp, lst in allm.items():
        for m in lst:
            gpath = os.path.join(m["_meta"]["run_dir"], "generations.parquet")
            if not os.path.exists(gpath) or "cost" not in m:
                continue
            gdf = pd.read_parquet(gpath)[["id", "correct", "n_gen"]]
            e_tok = m["cost"]["energy_j_per_problem"] / max(1e-9, gdf["n_gen"].mean())
            name = "fp16" if exp.startswith("I1") else f"{exp}:{m['_meta'].get('cell_name', '')}"
            gdf = gdf.assign(config=name, energy_j=gdf["n_gen"] * e_tok)
            rows.append(gdf)
    if rows:
        table = pd.concat(rows).merge(diff[["id", "q_tokens", "n_numbers", "q_chars", "first_entropy"]], on="id", how="left")
        table.to_parquet(os.path.join(_cache_dir(), "selector_table.parquet"), index=False)
        print("selector table:", table["config"].nunique(), "configs x", table["id"].nunique(), "problems")
    # energy repeats on headline configs (measured arms only)
    reps = int(base["cost"].get("repeats", 3))
    out = {}
    for cell in base.get("headline_cells", []):
        cfg = resolve(base, cell)
        es = []
        for r in range(reps):
            cfg_r = dict(cfg)
            cfg_r["_cell"] = {**cell, "repeat": r}
            m = run_cell(cfg_r)
            if m:
                es.append(m["cost"]["measured_energy_j_per_problem"])
        if es:
            out[cell_name(cell)] = median_with_spread(es)
    with open(os.path.join(_cache_dir(), "energy_repeats.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def cmd_analyze(args):
    from .analysis.figures import analyze

    analyze(args.results or results_root())


def cmd_smoke(args):
    from .smoke import run_smoke

    run_smoke(args.model)


def main(argv=None):
    hf_cache_dir()
    import_previous_results()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("make_subset", cmd_make_subset), ("baseline", cmd_baseline), ("control", cmd_control), ("frontier", cmd_frontier)):
        s = sub.add_parser(name); s.add_argument("--config", required=True); s.set_defaults(fn=fn)
    for name, fn in (("sweep", cmd_sweep), ("perhead", cmd_perhead)):
        s = sub.add_parser(name); s.add_argument("--config", required=True); s.add_argument("--minimum", action="store_true"); s.add_argument("--skip-existing", action="store_true"); s.set_defaults(fn=fn)
    s = sub.add_parser("analyze"); s.add_argument("--results", default=None); s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("smoke"); s.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct"); s.set_defaults(fn=cmd_smoke)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
