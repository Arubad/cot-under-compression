"""numpy/scipy-only tests; run with  python3 -I tests/test_numpy_only.py"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from src.harness.prompting import extract_answer, is_correct, gsm8k_prompt, GSM8K_EXEMPLARS
from src.harness.data import parse_gsm8k_gold, normalise_number, tertile_boundaries, assign_tertile, stratified_subset
from src.harness.difficulty import stratum_from_hits
from src.metrics.divergence import first_divergence, topk_kl, trajectory_divergence
from src.metrics.accuracy import accuracy_table, tertile_gap_ratio, within_stratum_gap_ratios
from src.interventions.kv_quant import fake_quant_np, quantise_block_np, kv_bytes
from src.interventions.sparsify import keep_mask_np, SparseConfig, choose_layer_subset
from src.frontier.pareto import frontier_table, pareto_front, h3_frontier_verdict, fit_latency_model, energy_per_correct
from src.frontier.selector import evaluate_selector, h3_selector_verdict
from src.common.config import resolve, parse_cell, iter_cells


def test_answer_extraction():
    assert extract_answer("so 3 + 4 = 7. The answer is 7.") == "7"
    assert extract_answer("The answer is $1,250.") == "1250"
    assert extract_answer("The answer is -3.50") == "-3.5"
    assert extract_answer("we get 12 then 15 and finally 18") == "18"
    assert extract_answer("The answer is C.", task="arc") == "C"
    assert extract_answer("", task="gsm8k") is None
    assert is_correct("7.0", "7") and not is_correct("8", "7") and is_correct("C", "c", "arc")
    assert parse_gsm8k_gold("blah\n#### 1,000") == "1000"
    assert normalise_number("$18.") == "18"
    assert len(GSM8K_EXEMPLARS) == 8 and gsm8k_prompt("Q?").count("Question:") == 9
    for e in GSM8K_EXEMPLARS:  # every exemplar's stated answer must match its own arithmetic
        assert extract_answer(e["a"]) is not None


def test_tertiles_and_strata():
    lens = np.arange(1, 301)
    b = tertile_boundaries(lens)
    t = assign_tertile(lens, b)
    assert set(t) == {0, 1, 2} and abs(np.bincount(t) - 100).max() <= 1
    ids = [f"p{i}" for i in range(300)]
    sub = stratified_subset(ids, t, 90, seed=0)
    assert len(sub) == 90 and len(set(sub)) == 90
    tt = {i: t[int(i[1:])] for i in sub}
    assert sorted(np.bincount(list(tt.values()))) == [30, 30, 30]
    assert list(stratum_from_hits(np.array([4, 2, 0, 1]), 4)) == ["easy", "medium", "hard", "medium"]


def test_first_divergence_and_kl():
    assert first_divergence([1, 2, 3], [1, 2, 3]) == -1
    assert first_divergence([1, 2, 4], [1, 2, 3]) == 2
    assert first_divergence([1, 2], [1, 2, 3]) == 2
    ids = np.array([5, 6, 7]); lp = np.log(np.array([0.5, 0.3, 0.1]))
    assert topk_kl(ids, lp, ids, lp) < 1e-9
    lp2 = np.log(np.array([0.3, 0.5, 0.1]))
    assert topk_kl(ids, lp, ids, lp2) > 0.05
    rec = {"gen_tokens": [1, 2, 9], "topk_ids": [ids.tolist()] * 3, "topk_logprobs": [lp.tolist()] * 3}
    ref = {"gen_tokens": [1, 2, 3], "topk_ids": [ids.tolist()] * 3, "topk_logprobs": [lp.tolist()] * 3}
    d = trajectory_divergence(rec, ref)
    assert d["first_divergence"] == 2 and d["diverged"] and abs(d["argmax_agreement"] - 2 / 3) < 1e-9


def test_fake_quant():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 64, 128)).astype(np.float32)
    for bits in (8, 4, 2):
        for scheme in ("per_token", "per_channel"):
            for mode in ("asym", "sym"):
                y = fake_quant_np(x, bits, axis=-1 if scheme == "per_token" else -2, mode=mode)
                err = np.abs(y - x).mean()
                assert y.shape == x.shape and np.isfinite(y).all()
                if bits == 8:
                    assert err < 0.02, (bits, scheme, mode, err)
                if bits == 2:
                    assert err > 0.1
    assert np.array_equal(fake_quant_np(x, 16, -1), x)
    e8 = np.abs(quantise_block_np(x, 8, "per_token") - x).mean(); e4 = np.abs(quantise_block_np(x, 4, "per_token") - x).mean(); e2 = np.abs(quantise_block_np(x, 2, "per_token") - x).mean()
    assert e8 < e4 < e2
    assert kv_bytes(28, 2, 128, 1000, 16) == 2 * 28 * 2 * 128 * 1000 * 2


def test_keep_masks():
    q, k = 1, 20
    valid = np.ones((q, k), dtype=bool)
    scores = np.random.default_rng(1).normal(size=(q, k))
    for m in ("sliding_window", "sink_window", "topk_heavy_hitter", "leverage"):
        keep = keep_mask_np(scores, valid, m, 0.5, key_norm_sq=np.abs(scores[0]), sink_tokens=4)
        n = keep.sum()
        assert 10 <= n <= 14, (m, n)
        if m == "sink_window":
            assert keep[0, :4].all()
        if m == "sliding_window":
            assert keep[0, -10:].all() and not keep[0, :10].any()
        if m == "topk_heavy_hitter":
            assert keep[0, np.argmax(scores[0])]
    assert keep_mask_np(scores, valid, "none", 0.5).all()
    # causal prefill: later queries see more keys
    valid2 = np.tril(np.ones((k, k), dtype=bool))
    keep2 = keep_mask_np(np.where(valid2, 0.0, -np.inf), valid2, "topk_heavy_hitter", 0.5)
    assert (keep2.sum(1) == np.maximum(1, np.ceil(0.5 * np.arange(1, k + 1)))).all()
    cfg = SparseConfig(method="topk_heavy_hitter", keep_frac=0.5, layers=[1, 2], heads=3)
    assert cfg.active and cfg.layer_set() == {1, 2} and cfg.head_set() == {3}
    assert not SparseConfig(method="none").active
    stats = {f"{l}.{h}": {"layer": l, "head": h, "entropy": l * 0.1 + h * 0.001, "topk_mass": 0.5, "n": 1} for l in range(28) for h in range(12)}
    layers = choose_layer_subset(stats, 28, 6)
    assert len(layers) == 6 and layers[0] == 0 and layers[-1] == 27


def test_accuracy_table_and_gaps():
    ids = [f"p{i}" for i in range(60)]
    meta = pd.DataFrame({"id": ids, "tertile": [i % 3 for i in range(60)], "stratum": ["easy" if i < 30 else "hard" for i in range(60)]})
    fp16 = [{"id": i, "correct": True, "n_gen": 100} for i in ids]
    # config loses only on the long tertile
    recs = [{"id": i, "correct": not (int(i[1:]) % 3 == 2 and int(i[1:]) % 2 == 0), "n_gen": 120} for i in ids]
    acc = accuracy_table(recs, meta, fp16, n_boot=200)
    assert abs(acc["token_inflation"] - 1.2) < 1e-9
    assert acc["by_tertile"][2]["gap_to_fp16"]["mean"] > 0 and acc["by_tertile"][0]["gap_to_fp16"]["mean"] == 0
    assert tertile_gap_ratio(acc) == float("inf")
    r = within_stratum_gap_ratios(acc)
    assert all(v == float("inf") for v in r.values())


def test_frontier_and_selector():
    rows = [
        {"name": "fp16", "accuracy": 0.70, "energy_j_per_problem": 100, "mean_gen_tokens": 200, "simulated": False, "weight_bits": 16, "kv_bits": 16, "keep_frac": 1.0},
        {"name": "w4", "accuracy": 0.68, "energy_j_per_problem": 60, "mean_gen_tokens": 210, "simulated": False, "weight_bits": 4, "kv_bits": 16, "keep_frac": 1.0},
        {"name": "w4kv4", "accuracy": 0.60, "energy_j_per_problem": 55, "mean_gen_tokens": 260, "simulated": True, "weight_bits": 4, "kv_bits": 4, "keep_frac": 1.0},
        {"name": "w4kv2", "accuracy": 0.20, "energy_j_per_problem": 50, "mean_gen_tokens": 400, "simulated": True, "weight_bits": 4, "kv_bits": 2, "keep_frac": 1.0},
    ]
    ft = frontier_table(rows)
    assert ft.loc[ft["name"] == "w4", "best_energy_per_correct"].item()
    v = h3_frontier_verdict(ft)
    assert v["most_aggressive_config"] == "w4kv2" and not v["best_is_most_aggressive"]
    assert set(ft[ft["pareto_energy_vs_acc"]]["name"]) == {"fp16", "w4", "w4kv4", "w4kv2"}
    assert energy_per_correct(100, 0.5) == 200
    # latency model: synthetic data generated from the model itself
    W, kvt, bw, c = 3e9, 28 * 2 * 128 * 2 * 2, 300e9, 2e-3
    seq = np.linspace(100, 2000, 20)
    per_tok = (W + kvt * seq) / bw + c
    fit = fit_latency_model(seq, per_tok, W, kvt)
    assert abs(fit["bandwidth_bytes_per_s"] / bw - 1) < 1e-6 and abs(fit["overhead_s"] - c) < 1e-9 and fit["r2"] > 0.999
    # selector: cheap config correct exactly on easy problems, which the features reveal
    rng = np.random.default_rng(0)
    n = 300
    ids = [f"p{i}" for i in range(n)]
    easy = rng.random(n) < 0.5
    feats = pd.DataFrame({"id": ids, "q_tokens": np.where(easy, 30, 80) + rng.normal(0, 3, n), "n_numbers": np.where(easy, 2, 6), "q_chars": np.where(easy, 150, 400), "first_entropy": rng.random(n)})
    tab = []
    for i, iid in enumerate(ids):
        tab.append({"id": iid, "config": "fp16", "correct": 1.0, "energy_j": 100.0})
        tab.append({"id": iid, "config": "cheap", "correct": float(easy[i]), "energy_j": 40.0})
    table = pd.DataFrame(tab).merge(feats, on="id")
    res = evaluate_selector(table, tau=0.8)
    assert res["selector"]["accuracy"] > 0.95 and res["selector"]["energy_j"] < 90
    assert res["oracle"]["accuracy"] == 1.0
    assert h3_selector_verdict(res)["verdict"] in ("confirmed", "under-powered", "refuted")


def test_config_cells():
    base = {"experiment_id": "X", "intervention": {"kv_bits": 16, "kv_scheme": "per_token", "sparsifier": "none", "keep_frac": 1.0}, "sweep": {"kv_bits": [8, 4], "kv_scheme": ["per_token", "per_channel"]}, "sweep_minimum": {"kv_bits": [4], "kv_scheme": ["per_token"]}}
    assert len(list(iter_cells(base))) == 4 and len(list(iter_cells(base, minimum=True))) == 1
    c = resolve(base, parse_cell("kv_bits=4,sparsifier=leverage,keep_frac=0.5,layer=3,head=7"))
    i = c["intervention"]
    assert i["kv_bits"] == 4 and i["sparsifier"] == "leverage" and i["keep_frac"] == 0.5 and i["layer"] == 3 and i["head"] == 7


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
