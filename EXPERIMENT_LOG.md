# Experiment log — cot-under-compression

Every configuration run gets an entry **before** launch (fill `Expected`) and is completed **after** it finishes (fill `Observed`, `Next change`). Append-only. The W&B project `cheap-tokens` mirrors entries, keyed by `config_hash`. Energy repeats are logged as separate lines under `Observed`. Progress at the milestone level is tracked in [`ROADMAP.md`](ROADMAP.md).

Config hash = first 12 hex characters of `sha256` over the resolved YAML (`python -m src.utils.config_hash configs/<file>.yaml`).

## Entry format

```
## YYYY-MM-DD — <experiment id> — <one-line title>
- Config: configs/<file>.yaml  (hash: <12 hex>)
- Git commit: <short sha>
- Model / task / n_problems:
- Hardware: <GPU, platform, driver>
- Wall-clock / GPU-h:
- Expected: <pre-registered prediction for this cell, with numbers>
- Observed: <accuracy ± CI overall and by tertile; median first-divergence; tokens/s; peak mem; energy J/problem (3 repeats); link to results/<run>/table.csv>
- Control status: <did the zero-change control reproduce FP16 bit-exactly on this commit?>
- Next change: <exactly one thing changed before the next run, and why>
```

---

## 2026-10-13 — TEMPLATE — I3 KV INT4 per-token, Qwen2.5-1.5B, GSM8K-300 (example entry, not a real run)

> **This entry is a template.** Placeholder values illustrate the required fields; nothing below was run.

- Config: configs/kv_quant_sweep_1p5b.yaml  (hash: `0000deadbeef`)
- Git commit: `abc1234`
- Model / task / n_problems: Qwen2.5-1.5B-Instruct / GSM8K stratified subset / 300
- Hardware: 1× T4 16 GB, Kaggle, driver _
- Wall-clock / GPU-h: 0.35 h
- Expected: overall accuracy a few points below FP16; long-tertile gap ≥ 2× short-tertile gap (H1); median first-divergence token lower than the INT8 run.
- Observed: _(to be filled) acc X ± CI (short Y, mid Z, long W); T_div median _; tokens/s _; peak mem _ GB; energy _ / _ / _ J per problem; results/I3_kv_int4_tok/table.csv_
- Control status: zero-change control on commit `abc1234`: 300/300 token sequences identical to FP16 cache, max |Δlogit| = _ — PASS/FAIL.
- Next change: _(to be filled) e.g. "none — proceed to INT4 per-channel" or "switch NVML polling 1 Hz → 10 Hz because energy repeats disagreed by > 15 %"._

---

_(real entries start below)_

## 2026-11-09 — SMOKE-01 — first smoke test on Kaggle T4: fp16 produces NaN logprobs

- Config: src/smoke.py defaults (dtype auto → fp16 on T4)
- Git commit: `db2dc4f`
- Model / task / n_problems: Qwen2.5-1.5B-Instruct / GSM8K / 12 control problems
- Hardware: Kaggle T4 x2, Internet On
- Wall-clock / GPU-h: < 0.1 h (download + 3 control passes)
- Expected: zero-change control bit-exact, fp16 accuracy in the 55–75 % range.
- Observed: `nan_in_logprobs: true`, `accuracy_fp16: 0.0`, control `passed: false`. Cache-path and sparse-path mismatches were both 0, so the three code paths agree with each other; the NaNs come from the fp16 forward pass itself (known Qwen2.5 fp16 overflow; T4 has no bf16).
- Control status: FAIL on `db2dc4f` (harness-level, not a sweep).
- Next change: switch `model.dtype` to `float32` in every config and in the smoke defaults. Decision for the pre-registered "fp16 vs fp32" question: **fp32 is the reference precision for all 1.5B experiments.** Consequence: measured latency/energy baselines are fp32, so the weight-quantisation arm (I2) compares INT8/NF4 against an fp32 rather than fp16 reference. Re-run the smoke test on the new commit.

## 2026-11-09 — SMOKE-02 — second smoke test (fp32): sparse attention path not bit-exact

- Config: src/smoke.py defaults (fp32)
- Git commit: `16d79df`
- Model / task / n_problems: Qwen2.5-1.5B-Instruct / GSM8K / 12 control problems
- Hardware: Kaggle T4 x2
- Observed: `nan_in_logprobs: false` (fp32 fix worked), `cache_path_mismatches: 0`, but `sparse_path_mismatches: 12/12`, `max_abs_logprob_diff_sparse_vs_eager: 14.03`. `accuracy_fp16: 0.17` on 12 problems with 64-token answers (not meaningful; smoke truncates CoT).
- Diagnosis: transformers' mask builder selects a mask function by `config._attn_implementation`. Our custom "sparse" implementation was registered with `AttentionInterface` but not `AttentionMaskInterface`, so it did not receive the eager float additive mask. Reproduced on CPU with a 2-layer random Qwen2 and left padding: the padded sequence diverges (new `tests/test_sparse_control_cpu.py`, fails on `16d79df`, passes after the fix).
- Control status: FAIL on `16d79df` (harness-level).
- Next change: register `eager_mask` for "sparse" in `AttentionMaskInterface`; `sparse_attention_forward` also normalises None / bool masks to float additive and prints the mask kind it receives once. Smoke prints torch/transformers versions. Re-run smoke.

## 2026-11-21 — SMOKE-03 — first Colab smoke test: zero-change control passes; NumPy-2 crash

- Config: src/smoke.py defaults (fp32)
- Git commit: `939eb32` (shipped as `git archive`)
- Model / task / n_problems: Qwen2.5-1.5B-Instruct / GSM8K / 30 (12 control)
- Hardware: Colab Tesla T4 15 GB, driver 580.82.07; Python 3.13.16, torch 2.14.1+cu130, transformers 5.22.1, NumPy 2
- Expected: control bit-exact on all three paths after the SMOKE-02 mask fix.
- Observed: `cache_path_mismatches: 0`, `sparse_path_mismatches: 0`, `max_abs_logprob_diff_sparse_vs_eager: 0.0`, no NaN. The sparse function receives a float mask (8, 1, 830, 830). Crashed in `PowerMeter.energy_j`: `np.trapz` removed in NumPy 2. NVML on the Colab T4 reports power (67 W against a 70 W cap) and a cumulative energy counter.
- Control status: **PASS** on `939eb32`.
- Next change: `np.trapezoid` fallback; re-run.

## 2026-11-21 — SMOKE-04 — smoke re-run: all cells run; diagnostics expose three harness bugs and a broken cost model

- Config: src/smoke.py defaults (fp32); git commit `939eb32` + NumPy-2 fix (uncommitted, shipped by hand)
- Hardware / stack: as SMOKE-03
- Expected: every cell runs or fails attributably.
- Observed: every generation cell ran; the final `analyze` crashed on `None` keys in `pareto.aggressiveness`. Per-cell diagnostics are in research/experiments/SMOKE.md. Single-head (L3 H0) and all-head top-k sparsification diverged at token 0 with acc 0 (pad-row NaN). `quanto`/`hqq` were identical to fake (the backend key was dropped by `resolve`). In-run latency fit: BW 0.27 GB/s, intercept −23 s, giving negative simulated energies. KV INT4 per-token T_div 1, per-channel T_div 17.5. LLM.int8 casts activations to fp16.
- Control status: PASS (as SMOKE-03).
- Next change: fixes and decisions D-004 / D-005 in research/decisions.md, with regression tests (22 tests pass). SMOKE-05 re-run at the fix commit after I1. Open PI questions Q9–Q11.

## 2026-11-21 — I1 — fp32 reference on the frozen GSM8K-300 subset

- Config: configs/baseline.yaml; git commit `30249b4`; subset frozen in the same commit (tertile boundaries 94 / 140)
- Hardware: Colab T4 (`cot-t4`), as SMOKE-03
- Expected: see research/experiments/I1_baseline.md §2 (accuracy 62–74 %, < 3 % truncated, strata easy 40–60 / medium 25–40 / hard 8–20 %, control bit-exact on 50).
- Observed: accuracy **64.7 %** [59.3, 70.0] on 300; 3/300 truncated; mean 137 generated tokens; strata easy 100 / medium 145 / hard 55 (greedy acc 95 / 63 / 15 %); tertile acc 84 / 63 / 47 %; control 0/50 mismatches on both paths, max |Δlogprob| 0.0; 67.0 W, **283 J/problem**, ≈130 ms/step at B = 8; top-10 % mass > 0.9 for 97 % of heads in layers 8–20, highest-entropy layers 27 and 1; in-run latency fit r² 0.04 (not identifiable, as D-004 expected). Config hash `90fe3aeb6018`; finished 10:03 UTC. Artefacts in `results/archive/I1_baseline/`.
- Control status: **PASS** (50/50).
- Next change: none needed for I1. Before I5, add a sink-share accumulator to the head recorder (Q12).
