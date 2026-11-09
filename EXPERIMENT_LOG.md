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
