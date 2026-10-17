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
