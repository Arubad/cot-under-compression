# src/ — module map

| Package | Files | What it does |
|---|---|---|
| `common/` | `config.py`, `seed.py`, `explog.py`, `stats.py`, `kaggle_env.py` | YAML configs + cell overrides + config hash; seeding; append-only experiment log; bootstrap / Spearman / linear fits; Kaggle paths and dtype choice (shared with the other two repos) |
| `harness/` | `data.py`, `prompting.py`, `generate.py`, `difficulty.py`, `power.py`, `control.py` | GSM8K / MATH-500 / ARC loaders, gold-length tertiles and the committed 300-problem stratified subset; the frozen 8-shot CoT template and answer extraction; greedy decoding with an on-the-fly top-16 logprob recorder (no full score tensor kept); pass@4 difficulty labels and cheap selector features; NVML power polling with a tokens/s fallback; the bit-exact zero-change control |
| `interventions/` | `weight_quant.py`, `kv_quant.py`, `sparsify.py` | bitsandbytes int8 / nf4 weights; `FakeQuantCache` (DynamicCache subclass: per-token / per-channel, sink protection, fp16 prefill then quantise; numpy reference `fake_quant_np` shared with the tests) and the HF `QuantizedCache` alternative; custom attention function registered through `AttentionInterface` with sliding-window, sink+window, top-k heavy-hitter and leverage (‖k‖²) masking on selected (layer, head) pairs, plus the per-head entropy / top-10%-mass recorder |
| `metrics/` | `accuracy.py`, `divergence.py`, `entropy.py` | accuracy with paired bootstrap, by tertile and by tertile within difficulty stratum, token inflation; first-divergence token and truncated top-k KL against the fp16 trajectory; head-statistics frames |
| `frontier/` | `pareto.py`, `selector.py` | Pareto sets, energy per correct answer, the "most aggressive" configuration, the decode-latency model (W_bytes + KV_bytes(t))/BW + c and its KV-exceeds-weights threshold, simulated energy; the difficulty-aware per-problem selector with 5-fold CV against the best fixed configuration and the oracle |
| `analysis/` | `hypotheses.py`, `figures.py` | the pre-registered H1 / H2 / H3 tests applied mechanically; Figures 1–4 and `hypothesis_report.json` |
| root | `run.py`, `smoke.py` | CLI (`make_subset`, `baseline`, `control`, `sweep`, `perhead`, `frontier`, `analyze`, `smoke`) |

Design notes:
- The fp16 greedy pass is cached once (`results/cache/fp16_gsm8k.parquet`) with top-16 logprobs per step; every configuration's first-divergence and KL are computed against it, and the same pass records per-head attention statistics and fits the latency model.
- KV quantisation is *fake-quant* by default: exact accuracy effect, no speed effect. Energy for those cells and for the sparse arm comes from the latency model and is flagged `simulated`; weight-quantised and fp16 cells carry measured NVML energy. Hollow markers in the frontier figure are simulated points.
- The zero-change control (`control.py`) runs three code paths — plain eager, the cache subclass at 16 bits, the custom attention function with no masking — and demands identical token sequences. It is re-run at the top of every sweep notebook.
