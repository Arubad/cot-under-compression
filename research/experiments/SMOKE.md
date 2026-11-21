# SMOKE — harness smoke test and zero-change control

**Status:** 🔄 SMOKE-04 running on Colab T4 (`cot-t4`)
**Hypothesis served:** none, this is a harness gate · **Runs:** `EXPERIMENT_LOG.md` SMOKE-01 … SMOKE-04

## 1. Question
Does every code path run end to end on the target hardware? Does the expected-to-fail control pass, meaning do all three code paths reproduce the reference token-for-token?

## 2. Prediction
The control is bit-exact on every path (cache path at 16 bits, sparse function at keep = 1.0), with no NaNs in fp32. Every smoke cell either runs or fails with a recorded, attributable error. Accuracy figures are meaningless here because answers are truncated at 64 tokens.

## 3. Design
`python -m src.run smoke`: 30 GSM8K problems, 64 new tokens, fp32, eager attention. The control runs on 12 problems. Cells: KV INT4 per-token, KV INT4 per-channel, top-k 50 %, single head (L3, H0) leverage 50 %, INT8 weights, plus the HF `quanto` / `hqq` backends.

## 4. Result

| Run | Platform / commit | Control | Outcome |
|---|---|---|---|
| SMOKE-01 | Kaggle, `db2dc4f`, fp16 | ❌ NaN logprobs | fp16 overflow → D-001 (fp32) |
| SMOKE-02 | Kaggle, `16d79df`, fp32 | ❌ sparse path 12/12 mismatched | mask not registered for the custom attention → fixed in `939eb32` |
| SMOKE-03 | Colab T4, `939eb32`, fp32, transformers 5.22.1 | ✅ **0/12 cache, 0/12 sparse mismatches, max \|Δlogprob\| = 0.0** | crashed after I1-smoke generation: `np.trapz` removed in NumPy 2 → fixed |
| SMOKE-04 | Colab T4, `939eb32` + NumPy-2 fix | _running_ | |

Side findings in SMOKE-03:
- The sparse function now receives a float additive mask `(8, 1, 830, 830)`. The 8-shot prompt plus a question is about 830 tokens.
- On the Colab T4, NVML exposes power (67 W while decoding, against a 70 W limit) and a hardware energy counter.

## 5. Interpretation
The controlled-measurement claim of the whole study rests on this gate. SMOKE-03 shows that, under the new stack, the instrumentation (custom cache, custom attention and entropy recorder) adds zero numerical change. Every later difference against the reference can therefore be attributed to the intervention, not to the harness.

## 6. Next
When SMOKE-04 completes: record the backend availability (`quanto`, `hqq`, bitsandbytes INT8) and per-cell timing, then launch I1.
