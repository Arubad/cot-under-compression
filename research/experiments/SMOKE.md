# SMOKE — harness smoke test and zero-change control

**Status:** ✅ gate passed (SMOKE-05 at `be82d77`, 2026-11-21)
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
| SMOKE-05 | Colab T4, `be82d77` | ✅ | **All cells run, and the fixes are verified on GPU.** top-k 50 %: acc 0.167, T_div 62.5 (was 0). Single head L3 H0: acc 0.167, T_div 37.5 (was 0). KV per-token 0.0 / T_div 1 and per-channel 0.267 / 17.5, both unchanged and explained by DIAG B. `quanto` / `hqq` were **genuinely exercised this time** and are *unavailable*: `optimum-quanto` and `hqq` are not installed on Colab. LLM.int8 logs the fp32→fp16 cast (Q9). First launch OOM'd on GPU memory held by the Colab kernel (runbook gotcha) |
| SMOKE-04 | Colab T4, `939eb32` + NumPy-2 fix | ✅ (same as SMOKE-03) | All generation cells ran; analysis crashed on `None` keys. **Diagnostics exposed 3 harness bugs and a broken cost model** (D-004, D-005), all fixed and unit-tested before any sweep |

SMOKE-04 per-cell diagnostics (30 problems, 64 tokens, so accuracies are meaningless and only *relative* behaviour counts):

| Cell | acc | median T_div | E/problem | Verdict |
|---|---|---|---|---|
| reference | 0.133 | — | 118.7 J measured (66.8 W, 35.7 tok/s) | OK |
| INT8 weights (LLM.int8) | 0.10 | 9 | 105.1 J measured | runs; **activations cast to fp16** → Q9 |
| KV INT4 per-token (fake) | 0.0 | 1 | −150 J simulated | plausible (key outliers) → verify, Q11. Energy: broken fit, D-004 |
| KV INT4 per-channel (fake) | 0.267 | 17.5 | −172 J simulated | plausible |
| KV INT4 `quanto` / `hqq` | identical to fake | | | **bug:** backend key ignored (D-005.3); must be re-run |
| top-k 50 %, all heads | 0.0 | 0 | −77 J | **bug:** pad-row NaN (D-005.1) |
| leverage 50 %, single head L3 H0 | 0.0 | 0 | −77 J | **bug:** pad-row NaN (D-005.1). One of 336 heads cannot do this |
| in-run latency fit | BW 0.27 GB/s, c = −23 s, r² 0.67 | | | **not identifiable** at these lengths (theory/03 §2.1) → D-004 |

Side findings in SMOKE-03:
- The sparse function now receives a float additive mask `(8, 1, 830, 830)`. The 8-shot prompt plus a question is about 830 tokens.
- On the Colab T4, NVML exposes power (67 W while decoding, against a 70 W limit) and a hardware energy counter.

## 5. Interpretation
The controlled-measurement claim of the whole study rests on this gate. SMOKE-03 shows that, under the new stack, the instrumentation (custom cache, custom attention and entropy recorder) adds zero numerical change. Every later difference against the reference can therefore be attributed to the intervention, not to the harness.

The smoke run did its job twice over. Beyond confirming the control, the theory predictions served as a bug detector. "A single head out of 336 cannot cause divergence at token 0" (theory/02 §4) and "simulated energy cannot be negative" pointed straight at the NaN bug and at the cost model. Without those priors, both would have looked like findings.

## 6. Next
- **SMOKE-05** (short, after I1 frees the GPU): re-run the smoke test at the fix commit. Expected: single-head T_div well above 0, top-k 50 % close to the reference, `quanto`/`hqq` genuinely exercised, and then their availability on T4 recorded (REVIEW §6.2).
- Run `python -m src.run latency --config configs/baseline.yaml` once to produce the cost model.
- I1 was launched in parallel (it uses none of the buggy paths).
