# I2 — weight quantisation (NF4 minimum; INT8 full)

**Status:** 🔒 prediction frozen 2026-11-21 · ✅ run 2026-11-23 (q2)
**Hypotheses served:** H1 (weight arm of the contrast), H3 (the axis expected to shape the frontier) · **Theory:** [01 §3](../theory/01_persistence_model_H1.md), [03 §2.2–2.3](../theory/03_cost_model_H3.md)

## 1. Question
How much accuracy and how many extra tokens does 4-bit weight quantisation cost? What does it save *as measured* on a T4, compared with what the roofline says?

## 2. Prediction (frozen 2026-11-21, before launch; draft bullets below are part of it)
**Design as amended (A6):** NF4 with fp32 compute; INT8 is full-version only. Measured energy comes from the NVML counter (A5).
- **Accuracy:** NF4 64.7 → 59–64 % (loss 1–5 pt). Token inflation +0–10 %.
- **KM median T_div:** finite and early (20–80 tokens). Weight noise perturbs every step, including the first, so a "flat across bits" claim can only be tested in the full version, which has INT8. The minimum version tests only the tertile-ratio clause (< 1.5×), **predicted to pass**.
- **Speed and energy:** measured tokens/s 0.7–1.5× the reference's 32.4 tokens/s. Energy per problem 190–400 J against 283 J (I1, polled). With fp32 compute, NF4 dequantisation goes to fp32 and the GEMV stays fp32, so the saving is well below the 3.7× byte roofline (T3.3).

- **Accuracy.** NF4 loses 1–5 pt. The model is small (1.5B), and small models are less robust to 4-bit than the "near-lossless" results at 7B+. INT8 loses < 1 pt.
- **Token inflation.** NF4 mean generated tokens rise by +0–10 % (Lian et al. direction).
- **Divergence.** The NF4 hazard is flat in t and lower than INT4-KV per-token (the H1 contrast requires τ^W₄ ≪ τ^KV₄).
- **Cost.** The roofline predicts 3.7× fewer bytes than fp32 (T3), but measured bitsandbytes NF4 tokens/s is only 1–2.5× faster than fp32, because dequant overhead dominates at B = 8 on sm75 (T3.3). INT8 (LLM.int8) may be *slower* than fp32.
- **Surprise.** NF4 at the roofline speed-up. bitsandbytes kernels would then be better on T4 than assumed, which strengthens H3.

## 3. Design
`sweep --config configs/weights_sweep.yaml --minimum`. Weights are measured on NVML, not simulated, so the energy repeats (I8) apply here. Est. 0.5 GPU-h.

## 4. Result (2026-11-23, q2, commit `2ab4d66`, Colab T4 `cot-t4`, hash `10774f228ab3`)
| | Predicted | Observed | |
|---|---|---|---|
| Accuracy | 59–64 % (loss 1–5 pt) | **59.0 %** [53.7, 64.3], gap **5.7 pt** [0.3, 11.7] | ✅ at the edge |
| Token count | +0–10 % | **0.87×** (120 vs 137 tokens) | ❌ shorter, not longer (INT4 KV was also shorter, 0.93×) |
| KM median T_div | 20–80 tokens; hazard below INT4-KV | **5 tokens** [3, 6], 2 % censored; hazard 9.4e-2 vs 6.75e-2 for INT4-KV | ❌ earlier than INT4-KV |
| Tertile gap ratio (< 1.5×) | pass | gaps 9 / 2 / 6 pt, longest/shortest 0.67× | ✅ |
| tokens/s | 0.7–1.5× of 32.4 | 31.0 (0.96×) | ✅ |
| Energy | 190–400 J | **258 J/problem** (NVML counter; polled 258.4) | ✅ in range, but see below |

Mean KL to the reference before divergence 0.066; top-1 agreement over the reference's top-16 0.18. Peak memory 3.4 GB (fp32 reference 7.9 GB). Wall 19.3 min. **The GPU was thermally throttling during this cell** (SM clock 765–1020 MHz at 77–82 °C, against 1440 MHz earlier in the session), so the energy number is provisional until the I8 repeats.

## 5. Interpretation
- **First divergence is not damage.** NF4 leaves the reference trajectory *earlier* than INT4 KV (5 vs 8 tokens) yet loses less than half the accuracy (5.7 vs 12.3 pt). Weight noise flips near-ties at every step, so it paraphrases early and mostly harmlessly; INT4-KV divergences are fewer per token but more often fatal. This matters for H1: the pre-registered primary statistic (KM median T_div) measures *any* token change, so the KV-vs-weights contrast cannot be read off it. Accuracy gaps, or a divergence defined on the answer-relevant tokens, carry the contrast. Raise as an amendment question before the write-up, not a change to the frozen test.
- **No energy saving from NF4 on this T4.** Against the fp32 fake-quant cells measured with the same counter (247–250 J, I3), NF4's 258 J is no cheaper. Throttling may inflate it, but tokens/s is flat (0.96×), which matches T3.3 (bitsandbytes NF4 with fp32 compute is dequant-bound on sm75). If I8 confirms, NF4 is *worse* on energy per correct answer than the reference (≈ 437 J vs ≈ 383 J per correct on the counter), and the frontier's weight axis gives memory savings only.
- **Shorter chains, not longer.** Both compressed arms so far *shorten* the chain (0.87×, 0.93×), against the Lian et al. direction. Token inflation does not hurt either cell's cost here.

## 6. Next
I8 repeats of NF4 and the reference (queued in `frontier`) settle the energy question. Q15 (new): whether H1's contrast should also be reported on an answer-level statistic, given that T_div is dominated by harmless paraphrase under weight noise.
