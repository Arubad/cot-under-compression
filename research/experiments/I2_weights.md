# I2 — weight quantisation (NF4 minimum; INT8 full)

**Status:** 🔒 prediction frozen 2026-11-21 · launch queued
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

## 4–6. Result / Interpretation / Next
_(after run)_
