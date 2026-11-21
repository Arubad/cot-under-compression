# I3 — KV-cache quantisation (INT8 / INT4 per-token; full version adds INT2 and per-channel)

**Status:** 🔒 prediction frozen 2026-11-21 · launch queued
**Hypothesis served:** H1 (primary), H3 (frontier points) · **Theory:** [01 §1–5](../theory/01_persistence_model_H1.md), [03 §2.1](../theory/03_cost_model_H3.md)

## 1. Question
Does the first-divergence token move earlier as KV bits drop? Does the KV accuracy gap grow with chain length *within* difficulty strata, beyond what exposure alone predicts?

## 2. Prediction (frozen 2026-11-21, before launch; draft bullets below are part of it)
**Design as amended (A3, A7):** per-token INT8 / INT4 / INT2 with a 64-token full-precision tail, 300 problems. Primary statistic: KM median T_div.
- **INT8 + tail:** ≥ 60 % of trajectories are censored (never diverge), the KM median is +∞, and accuracy is within ±1.5 pt of the reference (64.7 %).
- **INT4 + tail:** DIAG B gives first-step KL 0.09 and 7/8 top-1 agreement, so ~10–20 % diverge at token 0. KM median T_div 15–120 tokens. Accuracy loss 2–10 pt. Token inflation 0–10 %.
- **INT2 + tail:** KM median < 10 tokens and accuracy loss ≥ 20 pt. A collapse to < 10 % accuracy would not be a surprise.
- **Monotonicity (pre-registered clause): predicted to pass.** +∞ > INT4 > INT2, with non-overlapping CIs.
- **Within-stratum tertile clause (≥ 2× in every stratum): predicted to fail or be under-powered.** Strata × tertile cells hold ~15–50 problems, and T1.2 predicts a flat-to-decreasing hazard. **Overall H1 verdict predicted: inconclusive** (monotone passes, the length clause does not).
- **β (secondary):** ≈ 1 when fitted INT8 → INT4. INT2 saturates (theory/01 §2), so the three-point β is biased low.
- **Cost (A8):** primary ratio 0.976 / 0.974 / 0.972 for INT8 / INT4 / INT2 (2.2–2.8 % saving at these lengths, computed from LAT-01 and the I1 lengths). Eager-reference ratio 0.68 / 0.62 / 0.60. Token inflation changes both.

- **Censoring (T1.3).** INT8 per-token: ≥ 50 % of problems never diverge within the generation, so the Kaplan–Meier median is "> T_end". INT4 per-token: most problems diverge, with a median T_div in the tens of tokens.
- **Scaling (T1.1).** Where both are uncensored, the ratio of hazards INT4/INT8 is about 17 if the hazard is linear in σ, and about 290 if quadratic. The measured ratio estimates β.
- **Accuracy.** INT8 is within 1 pt of the reference, with a CI covering 0. INT4 per-token loses 3–15 pt, because key outliers make per-token K ranges wide (KIVI). Per-channel INT4 (full version) loses less.
- **Hazard shape (T1.2).** Per-token: flat or slightly decreasing in t.
- **Cost.** Fake-quant gives a measured saving of 0. Real INT4 would save ≤ 8 % of decode bytes (T3.1).
- **Surprise.** INT8 KV diverges early on most problems. That would point to a quantiser bug, such as range computed over the wrong axis, before any mechanism. Check reconstruction MSE against the theoretical Δ²/12.

**Pre-registered test fed:** median first-divergence decreases monotonically INT8 → INT4 → INT2 with non-overlapping CIs; long-tertile gap ≥ 2× short-tertile gap within each stratum; INT4-weight gap differs by < 1.5× across tertiles.

## 3. Design
`sweep --config configs/kv_sweep.yaml --minimum`. Reuses the I1 reference cache. Est. ~1 GPU-h minimum.

## 4–6. Result / Interpretation / Next
_(after run)_
