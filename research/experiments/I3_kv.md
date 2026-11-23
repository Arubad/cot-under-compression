# I3 — KV-cache quantisation (INT8 / INT4 per-token; full version adds INT2 and per-channel)

**Status:** 🔒 prediction frozen 2026-11-21 · ✅ minimum version done 2026-11-23 (INT8 / INT4 / INT2 per-token) · H1 **refuted** at minimum scale
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

## 4. Result (INT8, INT4: q1 2026-11-21 at `de32b6c`; INT2: q3 2026-11-23 at `16bd6f4`; all Colab T4, per-token + 64-token tail, 300 problems)
| | INT8 (`d8a8fbada0c4`) | INT4 (`9a1044994f1f`) | INT2 (`4bd25fe3f765`) |
|---|---|---|---|
| Accuracy (ref 64.7 %) | 65.0 % [59.7, 70.3] | 52.3 % [47.0, 58.7] | **9.7 %** [6.3, 13.0] |
| Gap | −0.3 pt [−2.3, 1.7] | 12.3 pt [6.3, 17.7] | **55.0 pt** [49.3, 60.7] |
| Tertile gaps short / mid / long | 2 / −2 / −1 | 11 / 13 / 13 | 61 / 58 / 46 |
| **KM median T_div** | **+∞** [216, ∞], 76 % censored | **8** [7, 11], 2 % censored | **3** [2, 3], 0.3 % censored |
| Hazard per token | 2.15e-3 | 6.75e-2 | 1.78e-1 |
| Token count vs ref | 1.00× | 0.93× | **1.51×** (89/300 hit the 512 cap) |
| Measured energy (counter) | 250 J | 247 J | 415 J |
| Simulated energy, primary / eager (A8) | 277 / 190 J | 256 / 162 J | 439 / 254 J |

INT2 needed `16bd6f4`: its first run (q2) crashed with an OverflowError on a degenerate digit run in the answer extractor.

**Pre-registered H1 test** (`analysis.hypotheses.test_h1`, after the analysis fix `ef98c6d`): **refuted.**
- Monotone KM median with non-overlapping CIs: **pass** (+∞ > 8 > 3).
- KV INT4 long/short tertile gap ratio **1.18×**. The tertile-0 and tertile-2 gap CIs overlap, which is the registered refutation clause. Within strata: easy 4.29× (n = 14 in the long cell), medium **0.53×**, hard 1.15×. The ≥ 2× in every stratum clause fails.
- INT4-weight (NF4) tertile ratio 0.67× (< 1.5×: pass).
- β (secondary, three points) **1.02**. Two-point slopes 1.22 (INT8 → INT4) and 0.60 (INT4 → INT2), so INT2 saturates as predicted.

| Prediction | Observed | |
|---|---|---|
| INT8: ≥ 60 % censored, KM ∞, ±1.5 pt | 76 %, ∞, −0.3 pt | ✅ |
| INT4: KM 15–120, loss 2–10 pt, tokens 1.00–1.10× | KM 8, loss 12.3 pt, 0.93× | ❌ earlier, larger, shorter |
| INT2: KM < 10, loss ≥ 20 pt, collapse < 10 % plausible | KM 3, loss 55 pt, 9.7 % | ✅ |
| Monotone clause passes | passes | ✅ |
| Length clause fails; overall *inconclusive* | fails; overall **refuted** (the registered refutation clause fires on overlapping tertile CIs) | ✅ clause, ❌ label |
| β ≈ 1 (INT8 → INT4); 3-point β biased low | 1.22; three-point 1.02 | ✅ |
| Cost ratio 0.976 / 0.974 / 0.972 | 0.978 / 0.904 / 1.55 (token count enters) | ❌ for INT2 |

## 5. Interpretation
- **KV damage under CoT is early and flat in length, not cumulative.** At INT4 the trajectory leaves the reference by token ~8 and the gap does not grow with chain length within difficulty strata. This matches T1.2 (persistence alone gives a flat or decreasing hazard) and DIAG B (the damage is to recent keys; sinks barely matter). It contradicts the README §4 story that persistent, early anchor positions carry the error. The "most interesting possible outcome" of the pre-registration happened, but with recent-key precision, not sinks, as the explanation.
- **The INT2 decreasing tertile gap (61 → 46 pt) is a ceiling effect**, not evidence of a decreasing hazard: hard, long problems have less accuracy to lose (reference 47 % in the long tertile).
- **β ≈ 1 holds**: the hazard is about linear in the quantisation step σ until it saturates.
- **INT2 is the only cell with real token inflation** (1.51×, 30 % of problems hitting the cap): degenerate text loops. It makes INT2 the most *expensive* KV cell per problem as well as the least accurate.
- **First divergence is not a clean H1 contrast statistic** (see I2 §5 and Q15): NF4 weights diverge earlier (KM 5) than INT4 KV yet lose less accuracy.

## 6. Next
H1 is settled at minimum scale. For the write-up: Q15 (answer-level divergence statistic). The full version's per-channel cells (with the Q10b padding fix) test whether the flat-in-length result survives a scheme that protects key-outlier channels.
