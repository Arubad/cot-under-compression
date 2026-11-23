# I6 — ARC-Challenge short-reasoning control · I8 — energy repeats

**Status:** 🔒 prediction frozen 2026-11-21 · ✅ I6 and I8 run 2026-11-23 (q3)
**Hypotheses served:** H1 (control), H3 (energy uncertainty) · **Theory:** [01 §4](../theory/01_persistence_model_H1.md), [03 §3](../theory/03_cost_model_H3.md)

## I6 question and prediction
If KV damage is about exposure (length), short-answer ARC should show a much smaller KV-INT4 gap than GSM8K at matched bits. **Prediction:** the ARC gap is ≤ ⅓ of the GSM8K gap for KV INT4 per-token, and roughly equal across benchmarks for NF4 weights. A large ARC gap under KV would point to damage at the prefill/question-reading stage, not along the chain.

**Frozen numbers (2026-11-21).** Cells: the reference, INT4 + tail per-token KV, and NF4 (fp32 compute), on ARC-Challenge test problems 0–199. The ARC reference accuracy is 65–85 %. KV INT4 gap on ARC ≤ max(1 pt, ⅓ × the GSM8K gap). NF4 gap on ARC within ±3 pt of its GSM8K gap.

## I8 question and prediction
How noisy is measured energy on a shared T4? **Prediction:** across 3 repeats, energy per problem spreads by ≤ 10 % when measured with the hardware energy counter, and by more with 10 Hz polling. Thermal throttling on the 70 W cap makes the first repeat of a cold session faster than later ones, so drop or flag the cold-start repeat.

**Frozen numbers (2026-11-21).** Headline cells are the reference and NF4 (INT8 dropped, A6), with 3 repeats each on GSM8K-300. Counter-based energy spread (max − min) / median ≤ 10 %, and the counter and polled integrals agree within 5 %. The reference median is 260–310 J/problem (I1 polled: 283 J).

## Design
`sweep --config configs/controls_math_arc.yaml --minimum` (I6, 200 ARC problems); `frontier` repeats (I8). Est. 0.5 h + ~1–2 h.

## I6 result (2026-11-23, q3, commit `16bd6f4`, ARC-Challenge test 0–199)
| | Predicted | Observed | |
|---|---|---|---|
| ARC reference accuracy | 65–85 % | **76.5 %** [70.5, 82.0]; mean answer 14.7 tokens | ✅ |
| KV INT4 + tail gap | ≤ max(1, ⅓ × 12.3) = 4.1 pt | **−1.5 pt** [−4.0, 1.0] (78.0 %); KM 16 [10, 27], 79 % censored | ✅ |
| NF4 gap | within ±3 pt of its GSM8K gap (5.7) | **5.5 pt** [1.5, 10.0] (71.0 %); KM 10 [8, ∞], 62 % censored; answers 7.4 tokens (0.50×) | ✅ |

## I6 interpretation
- **The benchmarks split the two arms exactly as predicted.** KV INT4 costs 12.3 pt on GSM8K and nothing on ARC, while NF4 costs ~5.5 pt on both. Weight damage does not depend on the task; KV damage does.
- **But the ARC KV control is confounded by the 64-token tail (A7), and it cannot separate "exposure" from "tail protection".** ARC answers are ~15 tokens, so every generated position, and the last ~50 prompt positions (the end of the question and the options), stay in full precision for the whole answer. DIAG B located the INT4 damage in *recent* keys. So a zero ARC gap is what both "damage needs a long chain" and "damage needs quantised recent keys" predict. It does not support exposure over the recent-key mechanism. A clean test is ARC with a shorter tail (e.g. 8 tokens), a full-version cell.
- **NF4 halves ARC answer length** (7.4 vs 14.7 tokens): the reference often writes a short justification, NF4 more often just the letter. Accuracy still drops 5.5 pt, so shorter answers are not harmless here.

## I8 result (q3, `16bd6f4`, NVML hardware counter, 300 problems each)
| Cell | Repeats (J/problem) | Median | Spread | tokens/s |
|---|---|---|---|---|
| Reference (fp32) | 224.5, 224.3, 224.4 | **224.4** | 0.11 % | 40.8 |
| NF4 (fp32 compute) | 257.7, 258.0, 258.0 | **258.0** | 0.13 % | 31.1 |

Prediction: spread ≤ 10 % with the counter ✅ (0.1 %). Accuracy is bit-identical across repeats (64.7 %, 59.0 %). No cold-start effect is visible: repeat 0 matches the others, so none is dropped.

**The reference energy moves from 283 J (I1, 10 Hz polling, another session) to 224.4 J.** The I1 number was 26 % high. Every simulated energy is rescaled to the new reference (the cost ratios are unchanged), and the frontier uses the repeat medians (A9). The NF4 single run of q2 (258.0 J, measured while throttling) matches its repeats, so the throttling did not bias it.
