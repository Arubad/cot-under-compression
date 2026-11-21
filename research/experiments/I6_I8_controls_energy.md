# I6 — ARC-Challenge short-reasoning control · I8 — energy repeats

**Status:** ⏳ draft prediction (frozen at launch)
**Hypotheses served:** H1 (control), H3 (energy uncertainty) · **Theory:** [01 §4](../theory/01_persistence_model_H1.md), [03 §3](../theory/03_cost_model_H3.md)

## I6 question and prediction
If KV damage is about exposure (length), short-answer ARC should show a much smaller KV-INT4 gap than GSM8K at matched bits. **Prediction:** the ARC gap is ≤ ⅓ of the GSM8K gap for KV INT4 per-token, and roughly equal across benchmarks for NF4 weights. A large ARC gap under KV would point to damage at the prefill/question-reading stage, not along the chain.

## I8 question and prediction
How noisy is measured energy on a shared T4? **Prediction:** across 3 repeats, energy per problem spreads by ≤ 10 % when measured with the hardware energy counter, and by more with 10 Hz polling. Thermal throttling on the 70 W cap makes the first repeat of a cold session faster than later ones, so drop or flag the cold-start repeat.

## Design
`sweep --config configs/controls_math_arc.yaml --minimum` (I6, 200 ARC problems); `frontier` repeats (I8). Est. 0.5 h + ~1–2 h.

## Result / Interpretation / Next
_(after run)_
