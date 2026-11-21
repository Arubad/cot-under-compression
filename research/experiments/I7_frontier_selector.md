# I7 — energy-per-correct-answer frontier and the per-problem selector

**Status:** 🔒 prediction frozen 2026-11-21 · launch queued
**Hypothesis served:** H3 (headline) · **Theory:** [03 (all)](../theory/03_cost_model_H3.md)

## 1. Question
Once token inflation is included, which configuration minimises energy per correct answer? Does a difficulty-aware per-problem selector beat the best fixed configuration?

## 2. Prediction (frozen 2026-11-21, before launch; draft bullets below are part of it)
**Frozen numbers (A8, A9).** On the primary (bandwidth) reading, every KV and sparse cell sits within 3 % of the reference's energy per problem, so their energy per correct answer is set by accuracy and token inflation. INT8 KV (if lossless) lands within ±3 % of the reference. NF4 is the best energy per correct answer if its measured energy per problem is more than 8 % below the reference's. Otherwise the reference or INT8 KV is best. Either way the best is **not** the most aggressive configuration (INT2 KV), so the first H3 clause passes. On the eager reading, INT8 KV is ~30 % cheaper and becomes the best configuration, so **the H3 verdict depends on the KV cost reference, and that dependence is itself the finding.** The selector gains < 1 pt over the best fixed configuration: **H3 is predicted under-powered or refuted on the selector clause.**

- **Frontier shape.** Non-monotone in aggressiveness: the most aggressive configuration (INT2 KV / 25 % keys / NF4) is not on the Pareto set, because its accuracy loss outweighs its small extra saving.
- **Optimum.** NF4 weights with reference-precision KV and no sparsity, or with INT8 KV if it is free in accuracy. The KV and sparse axes contribute < 10 % (T3.1). This matches H3's "not the most aggressive" clause but probably not its "mixed configuration" wording.
- **Selector.** The oracle gain over the best fixed configuration is ≥ 3 pt. The learned selector at n = 300 with 5-fold CV gains 0–1.5 pt with a CI covering the 1-pt threshold, so it is likely under-powered at minimum scale.
- **Measured versus simulated.** The measured NF4 energy saving is smaller than its roofline (T3.3).

## 3. Design
`frontier --config configs/frontier.yaml`. Reads every finished cell, plus the I8 energy repeats.

## 4–6. Result / Interpretation / Next
_(after run)_
