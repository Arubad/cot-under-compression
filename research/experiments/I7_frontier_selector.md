# I7 — energy-per-correct-answer frontier and the per-problem selector

**Status:** 🔒 prediction frozen 2026-11-21 · ✅ minimum version run 2026-11-23 (q3) · H3 **refuted** (selector clause), preliminary
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

## 4. Result (2026-11-23; `frontier` in q3 at `16bd6f4`; `analyze` locally at `ef98c6d`)
**I8 energy repeats (NVML counter, 3 each):** reference **224.4 J/problem** (224.3–224.5, spread 0.1 %); NF4 **258.0 J** (257.7–258.0). The reference used for every cost ratio is now 224.4 J, replacing I1's polled 283 J.

**Frontier, primary (bandwidth) KV reference, energy per correct answer:**
| Config | Acc | Tokens | E/problem | **E/correct** | Measured? |
|---|---|---|---|---|---|
| Top-k heavy-hitter 50 % | 65.3 % | 0.99× | 218.6 J | **334.6 J** (best) | simulated |
| Sink + window 50 % | 64.7 % | 0.99× | 218.1 J | 337.3 J | simulated |
| KV INT8 | 65.0 % | 1.00× | 219.4 J | 337.6 J | simulated |
| Reference (fp32) | 64.7 % | 1.00× | 224.4 J | 347.0 J | measured |
| KV INT4 | 52.3 % | 0.93× | 202.9 J | 387.8 J | simulated |
| NF4 weights | 59.0 % | 0.87× | 258.0 J | 437.3 J | measured |
| KV INT2 (most aggressive) | 9.7 % | 1.51× | 347.8 J | 3598 J | simulated |

Eager KV reference (secondary): best is **KV INT8 at 232.0 J/correct** (−33 % vs the reference).

**Selector** (5-fold CV, τ = 0.8, cheap pre-generation features): 65.0 % at 223.9 J/problem, choosing the reference on 80 % of problems. The best fixed configuration at equal or lower energy (top-k 50 %) reaches 65.3 % at 218.6 J. **Gain −0.33 pt.** Oracle: 80.7 % at 174.0 J.

**Pre-registered H3 test (`test_h3`): refuted.** Clause 1 (best ≠ most aggressive): pass. Clause 2 (selector gain ≥ 1 pt): fails, and the gain is below 0.5 pt, which is the registered refutation.

| Prediction | Observed | |
|---|---|---|
| KV and sparse cells within 3 % of the reference per problem | 2.2–2.8 % below (INT2 excepted, 1.55×) | ✅ |
| Best ≠ most aggressive (INT2) | best is top-k 50 % | ✅ |
| NF4 best only if > 8 % below the reference per problem; otherwise reference or INT8 KV | NF4 15 % *above*; best is top-k 50 %, then sink+window, INT8 KV (all within 0.9 %) | ✅ (sparse not named) |
| Eager reading: INT8 KV best; verdict depends on the KV reference | INT8 KV best (232 J); clause 1 passes on both readings | ✅ / ❌ (no dependence) |
| Oracle gain ≥ 3 pt | +15.4 pt | ✅ |
| Selector gain 0–1.5 pt; H3 under-powered or refuted | −0.33 pt; refuted | ✅ |
| Measured NF4 saving below roofline (T3.3) | negative saving (−15 %); fp16-projected −120 % | ✅ |

## 5. Interpretation
- **On the primary reading the frontier is flat.** The three lossless cells beat the reference by 2.7–3.6 % per correct answer, well inside the accuracy CIs (±5 pt). No configuration buys a meaningful saving at these lengths, which is T3.1 by bytes. The "mixed configuration wins" wording of H3 has nothing to stand on.
- **On the eager reading INT8 KV saves a third.** On a T4 with eager attention, KV traffic is the expensive part of a decode step (LAT-01), so INT8 KV would be a real win *if* a kernel delivered the eager-cost saving. That is the one deployment-relevant saving in the study, and it is not measured (fake quant).
- **NF4 is a loss on this hardware:** more energy, less accuracy. bitsandbytes NF4 with fp32 compute on sm75 is dequant-bound.
- **The oracle headroom is mostly trajectory luck, not difficulty.** Greedy decoding under different perturbations solves different subsets of problems, so "pick whichever config was right" gains 15 pt. Pre-generation features (question length, number count, first-token entropy) cannot predict which perturbation will land, and the selector falls back to the reference. A selector that sees the generation (short-draft agreement, the full-version design) is the only version with a chance.
- **Caveat:** at minimum scale the selector test is preliminary (README §5). The refutation is what the registered rule says for these data, not a full-version verdict.

## 6. Next
The full-version selector with draft-agreement features. Q16 (new, PI): with the frontier flat on the primary reading, whether to headline the eager-reference frontier (measured KV cost on this hardware) as the deployment-relevant result, stated alongside the primary verdict as A8 requires.
