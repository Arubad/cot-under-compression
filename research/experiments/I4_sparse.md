# I4 — attention sparsification (sink + window, top-k heavy-hitter at 50 %)

**Status:** 🔒 prediction frozen 2026-11-21 · ✅ run 2026-11-23 (q3)
**Hypotheses served:** H3 (frontier), context for H2 · **Theory:** [02 §1–3](../theory/02_heavy_attention_H2.md), [03 §2.1](../theory/03_cost_model_H3.md)

## 1. Question
How much accuracy does keeping 50 % of keys in every head cost? Which selection rule loses least?

## 2. Prediction (frozen 2026-11-21, before launch; draft bullets below are part of it)
**Frozen numbers:** top-k heavy-hitter 50 % loses ≤ 2 pt. Sink + window 50 % loses ≤ 3 pt. Primary cost ratio 0.984 (a 1.6 % saving); eager-reference 0.77. Both cells are simulated and labelled as such.

- **Top-k heavy-hitter 50 %.** < 2 pt loss. With ~830+ cached positions, the dropped mass at 50 % budget is tiny for nearly all heads (I1 head statistics will give the exact dropped-mass distribution, which bounds the error, theory/02 §1).
- **Sink + window 50 %.** The window keeps the most recent ~50 % of positions and so drops the *early* part of the prompt, which is mostly few-shot exemplars, not the target question. The question sits at the end of the prompt and stays inside the window for the whole generation. Prediction: small loss (< 3 pt), because the exemplars are droppable once the format is established. A larger loss would show that exemplars are re-read during the derivation, which is an interpretable finding.
- **Cost.** It is simulated by construction, and the saving is ≤ 4.5 % of decode bytes (T3.1).

## 3. Design
`sweep --config configs/sparse_sweep.yaml --minimum`. Accuracy is faithful; speed and energy are simulated and labelled as such. Est. ~1 GPU-h.

## 4. Result (2026-11-23, q3, commit `16bd6f4`, Colab T4 `cot-t4`, 300 problems)
| | Predicted | Sink + window 50 % | Top-k heavy-hitter 50 % |
|---|---|---|---|
| Accuracy (ref 64.7 %) | loss ≤ 3 / ≤ 2 pt | **64.7 %** [59.0, 70.0], gap 0.0 pt [−4.3, 4.3] ✅ | **65.3 %** [60.0, 70.7], gap −0.7 pt [−3.0, 1.7] ✅ |
| Tertile gaps | — | 0 / −1 / 1 pt | 0 / −2 / 0 pt |
| KM median T_div | — | **15** [10, 19], 10 % censored | **240** [203, ∞], 65 % censored |
| Hazard per token | — | 2.86e-2 | 3.44e-3 |
| Token count | — | 0.99× | 0.99× |
| Cost ratio primary / eager (simulated, A8) | 0.984 / 0.77 | 0.972 / 0.756 | 0.974 / 0.757 |
| Simulated energy primary / eager | — | 275 / 214 J | 276 / 215 J |

Measured wall-clock and NVML energy are recorded but are **not** the cost of these cells: masking in eager attention reads every key, so only the simulated numbers apply (2026-10-18 amendment).

## 5. Interpretation
- **Half the keys can go at no accuracy cost**, for both the oracle top-k and the deployable sink + window. Dropping the oldest half of the context (mostly few-shot exemplars) is harmless, so the exemplars are not re-read during the derivation.
- **Another case of divergence without damage.** Sink + window leaves the reference trajectory by token ~15 (only 10 % never diverge), yet loses 0.0 pt. This is the third arm (with NF4) where T_div and accuracy disagree, which strengthens Q15.
- **The saving is small in the primary cost model:** 2.6–2.8 % at GSM8K lengths, as T3.1 (bytes) predicted. On the eager reading it is 24 %. Both sparse cells are frontier points that cost nothing in accuracy, so on the primary reading H3's frontier is shaped by small savings everywhere.

## 6. Next
The 25 % budget and the leverage-score and plain sliding-window sparsifiers are full-version cells. Given 0 pt loss at 50 %, the 25 % budget is the informative next cell if GPU time allows.
