# I4 — attention sparsification (sink + window, top-k heavy-hitter at 50 %)

**Status:** 🔒 prediction frozen 2026-11-21 · launch queued
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

## 4–6. Result / Interpretation / Next
_(after run)_
