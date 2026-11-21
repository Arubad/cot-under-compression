# I5 — single-head sparsification over a 6-layer subset

**Status:** 🔒 prediction frozen 2026-11-21 · launch queued
**Hypothesis served:** H2 · **Theory:** [02 (all)](../theory/02_heavy_attention_H2.md)

## 1. Question
Does a head's accuracy sensitivity to 50 % sparsification track its attention concentration?

## 2. Prediction (frozen 2026-11-21, before launch; draft bullets below are part of it)
**Subset fixed by the rule on I1's head statistics (computed 2026-11-21, before launch): layers [6, 9, 14, 16, 18, 27]**, the lowest-entropy layer (6, H̄ = 1.87) through the highest (27, H̄ = 4.37). Minimum version: heads 0–3 of each layer, so 24 heads, top-k heavy-hitter at 50 %.
- **Pass/fail on Δacc:** |ρ(entropy, Δacc)| < 0.3, so **H2 is predicted to be refuted or under-powered on the registered outcome** (T2.2 noise floor; 21 of 24 heads have |Δacc| ≤ 2 pt).
- **Secondary (A4):** ρ(entropy, mean KL) ≥ 0.4, and |ρ(top-10 % mass, KL)| is larger than |ρ(entropy, KL)|. That would show the premise holds on a continuous outcome, which raw accuracy cannot resolve at n = 300.

- **Layer-selection rule (fixed now, before I1 head statistics exist).** Sort layers by mean per-head entropy from I1 and take the 0, 20, 40, 60, 80 and 100th percentiles, ties broken toward the lower index. *Checked 2026-11-21:* `choose_layer_subset` in `src/interventions/sparsify.py` implements this rule (quantile picks over layers sorted by mean entropy). Its sort was made stable so that ties go to the lower index.
- Most of the 72 heads (6 × 12) show |Δacc| ≤ 1 pt, inside noise (T2.2). A few heads (≤ 10) show multi-point drops. These are expected to be heads with *low* top-k mass and high query dependence (large ε_lev − ε_topk).
- ρ(top-k mass, Δ) is stronger than ρ(entropy, Δ) in magnitude (T2.1).
- On raw Δacc, ρ is attenuated. The KL outcome gives |ρ| larger by ≥ 0.1.
- **Surprise.** A large drop in a low-entropy, high-mass head. That would contradict the bound (theory/02 §1) unless the value contrast is huge, so inspect that head's values before concluding anything.

## 3. Design
`perhead --config configs/perhead_sweep.yaml --minimum`. Est. ~3 GPU-h.

## 4–6. Result / Interpretation / Next
_(after run)_
