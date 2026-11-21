# I5 — single-head sparsification over a 6-layer subset

**Status:** ⏳ draft prediction (frozen at launch)
**Hypothesis served:** H2 · **Theory:** [02 (all)](../theory/02_heavy_attention_H2.md)

## 1. Question
Does a head's accuracy sensitivity to 50 % sparsification track its attention concentration?

## 2. Prediction (draft)
- **Layer-selection rule (fixed now, before I1 head statistics exist).** Sort layers by mean per-head entropy from I1 and take the 0, 20, 40, 60, 80 and 100th percentiles, ties broken toward the lower index. *Checked 2026-11-21:* `choose_layer_subset` in `src/interventions/sparsify.py` implements this rule (quantile picks over layers sorted by mean entropy). Its sort was made stable so that ties go to the lower index.
- Most of the 72 heads (6 × 12) show |Δacc| ≤ 1 pt, inside noise (T2.2). A few heads (≤ 10) show multi-point drops. These are expected to be heads with *low* top-k mass and high query dependence (large ε_lev − ε_topk).
- ρ(top-k mass, Δ) is stronger than ρ(entropy, Δ) in magnitude (T2.1).
- On raw Δacc, ρ is attenuated. The KL outcome gives |ρ| larger by ≥ 0.1.
- **Surprise.** A large drop in a low-entropy, high-mass head. That would contradict the bound (theory/02 §1) unless the value contrast is huge, so inspect that head's values before concluding anything.

## 3. Design
`perhead --config configs/perhead_sweep.yaml --minimum`. Est. ~3 GPU-h.

## 4–6. Result / Interpretation / Next
_(after run)_
