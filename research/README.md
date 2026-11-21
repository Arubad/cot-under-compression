# Research notebook

This directory is the *research* record of the study: what we believe and why, what each experiment is supposed to show before it runs, what it actually showed, and how that moved our beliefs. It sits on top of the frozen documents and does not replace them:

| Layer | File | Role | Mutability |
|---|---|---|---|
| Contract | [`../PREREGISTRATION.md`](../PREREGISTRATION.md) | Hypotheses and numeric pass/fail tests | Frozen; dated amendments only |
| Theory | [`theory/`](theory/) | Derivations that generate the predictions, with their assumptions and known gaps | Living; every change dated |
| Experiments | [`experiments/`](experiments/) | One page per experiment: question → prediction → design → result → interpretation | Prediction section frozen at launch |
| Run log | [`../EXPERIMENT_LOG.md`](../EXPERIMENT_LOG.md) | One entry per configuration run (commit, hash, hardware, observed) | Append-only |
| Decisions | [`decisions.md`](decisions.md) | Why the protocol is the way it is (platform, precision, library versions) | Append-only |
| Compute | [`compute_ledger.md`](compute_ledger.md) | GPU-hours spent against the budget | Append-only |
| Questions | [`open_questions.md`](open_questions.md) | Things that need an answer, and who owns them | Living |
| Progress | [`../ROADMAP.md`](../ROADMAP.md) | Milestone checklist | Tick boxes |

**Rule of the notebook.** A result is written up only against a prediction that was written down first. For every experiment, the *Prediction* section of its page is filled in and committed before the run is launched. The *Result* and *Interpretation* sections are filled in afterwards and never edit the prediction.

---

## Status board

_Last updated 2026-11-21._

### Hypotheses

| | Claim (short) | Primary statistic | Status | Evidence |
|---|---|---|---|---|
| **H3** (headline) | Energy-per-correct frontier is non-monotone; per-problem selector beats best fixed config by ≥ 1 pt | E/correct Pareto set; selector Δacc at ≤ energy | ⏳ not tested | — |
| **H1** | KV error is persistent and position-specific; weight error is not → KV first divergence moves earlier with fewer bits, KV gap grows with length within difficulty strata | median first-divergence token per KV bit-width | ⏳ not tested | — |
| **H2** | Single-head sparsification is safe where attention is concentrated | Spearman ρ(head entropy, Δacc) over 6-layer subset | ⏳ not tested | — |

Status values: ⏳ not tested · 🟡 preliminary (minimum-scale) · ✅ confirmed · ❌ refuted · ⚪ under-powered.

### Theory-derived predictions to watch

These come out of [`theory/`](theory/). They are *not* pre-registered tests, but each was written before any data exists, so they count as genuine predictions.

| ID | Prediction | Source | Status |
|---|---|---|---|
| T1.1 | Uniform b-bit quantisation noise σ ∝ 2^-b; if the divergence hazard is linear in σ, E[T_div] scales ≈ 1 : 1/17 : 1/85 for INT8 : INT4 : INT2 | [H1 theory §2–3](theory/01_persistence_model_H1.md) | ⏳ |
| T1.2 | Under greedy decoding, persistence alone does **not** make the KV hazard grow with t; a KV gap that grows with length needs τ_t to grow with t. Predicted shape: flat to *decreasing* hazard (attention dilution) unless anchors carry most of the error | [H1 theory §4](theory/01_persistence_model_H1.md) | ⏳ |
| T1.3 | INT8 KV will mostly not diverge within 512 tokens → first-divergence data are right-censored; medians need Kaplan–Meier | [H1 theory §5](theory/01_persistence_model_H1.md) | ⏳ |
| T2.1 | Single-head output error ≤ 2·(dropped attention mass)·max‖v‖ → top-k mass should predict Δ at least as well as entropy (\|ρ_mass\| ≥ \|ρ_H\|) | [H2 theory §2](theory/02_heavy_attention_H2.md) | ⚠️ at risk: top-10 % mass is saturated in mid layers (I1), so range restriction may defeat it |
| T2.2 | Single-head Δacc on 300 problems sits near the noise floor (SE ≈ 1.5–2.5 pt); ρ on raw accuracy will be attenuated, while a continuous outcome (KL, T_div) will not | [H2 theory §4](theory/02_heavy_attention_H2.md) | ⏳ |
| T3.1 | With GQA (2 KV heads) at GSM8K lengths, KV bytes are ~1–10 % of per-sequence decode bytes → KV quantisation and sparsity buy < 10 % energy; the frontier is shaped by the weight axis | [H3 theory §2](theory/03_cost_model_H3.md) | ❌ for time, ✅ for bytes (LAT-01): eager KV moves at 12 GB/s, so KV is 46 % of step time at B = 8 against 9 % of bytes → Q14 |
| T3.2 | A config beats the reference on E/correct iff e_c/e₀ < (a_c/a₀)·(L₀/L_c); token inflation of x % cancels x % of per-token savings | [H3 theory §3](theory/03_cost_model_H3.md) | ⏳ |
| T3.3 | bitsandbytes NF4 on T4 is dequant-bound, not bandwidth-bound → measured per-token energy may be *higher* than the roofline predicts, possibly higher than the reference | [H3 theory §2.3](theory/03_cost_model_H3.md) | ⏳ |

### Experiments

| ID | Question | Page | Status | Headline |
|---|---|---|---|---|
| SMOKE | Does every code path run, and does the zero-change control pass? | [experiments/SMOKE.md](experiments/SMOKE.md) | 🟡 control ✅; 3 bugs + cost model fixed; SMOKE-05 re-run pending | control bit-exact on Colab T4 |
| I1 | FP32 reference: accuracy, cache, difficulty labels, head statistics | [experiments/I1_baseline.md](experiments/I1_baseline.md) | ✅ done | **64.7 %** acc, 283 J/problem, control 50/50 bit-exact; tertile acc 84/63/47 % |
| DIAG | Decode cost model (LAT-01); KV per-token collapse (Q11) | [experiments/DIAG_latency_kv.md](experiments/DIAG_latency_kv.md) | ✅ done | r² 0.996 cost fit; INT4 per-token collapse is real, keys only, and removed by a 64-token fp tail |
| I2 | INT4 (NF4) / INT8 weights | [experiments/I2_weights.md](experiments/I2_weights.md) | ⏳ | — |
| I3 | KV INT8 / INT4 (/ INT2) per-token (/ per-channel) | [experiments/I3_kv.md](experiments/I3_kv.md) | ⏳ | — |
| I4 | Sparse attention: sink+window, top-k at 50 % | [experiments/I4_sparse.md](experiments/I4_sparse.md) | ⏳ | — |
| I5 | Per-head single-head sparsification, 6-layer subset | [experiments/I5_perhead.md](experiments/I5_perhead.md) | ⏳ | — |
| I6 | ARC-Challenge short-reasoning control | [experiments/I6_I8_controls_energy.md](experiments/I6_I8_controls_energy.md) | ⏳ | — |
| I7 | Frontier + per-problem selector | [experiments/I7_frontier_selector.md](experiments/I7_frontier_selector.md) | ⏳ | — |
| I8 | Energy repeats on headline configs | [experiments/I6_I8_controls_energy.md](experiments/I6_I8_controls_energy.md) | ⏳ | — |

### Current blockers
- **Before I2:** Q9 (weight-arm compute dtype) and Q1 (amendments), both PI decisions.
- **Before I3:** Q3 (censoring-aware primary H1 statistic) and **Q13 (per-token INT4 collapses as configured)**, both PI amendments. Q14 (KV cost reference) before any KV energy number.
- **Before I5:** Q4 (continuous outcomes, PI amendment).
- **Before I5:** Q12 (sink-share accumulator).
- See [`open_questions.md`](open_questions.md).

---

## How to add an experiment
1. Copy [`experiments/_TEMPLATE.md`](experiments/_TEMPLATE.md) to `experiments/<ID>_<slug>.md`.
2. Fill in *Question*, *Link to theory* and *Prediction* with numbers. Commit with message `research: <ID> prediction`.
3. Launch on Colab (see [`colab_runbook.md`](colab_runbook.md)), add the `EXPERIMENT_LOG.md` entry, and add the GPU-hours to the compute ledger.
4. Fill in *Result* and *Interpretation*, update the status board above, and tick the ROADMAP item.
