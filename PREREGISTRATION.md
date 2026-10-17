# Pre-registration — Cheaper tokens, worse thoughts?

**Project:** `cot-under-compression` (local working name: `cheap-tokens-worse-thoughts`)
**Author:** Arush Badhe
**Date frozen:** 2026-10-13
**Status:** Frozen before the first sweep. Not edited after the freezing commit; changes go in the dated "Amendments" appendix.

## 1. Problem statement

Quantify and explain how weight quantisation, KV-cache quantisation and attention sparsification each degrade chain-of-thought reasoning as a function of reasoning length, and place them on an accuracy-per-second and accuracy-per-joule frontier.

## 2. Research question

How do weight quantisation, KV-cache quantisation and attention sparsification each affect chain-of-thought reasoning accuracy as a function of reasoning length, what mechanism explains the differences, and what is the resulting accuracy-per-second and accuracy-per-joule Pareto frontier on a single consumer GPU (and, if available, a Jetson)?

## 3. Hypotheses and pre-specified numeric tests

### H1 — Cache errors accumulate, weight errors do not
- **Confirmed if:** the GSM8K accuracy gap (FP16 − INT4 KV) in the longest-solution tertile is ≥ 2× the gap in the shortest tertile with non-overlapping bootstrap CIs, and median first-divergence token decreases monotonically INT8 → INT4 → INT2, while the INT4-weight gap differs by < 1.5× across tertiles.
- **Refuted if:** KV gaps are within CI across tertiles.

### H2 — Sparsification is safe where attention is concentrated
- **Confirmed if:** Spearman correlation between a layer's attention entropy and its accuracy drop under 50 % single-layer sparsification is ≥ 0.6 (and ≥ 0.5 for top-k mass in the opposite direction).
- **Refuted if:** |ρ| < 0.3.

### H3 — The frontier is non-monotone and selectable
- **Confirmed if:** the accuracy-per-joule Pareto-optimal configuration is not the most aggressive one and a difficulty-aware per-problem selector beats the best fixed configuration by ≥ 1 accuracy point at equal or lower energy.
- **Refuted if:** the most aggressive configuration Pareto-dominates or the selector gains < 0.5 points.

### Pre-specified secondary analyses (not pass/fail)
- Fit of E[T_div] against 1/σ² (σ² estimated from the quantiser's measured reconstruction error on cached K/V); per-token vs per-channel slope comparison.
- Fit of the latency model latency(t) ≈ (W_bytes + KV_bytes(t))/BW + c and the sequence-length threshold where KV bytes exceed weight bytes, per model.
- Sink protection ablation (first 4 cached positions kept at FP16) if H1 is refuted.
- Self-consistency at equal energy (I7): accuracy of k cheap samples with majority vote vs 1 FP16 sample at matched joules.

## 4. Expected-to-fail control

**Zero-change control:** running the harness with quantisation enabled at full precision (weights FP16, KV FP16, no sparsification, through the same code path) must reproduce the plain FP16 run **bit-exactly** — identical token sequences on all problems and logits identical to within fp16 rounding. The control is executed on every harness commit before any sweep result is logged; a mismatch halts sweeps.

## 5. Models, data and fixed settings

| Item | Value |
|---|---|
| Primary model | Qwen2.5-1.5B-Instruct (28 layers) |
| Secondary | Qwen2.5-3B-Instruct; Llama-3.2-3B-Instruct; 7B in 4-bit as a stretch |
| Fallback | Qwen2.5-0.5B-Instruct or SmolLM-1.7B if 1.5B is too slow |
| Primary task | GSM8K test (1,319); minimum version uses a fixed 300-problem subset stratified by gold-solution-length tertile (subset indices committed in `configs/gsm8k_300_stratified.json` before the first sweep) |
| Secondary tasks | MATH-500 subset; ARC-Challenge (short-reasoning control, 200 problems in minimum version); long-context retrieval probe |
| Prompt | fixed 8-shot CoT template, identical across configurations |
| Decoding | greedy; max 512 new tokens; batch size fixed per model |
| Weight precision | FP16, INT8, INT4 (GPTQ/AWQ-style) |
| KV precision × scheme | FP16, INT8, INT4, INT2 × per-token, per-channel; prefill at FP16, cache quantised after prefill |
| Sparsification | none; sliding window; sink + window; top-k heavy-hitter; leverage-score sampling × {50 %, 25 % keys kept} |
| Per-layer sweep | all 28 layers at 50 % (6-layer subset spanning low and high entropy in the minimum version) |
| Energy | NVML polling at 10 Hz; median of 3 repeats; tegrastats on Jetson; tokens/s × published power model as fallback |
| Seeds | n/a for greedy runs; {0, 1, 2} for sampled self-consistency only |

## 6. Statistics

- 1,000-resample bootstrap over problems for every accuracy and gap, paired across configurations on the same problems.
- Tertiles defined by gold-solution token length on the FP16 tokenizer, boundaries fixed before the first sweep.
- Spearman ρ for H2 over all swept layers (reported as preliminary when < 28).
- Energy figures use medians; CIs from the three repeats are shown but not used for pass/fail.
- Relative degradation (config minus FP16 on the same problems) is the headline quantity.

## 7. What will be reported regardless of outcome

The frontier figure and the harness table for every configuration run, plus a negative-results section marking each hypothesis confirmed / refuted / under-powered with the mechanism now believed.

## 8. Amendments

Format: `YYYY-MM-DD — what changed — why — git commit`. All amendments below were made **before any experiment was run** (Week 0), following an independent review recorded in [`REVIEW.md`](REVIEW.md).

- 2026-10-18 — **H3 promoted to the headline hypothesis and the cost unit changed to energy per correct answer.** The cost model now multiplies per-token energy by the *measured* number of generated tokens per configuration (token inflation) and divides by correct answers. Test text unchanged except that "accuracy-per-joule" reads "energy-per-correct-answer". — Why: Lian et al. (arXiv 2606.25519) and Lotfi et al. (arXiv 2606.00206) show low-bit quantisation lengthens chains, so per-token energy understates the true cost; The Quantization Trap (arXiv 2602.13595) partly pre-empts the non-monotone frontier, but the per-problem selector has no prior art and is the strongest contribution. — commit: this one.
- 2026-10-18 — **H1 reframed as a controlled measurement; first-divergence position made the primary statistic; tertile test run within difficulty strata.** New confirm clause: median first-divergence token decreases monotonically INT8 → INT4 → INT2 for KV with non-overlapping CIs, AND the longest-tertile gap is ≥ 2× the shortest-tertile gap *within each difficulty stratum* (difficulty = FP16 pass@k over 5 sampled decodes, labelled in I1 before any sweep), AND the INT4-weight gap differs by < 1.5× across tertiles. Refuted if KV first-divergence positions are within CI across bit-widths or KV gaps are within CI across tertiles. — Why: PM-KVQ (arXiv 2505.18610) and KVarN (arXiv 2606.03458) already argue KV error accumulates over long CoT, so the contribution is the measurement, not the claim; and on GSM8K longer gold solutions are harder problems, which confounds the tertile test. — commit: this one.
- 2026-10-18 — **Mechanism rewritten from "accumulation" to "persistence".** The original derivation summed per-position noise (σ√t or σt); the attention output is a softmax-weighted average, so value-noise variance scales as σ² Σ a², which shrinks as attention spreads. The compounding that matters is autoregressive and applies to weight quantisation too. The asymmetry now tested is persistent position-specific KV error versus step-independent weight error. — Why: the old derivation predicted the KV-vs-weight contrast for the wrong reason. — commit: this one.
- 2026-10-18 — **H2 moved from per-layer to per-head statistics; per-layer sweep replaced by a per-head sweep over a 6-layer subset spanning low and high entropy.** Test thresholds unchanged (ρ ≥ 0.6 entropy, ≥ 0.5 top-k mass; refuted if |ρ| < 0.3). — Why: EntroKV (ICML 2026) and DuoAttention (arXiv 2410.10819) show retrieval and streaming heads coexist within a layer, so a layer average hides the signal; the 28-layer sweep was the most expensive arm for the least information. — commit: this one.
- 2026-10-18 — **Sparse-arm speed and energy declared simulated.** On a T4 there is no FlashAttention; sparsification is top-k masking in eager/SDPA attention, which measures accuracy but not speed. All sparse-arm latency/energy numbers come from the latency model and are labelled simulated. — Why: honest reporting. — commit: this one.
- 2026-10-18 — **Jetson replication (I9), 7B-in-4-bit (I8) and COAT-vs-per-channel moved to stretch.** — Why: no hardware for the first, budget for the second, and the third is a weight-PTQ method comparison outside the study's question. — commit: this one.
- 2026-10-18 — **Energy protocol fixed:** pynvml polling at 10 Hz, medians over 3 repeats with the spread reported, tokens/s × latency model as fallback; I8 (energy repeats) added to the budget. — Why: Part-time Power Measurements (arXiv 2312.02741) reports up to 70 % naive energy error from coarse sampling; free-tier GPUs are shared and throttled. — commit: this one.
- 2026-10-18 — **Precision note:** fp16 on T4 (no bf16); fp32 fallback for the 1.5B model if NaNs appear in the zero-change control. — commit: this one.
- 2026-10-18 — **Difficulty labelling added to I1** (5 sampled decodes per problem at FP16 for pass@k). — Why: needed for the stratified H1 test and as the selector's training signal. — commit: this one.
