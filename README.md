# Cheaper tokens, worse thoughts? KV-cache quantisation, weight quantisation and sparse attention versus chain-of-thought accuracy, with a single-GPU cost model

**Status:** Pre-registration v0.2 · Week 0 of 10 · Pre-registered 2026-10-13, amended 2026-10-18 before any run · Models: Qwen2.5-1.5B-Instruct (primary), Qwen2.5-3B-Instruct (secondary) · Uncertainty: 1,000-sample bootstrap over problems; seeds 0, 1, 2 for sampled self-consistency only

| Document | What it is |
|---|---|
| [`PREREGISTRATION.md`](PREREGISTRATION.md) | Frozen hypotheses, numeric pass/fail tests, models, settings. Changes only via the dated Amendments appendix. |
| [`REVIEW.md`](REVIEW.md) | Independent review of 2026-10-18: citation corrections, novelty against the literature, flaws in the derivation and design, what changed and why. Read this before the pre-registration. |
| [`ROADMAP.md`](ROADMAP.md) | Milestones as GitHub task lists. Tick them off; this is the progress tracker. |
| [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md) | Append-only log, one entry per configuration run, expectation written before launch. |
| [`READING.md`](READING.md) | Ordered reading list with the reason for each item. |

**Problem in one sentence.** We measure how weight quantisation, KV-cache quantisation and attention sparsification each degrade chain-of-thought reasoning on a small instruct model, locate *where* in the chain the damage starts, and put every configuration on an energy-per-correct-answer frontier that includes the extra tokens cheap inference makes the model emit.

**Why this matters.** Efficiency techniques for LLM inference are validated on perplexity or short-answer benchmarks, and multi-step reasoning is where the cost of a cheap token shows up: a derivation that goes wrong at token 150 is wrong from then on. That KV-cache error accumulates over long chains has now been argued in print ([PM-KVQ](https://arxiv.org/abs/2505.18610); [KVarN](https://arxiv.org/abs/2606.03458)), and that low-bit quantisation makes reasoning models think longer has too ([Lian et al.](https://arxiv.org/abs/2606.25519); [Lotfi et al.](https://arxiv.org/abs/2606.00206)). What nobody has done is measure the three interventions head-to-head on the same problems with the divergence point recorded, fold token inflation into a measured cost model so the unit is *energy per correct answer*, and ask whether a per-problem selector beats any fixed configuration. Practitioners on one consumer GPU currently choose by folklore; this gives them a frontier to read off.

---

## 1. Research proposition

### The gap

IISc's systems and algorithms groups surround this problem from both sides, but none of the cited work measures reasoning accuracy.

On the **systems side**, Arya and Simmhan benchmarked LLM inference on a Jetson Orin AGX across batch size, sequence length, quantisation and power modes in a short PAISE 2025 paper ([arXiv 2506.09554](https://arxiv.org/abs/2506.09554)); the same group built forecast-aware LLM serving with Microsoft (SageServe, [arXiv 2502.14617](https://arxiv.org/abs/2502.14617), SIGMETRICS 2026 to appear) and energy/time rooflines for DNN workloads on edge accelerators (Pagoda, [arXiv 2509.20189](https://arxiv.org/abs/2509.20189), ICPP 2026; DNNs, not LLMs) ([Simmhan faculty page](https://cds.iisc.ac.in/faculty/simmhan/)). Sanchari Sen's group does transformer compression and performance/energy modelling ([Sen page](https://eecs.iisc.ac.in/people/sanchari-sen/)). Arkaprava Basu's GraCE (OSDI 2026) and observability-aided GPU memory oversubscription (ISCA 2026) are GPU-systems work that does not touch model accuracy ([Basu papers](https://www.csa.iisc.ac.in/~arkapravab/papers.html)).

On the **algorithms side**, LevAttention (Kannan, Bhattacharyya, Kacham, Woodruff; [ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/30728), [arXiv 2410.05462](https://arxiv.org/abs/2410.05462)) gives a time-, space- and streaming-efficient algorithm for *heavy* attentions, where mass concentrates on a few keys; its experiments are on vision transformers, so whether reasoning layers in an LLM satisfy the premise is an extrapolation we test, not a claim the paper makes. COAT (Patra, Yousuf, Mathew, Seelamantula), a rotation-based weight/activation PTQ method, is a poster at the AdaptFM workshop at ICML 2026 ([ICML page](https://icml.cc/virtual/2026/75331)).

The **efficient-reasoning literature** already has the aggregate picture. [Liu et al. 2025](https://arxiv.org/abs/2504.04823) benchmark weight, activation and KV quantisation on R1-distilled 1.5B–70B reasoning models (W8A8 and W4A16 near-lossless, lower bits risky); [When Reasoning Meets Compression](https://arxiv.org/abs/2504.02010) (ICLR 2026) benchmarks quantisation, pruning and distillation; [The Quantization Trap](https://arxiv.org/abs/2602.13595) shows 4- and 8-bit can raise net energy and lower multi-hop accuracy. None records the first-divergence token, separates length from difficulty, or reports energy per correct answer with token inflation included.

### The research question

On a single consumer GPU, which of weight quantisation, KV-cache quantisation and attention sparsification gives the most correct chain-of-thought answers per joule once the extra tokens cheap inference induces are counted; where in the chain does each intervention first break the derivation; and can a per-problem difficulty-aware selector beat every fixed configuration?

### Why inference-only, and why relative degradation

No training is involved, so the entire budget is decodes, which makes the study cheap and exactly reproducible under greedy decoding. We report *relative* degradation against the FP16 baseline as the main quantity because it is insensitive to benchmark contamination, which can inflate absolute accuracy.

---

## 2. Pre-registered hypotheses (v0.2)

Full text in [`PREREGISTRATION.md`](PREREGISTRATION.md); the 2026-10-18 amendments are in its appendix. Tests reproduced verbatim.

**H3 (headline) — the energy-per-correct-answer frontier is non-monotone and selectable.** Once token inflation is included in the cost model, the best energy-per-correct-answer configuration is a mixed one (4-bit weights, 8-bit KV, moderate sparsity), not the most aggressive, and a per-problem difficulty predictor beats any fixed configuration.
*Confirmed* if the Pareto-optimal configuration on the energy-per-correct-answer axis is not the most aggressive one and a difficulty-aware per-problem selector beats the best fixed configuration by ≥ 1 accuracy point at equal or lower energy. *Refuted* if the most aggressive configuration Pareto-dominates or the selector gains < 0.5 points.

**H1 — cache errors persist, weight errors do not (controlled measurement of a claimed mechanism).** A quantised KV entry is a persistent, position-specific error that every later step reads; a quantised weight is the same perturbation at every step. Prediction: under KV quantisation the first-divergence token moves earlier as bits drop and the accuracy gap grows with chain length *after controlling for difficulty*; under weight quantisation the first-divergence position is flat across bit-widths and the gap is length-independent.
*Confirmed* if median first-divergence token decreases monotonically INT8 → INT4 → INT2 for KV quantisation with non-overlapping bootstrap CIs, the GSM8K accuracy gap (FP16 − INT4 KV) in the longest-solution tertile is ≥ 2× the gap in the shortest tertile within each difficulty stratum, and the INT4-weight gap differs by < 1.5× across tertiles. *Refuted* if KV first-divergence positions are within CI across bit-widths, or KV gaps are within CI across tertiles.

**H2 — sparsification is safe where attention is concentrated (per head).** Heavy-hitter or leverage-score sparsification preserves accuracy in heads with low attention entropy; per-head accuracy drop under single-head sparsification correlates with that head's entropy. This is an empirical test of the LevAttention heavy-attention premise on an LLM reasoning workload.
*Confirmed* if Spearman correlation between a head's attention entropy and its accuracy drop under 50 % single-head sparsification is ≥ 0.6 (and ≥ 0.5 for top-k mass, in the opposite direction) over the heads of a 6-layer subset chosen to span low and high entropy. *Refuted* if |ρ| < 0.3.

**Most interesting possible outcome.** If the KV loss is flat in chain length even after difficulty stratification, the persistence mechanism is wrong for these models, and the likely reason is that attention sinks absorb the perturbation. We then check whether protecting the first cached positions removes the loss. That is a mechanism either way.

---

## 3. Experimental design

| Role | Variable | Levels |
|---|---|---|
| Independent | Weight precision | FP16, INT8, INT4 (GPTQ/AWQ-style) |
| Independent | KV-cache precision | {FP16, INT8, INT4, INT2} × {per-token, per-channel} |
| Independent | Attention sparsification | {none, sliding window, sink + window, top-k heavy-hitter, leverage-score sampling} × {50 %, 25 % keys kept} |
| Independent | Which single head is sparsified | per-head sweep over a 6-layer subset spanning low and high entropy |
| Dependent | Accuracy | GSM8K; MATH-500 subset; ARC-Challenge as short-reasoning control |
| Dependent | Accuracy by gold-solution-length tertile, stratified by difficulty | difficulty = FP16 pass@k on sampled decodes, fixed before the sweep |
| Dependent | First-divergence token position; per-token KL to FP16 | from cached FP16 logits; **primary statistic for H1** |
| Dependent | Generated tokens per problem (token inflation) | enters the cost model |
| Dependent | Tokens/s; peak memory; energy per problem; energy per correct answer | pynvml polling; medians over 3 repeats with spread reported |
| Dependent | Per-head attention entropy and top-k mass | measured at FP16 |
| Controlled | Models | Qwen2.5-1.5B-Instruct primary, Qwen2.5-3B-Instruct secondary |
| Controlled | Prompting | identical prompts and 8-shot CoT template; greedy decoding; max 512 new tokens |
| Controlled | Precision | fp16 on T4 (no bf16); fp32 fallback for 1.5B if NaNs appear |
| Controlled | Batch size | fixed per model |
| Controlled | Uncertainty | 1,000-sample bootstrap over problems; 3 seeds only for sampled self-consistency |
| Control (expected to fail) | "Zero-change" control | quantisation at full bits must reproduce FP16 outputs bit-exactly |

**Tasks.** GSM8K (1,319 problems; minimum version a fixed 300-problem subset stratified by solution length *and* FP16 difficulty) and a MATH-500 subset with chain-of-thought; ARC-Challenge as the short-reasoning control.

**Baselines.** FP16 everything, with generations and logits cached once and reused for every KL and first-divergence comparison.

**Ablations.** Sequence length; number of sampled chains for self-consistency (does cheaper inference plus more samples beat full precision at equal energy?); sink protection (first cached positions kept at FP16).

**What is simulated.** On a T4 there is no FlashAttention, so sparsification is implemented as top-k masking inside eager/SDPA attention. That measures the accuracy effect faithfully but not the speed or energy effect. Every speed or energy number for the sparse arm is labelled *simulated* and derived from the latency model, never from a wall clock.

**Stretch (2026-10-18).** The Jetson Orin replication, the 7B-in-4-bit arm and the COAT-rotation versus per-channel comparison are moved to stretch; they run only if hardware or budget appears after Milestone 4.

**The expected-to-fail control.** Our KV-quantisation and sparsification hooks are run at "no quantisation / 100 % keys kept" and must reproduce the FP16 token sequence bit-exactly on every problem. A harness that fails this is fixed before any sweep is run.

---

## 4. The reasoning component

The explanation under test is a **persistence model** of error, corrected on 2026-10-18 from an earlier accumulation model that did not hold up (see [`REVIEW.md`](REVIEW.md) §4).

Treat KV quantisation as additive noise with variance $\sigma^2$ on each cached key and value. At decode step $t$ the attention output is a softmax-weighted average of the cached values, so value noise enters with variance $\sigma^2 \sum_i a_{t,i}^2$, where $a_{t,i}$ are the attention weights. That sum is at most 1 and *shrinks* as attention spreads over more positions, so the per-step logit perturbation does not grow with $t$ by itself. Key noise perturbs the logits before the softmax and is likewise weighted, not summed.

What does compound is autoregression: a single flipped token changes the context for every later step, and under greedy decoding the trajectory never returns. This mechanism applies equally to weight quantisation. The asymmetry between the two interventions is therefore not "accumulates versus does not", but:

1. A quantised KV entry is a **persistent, position-specific** error. Once position $i$ is cached with error, every later step that attends to $i$ reads the same wrong value, and the steps that attend most to early positions (sinks, the question statement) are exactly the ones that anchor the derivation.
2. A quantised weight is the **same perturbation at every step**, with no position memory; the model can be fine-tuned, or simply trained, to be robust to it, which is why 4-bit weights are usually near-lossless ([Dettmers and Zettlemoyer](https://arxiv.org/abs/2212.09720)).

Two predictions follow. The first-divergence position under KV quantisation should move earlier as bits drop, because the probability that a persistent error flips an argmax rises with the error magnitude and the number of anchored positions; under weight quantisation it should be roughly flat across bit-widths. And the accuracy gap under KV quantisation should grow with chain length *after* difficulty is held fixed, because longer chains read the corrupted anchors more times. Length and difficulty are confounded on GSM8K (longer gold solutions are harder problems), so the tertile test is run within difficulty strata; the first-divergence position does not have the confound and is the primary statistic.

For sparsification, the leverage-score view says that if attention mass concentrates on a few keys, dropping the rest changes the output negligibly. We measure per-head attention entropy and the fraction of mass in the top-$k$ keys at FP16, sparsify one head at a time, and regress accuracy loss on those statistics. Per-head, not per-layer, because retrieval and streaming heads coexist within a layer ([DuoAttention](https://arxiv.org/abs/2410.10819)).

For the systems side we fit a decode-latency model,

$$\text{latency}(t) \approx \frac{W_{\text{bytes}} + \text{KV}_{\text{bytes}}(t)}{\text{BW}} + \text{compute term},$$

check it against a roofline, and show that KV bytes dominate decode cost beyond a sequence-length threshold that depends on model size. The cost model then multiplies per-token energy by the *measured* number of generated tokens for that configuration, because low-bit quantisation lengthens chains, and divides by the number of correct answers. Energy per correct answer, not per token, is the headline unit.

---

## 5. Compute budget (hard cap)

Inference-only; the driver is the number of (configuration × model × problem) decodes. A GSM8K pass of 1,319 problems with 512-token CoT on a 1.5B model takes roughly 1.5 T4-hours batched; a 500-problem subset about 0.6 h; 3B about 2×. Kaggle's free quota is commonly reported as ~30 GPU-hours per week (not an official figure); Colab free is unpublished and interruptible.

| Experiment | Model | Configs × problems | GPU-h (full) | Kaggle weeks |
|---|---|---|---|---|
| I1 FP16 baselines + zero-change control + power logging + difficulty labels (pass@k) | 1.5B, 3B | 2 × 1,319 (+ 5 sampled decodes for difficulty) | 7 | 0.25 |
| I2 Weight quantisation (INT8, INT4) | 1.5B, 3B | 4 × 1,319 | 9 | 0.3 |
| I3 KV-cache quantisation (3 bit-levels × 2 schemes) | 1.5B, 3B | 12 × 1,319 | 27 | 0.9 |
| I4 Attention sparsification (4 methods × 2 budgets); speed/energy simulated | 1.5B, 3B | 16 × 1,319 | 36 | 1.2 |
| I5 Per-head single-head sparsification sweep, 6-layer subset | 1.5B | ~70 heads × 300 | 8 | 0.3 |
| I6 MATH-500 subset + ARC-Challenge control (headline configs) | 1.5B, 3B | 6 × 500 | 4 | 0.15 |
| I7 Self-consistency at equal energy (3 seeds × 5 samples) | 1.5B | 3 × 500 × 5 | 5 | 0.2 |
| I8 Energy repeats (3 repeats of every headline config) | 1.5B | headline configs | 6 | 0.2 |
| **Total full version** | | | **~100** | **~3.5** |
| Stretch: I9 7B-in-4-bit; I10 Jetson; I11 COAT vs per-channel | | | +15 | |

**Minimum viable version (~18 GPU-hours, one Kaggle week).** Qwen2.5-1.5B only; a fixed 300-problem GSM8K subset stratified by solution length and FP16 difficulty; INT4 weights only (I2 = 0.5 h); KV at INT8 and INT4 per-token only (I3 = 1 h); sink + window and top-k at 50 % (I4 = 1 h); the per-head sweep on the 6-layer subset (I5 = 3 h); ARC control at 200 problems (0.5 h); baselines, difficulty labels and power logging (I1 = 3 h). Remaining hours go to energy repeats, which are noisy on shared GPUs.

*What survives at minimum scale:* H1's first-divergence test and the within-stratum tertile contrast; H2 on the 6-layer subset; H3's frontier **and** a first version of the selector on the small grid, reported as preliminary. The full version is required for the 3B generalisation claim, self-consistency at equal energy, any per-channel versus per-token conclusion, and the selector's headline test.

**Caching rules.** Compute FP16 baseline generations and logits once and reuse them for every KL and first-divergence comparison. Share prefill KV across configurations that differ only in decode-time cache precision by running prefill at FP16 and quantising the cache afterwards, which is also the realistic deployment setting. Report energy as medians over three repeats with the spread; keep the tokens/s-based cost model as the fallback whenever direct power measurement is unavailable or implausible.

---


## 6. How to run (Kaggle)

Every experiment is one notebook in `notebooks/` (generated by `tools/make_notebooks.py`) that clones this repo and calls `python -m src.run`. Kaggle settings: Accelerator **GPU T4 x2**, Internet **On**. Results land in `/kaggle/working/results`; save that as a Kaggle dataset and attach it to the next notebook (set `PREV` in the setup cell).

| Order | Notebook | Command it runs | GPU-h (1.5B, minimum) |
|---|---|---|---|
| 0 | `00_smoke_test` | `python -m src.run smoke` — every code path on 30 problems; reports which KV backends run on this GPU | 0.2 |
| 1 | `01_I1_baseline_control` | `make_subset`, then `baseline` (fp16 cache + control + pass@4 difficulty + head statistics + latency model) | 3 |
| 2 | `02_I2_weights` | `sweep --config configs/weights_sweep.yaml --minimum` | 0.5 |
| 3 | `03_I3_kv` | `sweep --config configs/kv_sweep.yaml --minimum` | 1 |
| 4 | `04_I4_sparse` | `sweep --config configs/sparse_sweep.yaml --minimum` (speed/energy simulated) | 1 |
| 5 | `05_I5_perhead` | `perhead --config configs/perhead_sweep.yaml --minimum` | 3 |
| 6 | `06_I6_I7_frontier_selector` | ARC control, `frontier` (selector table + energy repeats) | 2 |
| 7 | `07_analysis_and_figures` | `analyze` — pre-registered tests + Figures 1–4 | 0 |

Locally: `pip install -r requirements.txt && python tests/test_numpy_only.py` runs the numpy-only tests (answer extraction, tertiles, first divergence, top-k KL, fake quantisation, keep masks, Pareto / latency model / selector, config cells) without a GPU.

**Status of the code (2026-10-20).** All modules are written and compile; the numpy parts are unit-tested; the torch parts (generation, the cache subclass, the custom attention function, bitsandbytes loading, NVML polling) have not yet been executed on a GPU. The smoke test is the first thing to run, and the zero-change control is the second. Two API points to confirm on the first run and log: that `model.config._attn_implementation = "sparse"` is honoured by every attention module in the installed transformers version (it is in 4.48–4.5x), and that the `DynamicCache` storage helper in `kv_quant.py` finds the right attribute (`layers[i].keys` in newer versions, `key_cache[i]` in older ones).

## 6b. Plan

The week-by-week plan with task lists lives in [`ROADMAP.md`](ROADMAP.md). Summary: W1–2 reading and rewrite of the pre-registration in your own words; W3–4 run the smoke test and the zero-change control, fix what breaks; W5–7 minimum-viable sweeps; W8–10 frontier, selector, 4-page write-up, one-page summary for Sanchari Sen.

---

## 7. Deliverables

1. **Preprint** (4-page workshop version first): *"Energy per correct answer: KV-cache quantisation, weight quantisation and sparse attention under chain-of-thought reasoning."*
2. **This repository:** a benchmarking harness that logs accuracy, first-divergence position, latency, memory and energy per configuration, and a notebook that regenerates the frontier.
3. **One-page summaries** for two readers only. Sanchari Sen: the energy-per-correct-answer frontier and the selector. Yogesh Simmhan: the latency model, roofline check and the measured-versus-simulated energy comparison.

---

## 8. Research protocol

- **Pre-register before running.** `PREREGISTRATION.md` is frozen before the first sweep; changes go in its Amendments appendix with a date and a reason.
- **Experiment log.** `EXPERIMENT_LOG.md` records config hash, git commit, expectation, observation and next change per run.
- **Determinism and uncertainty.** Greedy decoding is deterministic, so uncertainty comes from a 1,000-sample bootstrap over problems; seeds 0, 1, 2 are used only for sampled self-consistency. 95 % CIs on every plotted number; comparisons are paired over identical problems.
- **Energy.** Medians over three repeats with spread shown; coarse pynvml sampling is a known error source ([Part-time Power Measurements](https://arxiv.org/html/2312.02741v3)), so the tokens/s cost model is the fallback. Simulated numbers are labelled simulated.
- **Negative results.** The preprint states which hypotheses were refuted and what we now believe instead.
- **Expected-to-fail control.** The zero-change control must be bit-exact before any sweep; it is re-run after every harness change.
- **Figures** come only from single-command-reproducible runs. Relative degradation is the headline quantity; a 1.5B result is reported as a 1.5B result.
- **"First" claims** are made only after a Semantic Scholar sweep recorded in `REVIEW.md`.

---

## 9. Repository layout

```
cot-under-compression/
├── README.md
├── PREREGISTRATION.md            # frozen 2026-10-13; amendments appendix
├── REVIEW.md                     # independent review, 2026-10-18
├── ROADMAP.md                    # milestones as task lists (progress tracker)
├── EXPERIMENT_LOG.md             # append-only run log
├── READING.md                    # ordered reading list
├── requirements.txt
├── configs/                      # weights/, kv/, sparse/, per_head/, frontier/; one YAML per cell
├── src/                          # harness/, interventions/, metrics/, cost/, analysis/ (planned; see src/README.md)
├── notebooks/                    # control proof, weekly artifacts, frontier regeneration
├── results/                      # cached FP16 generations+logits, per-config CSVs, power traces (git-ignored)
└── paper/                        # preprint source, one-pagers/, notes/
```

---

## 10. Related work to read first

See [`READING.md`](READING.md) for the ordered list with reasons. The must-reads: Wei et al. (CoT), Cobbe et al. (GSM8K), Dettmers et al. (LLM.int8 and the 4-bit case), GPTQ, AWQ, KVQuant, KIVI, StreamingLLM, H2O, DuoAttention, Liu et al. 2025, PM-KVQ, KVarN, LevAttention, Arya and Simmhan, Pagoda.

---

## 11. Faculty this work is aimed at

| Faculty | Dept | Work this extends | Role |
|---|---|---|---|
| Sanchari Sen | CSA | LLM-inference energy and performance modelling; transformer compression | Primary reader |
| Yogesh Simmhan | CDS | LLM inference on edge accelerators (PAISE 2025); Pagoda rooflines (ICPP 2026) | Second reader |

---

## 12. License and citation

Code: MIT. Data and figures: CC BY 4.0.

```bibtex
@misc{cot-under-compression-2026,
  title  = {Energy per correct answer: KV-cache quantisation, weight quantisation and sparse attention under chain-of-thought reasoning},
  author = {Arush Badhe},
  year   = {2026},
  note   = {Pre-registered 2026-10-13, amended 2026-10-18. https://github.com/Arubad/cot-under-compression}
}
```
