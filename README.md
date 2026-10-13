# Cheaper tokens, worse thoughts? KV-cache, quantisation and sparse attention versus chain-of-thought accuracy, with a single-GPU cost model

**Status:** Pre-registration · Week 0 of 5 · Pre-registered 2026-10-13 · Models: Qwen2.5-1.5B-Instruct (primary), Qwen2.5-3B-Instruct (secondary) · Uncertainty: 1,000-sample bootstrap over problems; seeds 0, 1, 2 for sampled self-consistency only

**Problem in one sentence.** We quantify and explain how weight quantisation, KV-cache quantisation and attention sparsification each degrade chain-of-thought reasoning as a function of reasoning length, and place them on an accuracy-per-second and accuracy-per-joule frontier.

**Why this matters.** Every efficiency technique for LLM inference is validated on perplexity or short-answer benchmarks, and multi-step reasoning is where the cost of a cheap token actually shows up: an autoregressive derivation that goes wrong at token 150 is wrong from then on. Practitioners deploying small reasoning models on a single consumer GPU or an edge device need to know which knob to turn first, and they currently choose by folklore. A study that measures degradation against reasoning length, explains it with an error-propagation argument, and couples it to a measured cost model gives them an "accuracy per joule" frontier they can read off.

---

## 1. Research proposition

### The gap

IISc's systems and algorithms groups surround this problem from both sides.

On the **systems side**, Yogesh Simmhan's group benchmarked 2.7B–32.8B LLMs on a Jetson Orin AGX across batch size, sequence length, quantisation and power modes ([arXiv 2506.09554](https://arxiv.org/abs/2506.09554)), built forecast-aware LLM serving (SageServe, SIGMETRICS 2026) and time/energy rooflines for edge accelerators (Pagoda, ICPP 2026) ([Simmhan faculty page](https://cds.iisc.ac.in/faculty/simmhan/)). Arkaprava Basu's lab works on GPU memory oversubscription with observability (ISCA 2026) and compiler support for CUDA Graphs in ML workloads (GraCE, OSDI 2026) ([Basu papers](https://www.csa.iisc.ac.in/~arkapravab/papers.html)); Sanchari Sen's ACES lab does transformer compression and performance/energy modelling ([Sen page](https://www.csa.iisc.ac.in/~sancharisen/)); Sumit Mandal builds ML accelerators; Uday Bondhugula's PolyBlocks and R. Govindarajan's ML-for-compiler autotuning supply the compiler view ([Bondhugula home](https://www.csa.iisc.ac.in/~uday)).

On the **algorithms side**, Chiranjib Bhattacharyya's LevAttention (ICLR 2025, with Kannan, Kacham and Woodruff) gives a time-, space- and streaming-efficient algorithm for *heavy* attentions, where mass concentrates on a few keys ([ML Lab publications](http://mllab.csa.iisc.ac.in/publications)); Chandra Sekhar Seelamantula's SPECTRUM lab presented COAT, a correlation-aware orthogonal transform for LLM quantisation, at an ICML 2026 workshop ([SPECTRUM publications](https://spectrum.ee.iisc.ac.in/publications/)). The 2025 IISc CS Workshop featured sparse attention for long-context inference, and the KIAC pre-summit meeting in January 2026 framed "efficient, resilient AI" under compute constraints as a national priority ([CS Workshop 2025](https://events.csa.iisc.ac.in/csworkshop2025/schedule.html); [KIAC event page](https://kiac.iisc.ac.in/?p=8765)).

What is missing between them is a study that measures how each efficiency intervention degrades **multi-step reasoning specifically**, explains why through an error-propagation argument, and couples the result to a cost model so the accuracy-per-joule frontier can be read off. The Jetson benchmark measures throughput and energy but not reasoning accuracy; LevAttention proves bounds under a heavy-attention premise but does not test whether real reasoning layers satisfy it; COAT improves quantisation error but is evaluated on perplexity. We join the three.

### The research question

How do weight quantisation, KV-cache quantisation and attention sparsification each affect chain-of-thought accuracy as a function of reasoning length, what mechanism explains the differences, and what is the resulting accuracy-per-second and accuracy-per-joule Pareto frontier on a single consumer GPU (and, if available, a Jetson)?

### Why inference-only, and why relative degradation

No training is involved, so the entire budget is decodes, which makes the study cheap and exactly reproducible under greedy decoding. We report *relative* degradation against the FP16 baseline as the main quantity because it is insensitive to benchmark contamination, which can inflate absolute accuracy.

---

## 2. Pre-registered hypotheses

Full text in [`HYPOTHESES.md`](HYPOTHESES.md); tests reproduced verbatim.

**H1 — cache errors accumulate, weight errors do not.** Accuracy loss from KV-cache quantisation grows with the length of the generated chain, because noise in cached keys/values perturbs every later step; weight-only quantisation adds roughly length-independent per-token noise.
*Confirmed* if the GSM8K accuracy gap (FP16 − INT4 KV) in the longest-solution tertile is ≥ 2× the gap in the shortest tertile with non-overlapping bootstrap CIs, and median first-divergence token decreases monotonically INT8 → INT4 → INT2, while the INT4-weight gap differs by < 1.5× across tertiles. *Refuted* if KV gaps are within CI across tertiles.

**H2 — sparsification is safe where attention is concentrated.** Heavy-hitter or leverage-score sparsification preserves accuracy in layers with low attention entropy; per-layer accuracy drop under single-layer sparsification correlates with that layer's entropy.
*Confirmed* if Spearman correlation between a layer's attention entropy and its accuracy drop under 50% single-layer sparsification is ≥ 0.6 (and ≥ 0.5 for top-k mass, in the opposite direction). *Refuted* if $|\rho| < 0.3$.

**H3 — the frontier is non-monotone and selectable.** The best accuracy-per-joule configuration is a mixed one (4-bit weights, 8-bit KV, moderate sparsity), not the most aggressive, and a per-problem difficulty predictor can beat any fixed configuration.
*Confirmed* if the accuracy-per-joule Pareto-optimal configuration is not the most aggressive one and a difficulty-aware per-problem selector beats the best fixed configuration by ≥ 1 accuracy point at equal or lower energy. *Refuted* if the most aggressive configuration Pareto-dominates or the selector gains < 0.5 points.

---

## 3. Experimental design

| Role | Variable | Levels |
|---|---|---|
| Independent | Weight precision | FP16, INT8, INT4 (GPTQ/AWQ-style) |
| Independent | KV-cache precision | {FP16, INT8, INT4, INT2} × {per-token, per-channel} |
| Independent | Attention sparsification | {none, sliding window, sink + window, top-k heavy-hitter, leverage-score sampling} × {50%, 25% keys kept} |
| Independent | Which single layer is sparsified | per-layer sweep (28 layers at 1.5B) |
| Dependent | Accuracy | GSM8K; MATH-500 subset; ARC-Challenge as short-reasoning control |
| Dependent | Accuracy by gold-solution-length tertile | short / medium / long |
| Dependent | Per-token KL to the FP16 model; first-divergence token position | from cached FP16 logits |
| Dependent | Tokens/s; peak memory; energy per problem | polled GPU power or tegrastats |
| Dependent | Per-layer attention entropy and top-k mass | measured at FP16 |
| Controlled | Models | Qwen2.5-1.5B-Instruct primary, 3B secondary (Llama-3.2-3B-Instruct; 7B in 4-bit as stretch) |
| Controlled | Prompting | identical prompts and few-shot CoT template; greedy decoding; max 512 new tokens |
| Controlled | Batch size | fixed per model |
| Controlled | Uncertainty | 1,000-sample bootstrap over problems; 3 seeds only for sampled self-consistency |
| Control (expected to fail) | "Zero-change" control | quantisation at full bits must reproduce FP16 outputs bit-exactly |

**Tasks.** GSM8K (1,319 problems) and a MATH-500 subset with chain-of-thought; ARC-Challenge as the short-reasoning control; a long-context retrieval probe.

**Baselines.** FP16 everything, with generations and logits cached once and reused for every KL and first-divergence comparison.

**Ablations.** Sequence length; batch size; number of sampled chains for self-consistency (does cheaper inference plus more samples beat full precision at equal energy?); COAT-style rotation versus plain per-channel quantisation; sink protection (first cached positions kept at FP16).

**The expected-to-fail control.** Our KV-quantisation and sparsification hooks are run at "no quantisation / 100% keys kept" and must reproduce the FP16 token sequence bit-exactly on every problem. A harness that fails this is fixed before any sweep is run.

---

## 4. The reasoning component

The explanation under test is an **error-propagation model**.

Treat KV quantisation as additive noise with variance $\sigma^2$ on cached keys and values. The attention logits at decode step $t$ receive a perturbation whose variance grows with the number of cached positions: the logit error is of order $\sigma\sqrt{t}$ under independent per-position errors and of order $\sigma t$ if errors are correlated across positions, which per-channel schemes induce. The probability that the argmax token flips at step $t$ rises accordingly, and because chain-of-thought is autoregressive, one flipped token can change the rest of the derivation. Two predictions follow:

1. Expected first-divergence position falls as $1/\sigma^2$, or faster under correlation.
2. The accuracy gap between quantised and full-precision models widens with solution length.

Weight quantisation perturbs every step with roughly the same variance and no accumulation, which predicts a flat gap across tertiles. That contrast is H1.

For sparsification, the leverage-score view says that if attention mass concentrates on a few keys, dropping the rest changes the output negligibly. We measure per-layer attention entropy and the fraction of mass in the top-$k$ keys at FP16, sparsify one layer at a time, and regress accuracy loss on those statistics. That is H2, and it is a direct empirical test of LevAttention's heavy-attention premise on reasoning workloads.

For the systems side we fit a decode-latency model,

$$\text{latency}(t) \approx \frac{W_{\text{bytes}} + \text{KV}_{\text{bytes}}(t)}{\text{BW}} + \text{compute term},$$

check it against a roofline, and show that KV bytes dominate decode cost beyond a sequence-length threshold that depends on model size. This explains why KV quantisation is where the energy win lives, and hence why its accuracy cost matters most. The methodology mirrors the observe-then-model style of Simmhan's Jetson paper and Pagoda rooflines and of Sen's performance modelling.

---

## 5. Compute budget (hard cap)

Inference-only; the driver is the number of (configuration × model × problem) decodes. A GSM8K pass of 1,319 problems with 512-token CoT on a 1.5B model takes roughly 1.5 T4-hours batched; a 500-problem subset about 0.6 h; 3B about 2×, 7B in 4-bit about 4×. Cost assumptions: Colab Pro about US$0.20 per T4-hour; Kaggle about 30 free GPU-hours per week; rented 3090/4090 at US$0.30–0.50 per hour; Lightning AI free monthly credit.

| Experiment | Model | Configs × problems | GPU-h (full) | Colab Pro | Kaggle free | Rented 3090/4090 | Lightning free |
|---|---|---|---|---|---|---|---|
| I1 FP16 baselines + zero-change control + power logging | 1.5B, 3B | 2 × 1,319 | 4.5 | ~$0.90 | $0 | ~$1.80 | within quota |
| I2 Weight quantisation (INT8, INT4) | 1.5B, 3B | 4 × 1,319 | 9 | ~$1.80 | $0 | ~$3.60 | partial |
| I3 KV-cache quantisation (3 bit-levels × 2 schemes) | 1.5B, 3B | 12 × 1,319 | 27 | ~$5.40 | $0 (one week's quota) | ~$11 | no |
| I4 Attention sparsification (4 methods × 2 budgets) | 1.5B, 3B | 16 × 1,319 | 36 | ~$7.20 | $0 (over 2 weeks) | ~$14 | no |
| I5 Per-layer single-layer sparsification sweep (28 layers) | 1.5B | 28 × 500 | 8 | ~$1.60 | $0 | ~$3.20 | partial |
| I6 MATH-500 subset + ARC-Challenge control (headline configs) | 1.5B, 3B | 6 × 500 | 4 | ~$0.80 | $0 | ~$1.60 | within quota |
| I7 Self-consistency at equal energy (3 seeds × 5 samples) | 1.5B | 3 × 500 × 5 | 5 | ~$1 | $0 | ~$2 | within quota |
| I8 7B-in-4-bit stretch (headline configs) | 7B | 3 × 500 | 6 | ~$1.20 | $0 (P100 only) | ~$2.40 | no |
| I9 Optional Jetson Orin Nano replication (if hardware is borrowed) | 1.5B | 6 × 300 | n/a (device) | $0 | $0 | $0 | $0 |
| **Total full version** | | | **~100 (≈75 on 1.5B + 3B only)** | **~$20** | **$0 over ~4 weeks** | **~$40** | — |

**Minimum viable version (~18 GPU-hours, ≤US$4 on Colab Pro or free on Kaggle).** Qwen2.5-1.5B only (Qwen2.5-0.5B or SmolLM-1.7B as fallback); a fixed 300-problem GSM8K subset **stratified by solution length** so the tertile test keeps power; INT4 weights only (I2 = 0.5 h); KV at INT8 and INT4 per-token only (I3 = 1 h); sink + window and top-k at 50% (I4 = 1 h); a 6-layer subset for the per-layer sweep chosen to span low and high entropy (I5 = 1 h); ARC control at 200 problems (0.5 h); baselines and power logging (I1 = 1.5 h). Remaining hours go to repeats of the energy measurements, which are noisy on shared GPUs.

*What survives at minimum scale:* H1 (the length-tertile contrast is a large effect at 300 stratified problems); H2 as a 6-point correlation, reported as preliminary; H3 for the frontier but **not** the selector, which needs the full grid to have anything to select among. The full version is required for the per-layer correlation across all layers, the 3B/7B generalisation claim, self-consistency at equal energy, and any per-channel versus per-token conclusion.

**Caching rules.** Compute FP16 baseline generations and logits once and reuse them for every KL and first-divergence comparison. Share prefill KV across configurations that differ only in decode-time cache precision by running prefill at FP16 and quantising the cache afterwards, which is also the realistic deployment setting. Report energy as medians over three repeats; keep the tokens/s-based cost model as the fallback whenever direct power measurement is unavailable.

---

## 6. Week-by-week plan

- [ ] **W1 — Harness with bit-exact zero-change control, FP16 baselines, power logging.** Artifact: baseline table and a notebook proving the control passes; this pre-registered README committed.
- [ ] **W2 — Weight and KV quantisation sweeps.** Artifact: accuracy-by-tertile plot and first-divergence histogram.
- [ ] **W3 — Sparsification sweeps and per-layer entropy.** Artifact: entropy-vs-drop scatter with $\rho$.
- [ ] **W4 — Latency/energy model, Pareto frontier, difficulty-aware selector.** Artifact: frontier figure and selector table.
- [ ] **W5 — 3B/7B replication and self-consistency (full budget only), negative results, preprint v0.** Artifact: Figure 1 and the benchmarking harness README.

---

## 7. Expected results and how to read either outcome

We expect INT8 KV to be nearly free, INT4 KV to cost several GSM8K points with the loss concentrated in long solutions, INT2 to collapse, and 4-bit weights to cost a roughly length-independent amount.

If the KV loss is **flat** in length, the error-propagation model is wrong for these models, and the likely reason is that attention sinks absorb the perturbation. We then check whether the first few cached positions dominate attention mass and whether protecting them (keeping sinks at full precision) removes the loss; that is a mechanism either way.

If sparsification hurts **even in low-entropy layers**, the probable explanation is that those layers route rare but decisive tokens; we check by inspecting which dropped keys receive attention at the divergence step.

The Pareto frontier plus the difficulty-aware selector is the deliverable most likely to interest the systems labs regardless of which way the hypotheses fall.

---

## 8. Deliverables

1. **Preprint:** *"Accuracy per joule: how KV-cache quantisation and sparse attention degrade chain-of-thought reasoning."*
2. **This repository:** a benchmarking harness that logs accuracy, latency, memory and energy per configuration, and a notebook that regenerates the frontier.
3. **Per-professor one-pagers.** Simmhan: the energy frontier and Jetson numbers if obtained. Basu: memory behaviour and the latency model. Sen and Mandal: compression-versus-accuracy modelling. Bhattacharyya: the leverage-score sparsification result. Seelamantula: COAT-style rotations versus plain per-channel quantisation. Bondhugula and Govindarajan: where a compiler could exploit the per-layer sparsity pattern.

---

## 9. Research protocol

- **Pre-register before running.** `HYPOTHESES.md` is committed before the first sweep.
- **Experiment log.** `logs/EXPERIMENT_LOG.md` records config hash, git commit, expectation, observation and next change per run.
- **Determinism and uncertainty.** Greedy decoding is deterministic, so uncertainty comes from a 1,000-sample bootstrap over problems; seeds 0, 1, 2 are used only for sampled self-consistency. 95% CIs on every plotted number; comparisons are paired over identical problems.
- **Energy.** Medians over three repeats; tokens/s cost model as fallback.
- **Negative results.** The preprint states which hypotheses were refuted and what we now believe instead.
- **Expected-to-fail control.** The zero-change control must be bit-exact before any sweep; it is re-run after every harness change.
- **Figures** come only from single-command-reproducible runs. Relative degradation is the headline quantity; a 1.5B result is reported as a 1.5B result.

---

## 10. Repository layout (planned)

```
cot-under-compression/
├── README.md
├── HYPOTHESES.md                 # pre-registration, frozen 2026-10-13
├── configs/                      # weights/, kv/, sparse/, per_layer/, frontier/; one YAML per cell
├── src/
│   ├── harness/                  # generation, bit-exact control, power polling, latency model
│   ├── interventions/            # weight quant, kv quant (per-token/per-channel), sparsifiers (window, sink, topk, leverage)
│   ├── metrics/                  # accuracy, tertiles, per-token KL, first divergence, entropy, top-k mass
│   └── frontier/                 # Pareto computation, difficulty-aware selector
├── notebooks/                    # control proof, weekly artifacts, frontier regeneration
├── results/                      # cached FP16 generations+logits, per-config CSVs, power traces
├── logs/EXPERIMENT_LOG.md
└── paper/                        # preprint source, one-pagers/
```

---

## 11. Related work to read first

**IISc papers this study builds on**

- Y. Simmhan et al., *LLM inferencing on edge accelerators* (Jetson Orin AGX benchmark), 2025, [arXiv:2506.09554](https://arxiv.org/abs/2506.09554); SageServe, SIGMETRICS 2026; Pagoda, ICPP 2026. [Simmhan faculty page](https://cds.iisc.ac.in/faculty/simmhan/)
- C. Bhattacharyya, R. Kannan, P. Kacham, D. Woodruff, *LevAttention*, ICLR 2025. [ML Lab publications](http://mllab.csa.iisc.ac.in/publications)
- C. S. Seelamantula et al., COAT: correlation-aware orthogonal transform for LLM quantisation, ICML 2026 workshop. [SPECTRUM publications](https://spectrum.ee.iisc.ac.in/publications/)
- A. Basu et al., GraCE (OSDI 2026); GPU memory oversubscription (ISCA 2026). [Basu papers](https://www.csa.iisc.ac.in/~arkapravab/papers.html)
- S. Sen et al., transformer compression, NeurIPS 2022. [Sen page](https://www.csa.iisc.ac.in/~sancharisen/)

**Canonical external papers**

- J. Wei et al., *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models*, NeurIPS 2022, arXiv:2201.11903.
- K. Cobbe et al., *Training Verifiers to Solve Math Word Problems* (GSM8K), 2021, arXiv:2110.14168.
- E. Frantar et al., *GPTQ*, ICLR 2023, arXiv:2210.17323; J. Lin et al., *AWQ*, 2023, arXiv:2306.00978.
- C. Hooper et al., *KVQuant*, 2024, arXiv:2401.18079; Z. Liu et al., *KIVI*, 2024, arXiv:2402.02750.
- G. Xiao et al., *Efficient Streaming Language Models with Attention Sinks*, 2023, arXiv:2309.17453; Z. Zhang et al., *H2O: Heavy-Hitter Oracle*, 2023, arXiv:2306.14048.
- X. Wang et al., *Self-Consistency Improves Chain of Thought Reasoning*, 2022, arXiv:2203.11171.

---

## 12. Faculty this work is aimed at

| Faculty | Dept | Paper this extends |
|---|---|---|
| Yogesh Simmhan | CDS | Jetson LLM benchmarking (arXiv 2506.09554); SageServe (SIGMETRICS 2026); Pagoda (ICPP 2026) |
| Arkaprava Basu | CSA | GraCE (OSDI 2026); GPU memory oversubscription (ISCA 2026) |
| Sanchari Sen | CSA | Transformer compression (NeurIPS 2022); performance/energy modelling |
| Sumit K. Mandal | CSA | ML accelerators; SoC power management |
| Chiranjib Bhattacharyya | CSA | LevAttention (ICLR 2025) |
| C. S. Seelamantula | EE | COAT LLM quantisation (ICML 2026 workshop) |
| Uday Bondhugula | CSA | PolyBlocks (2026) |
| R. Govindarajan | CSA | ML-for-compiler autotuning (ICS 2024) |

---

## 13. License and citation

Code: MIT. Data and figures: CC BY 4.0.

```bibtex
@misc{cot-under-compression-2026,
  title  = {Accuracy per joule: how KV-cache quantisation and sparse attention degrade chain-of-thought reasoning},
  author = {<Your Name>},
  year   = {2026},
  note   = {Pre-registered 2026-10-13. https://github.com/Arubad/cot-under-compression}
}
```
