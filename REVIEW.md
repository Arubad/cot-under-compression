# Review of the pre-registration (2026-10-18)

Independent review of the 2026-10-13 pre-registration, done before any experiment was run. Literature and author pages were opened on 17 October 2026. Every claim below links to the page it came from. The changes this review caused are logged in the Amendments appendix of [`PREREGISTRATION.md`](PREREGISTRATION.md).

**Verdict.** Doable and a good systems-side portfolio piece, but the most crowded of the three projects; its original headline hypothesis (H1, KV-cache error accumulates over long chains) has already been argued in print, so the study has been repositioned as a controlled measurement plus a cost-model study whose headline is the energy-per-correct-answer frontier and the per-problem selector. Novelty is low on H1, moderate on H2 and H3. Feasibility on a T4 is fine for accuracy, shaky for speed and energy.

## 1. Citation check

None of the IISc citations is fabricated, but several were mis-described.

| Pre-registration said | Reality | Source |
|---|---|---|
| Simmhan Jetson benchmark, arXiv 2506.09554 | Real, but it is a short PAISE 2025 workshop paper (Arya and Simmhan, *Understanding the Performance and Power of LLM Inferencing on Edge Accelerators*), not a main-track paper. Simmhan's page also lists a different PAISE 2026 paper on sandboxed Jetson inference; do not conflate them. | [arXiv 2506.09554](https://arxiv.org/abs/2506.09554), [Simmhan page](https://cds.iisc.ac.in/faculty/simmhan/) |
| SageServe, SIGMETRICS 2026 | Real; a Microsoft collaboration, listed as SIGMETRICS 2026 to appear with a POMACS 2025 extended version | [arXiv 2502.14617](https://arxiv.org/abs/2502.14617) |
| Pagoda, ICPP 2026 | Real; energy/time rooflines for DNN workloads on Jetson Orin power modes, not LLMs or quantisation. Useful for the roofline model only. | [arXiv 2509.20189](https://arxiv.org/abs/2509.20189) |
| GraCE (OSDI 2026); GPU oversubscription (ISCA 2026) | Both real on Basu's page (the second is *Observability-aided GPU Memory Oversubscription*); neither is about LLM accuracy, so they are tangential. **Removed from the target list.** | [Basu papers](https://www.csa.iisc.ac.in/~arkapravab/papers.html) |
| LevAttention, ICLR 2025 | Real (poster). Experiments are on vision transformers and the result is a theoretical "universal set" of heavy keys; it makes no empirical claim about LLM reasoning. H2 tests an extrapolation of the paper, not its claim. | [ICLR 2025](https://iclr.cc/virtual/2025/poster/30728), [arXiv 2410.05462](https://arxiv.org/abs/2410.05462) |
| COAT, ICML 2026 workshop | Real: *COrrelation-Aware Orthogonal Transform for LLM Quantization* (Patra, Yousuf, Mathew, Seelamantula), poster at the AdaptFM workshop at ICML 2026. Rotation-based weight/activation PTQ. Moved to stretch. | [ICML 2026](https://icml.cc/virtual/2026/75331) |
| Sanchari Sen, transformer compression NeurIPS 2022 | Her page lists LLM-inference energy and performance modelling and HW-SW co-design; the specific NeurIPS 2022 citation was not verified. | [Sen page](https://eecs.iisc.ac.in/people/sanchari-sen/) |
| Yuan et al. 2024, *KV cache compression, but what must we give in return* | **Could not find this paper**; search returned *Rethinking Key-Value Cache Compression Techniques for LLM Serving* (arXiv 2503.24000) instead. Verify before citing. | [arXiv 2503.24000](https://arxiv.org/pdf/2503.24000) |

## 2. Novelty against the literature

- **H1 (KV error over long chains) is the most scooped.** [PM-KVQ](https://arxiv.org/abs/2505.18610) argues per-step KV quantisation lets errors build over long CoT and proposes progressive mixed precision (7B–70B). [KVarN](https://arxiv.org/abs/2606.03458) (June 2026) is explicitly about mitigating error accumulation in reasoning at 2-bit, on MATH500 / AIME24 / HumanEval. [Liu et al. 2025](https://arxiv.org/abs/2504.04823) benchmark weight, activation and KV quantisation on R1-distilled 1.5B–70B reasoning models and find W8A8 and W4A16 near-lossless. [Li et al.](https://arxiv.org/abs/2501.03035) do weight PTQ on Llama-3 math reasoning with error diagnosis. [Semantic Integrity Matters](https://arxiv.org/abs/2502.01941) (formerly *Can LLMs Maintain Fundamental Abilities under KV Cache Compression*) shows reasoning degrades under KV compression more than retrieval does. [When Reasoning Meets Compression](https://arxiv.org/abs/2504.02010) (ICLR 2026) benchmarks quantisation, pruning and distillation on reasoning models. None of them reports the first-divergence token, a length-tertile analysis, or a paired weight-vs-KV contrast on one problem set; that finer measurement is what remains.
- **H2 (attention entropy predicts safe sparsification)** has prior art per head: [EntroKV](https://proceedings.mlr.press/v306/gao26a.html) (ICML 2026) allocates KV budgets across heads by attention entropy; [DuoAttention](https://arxiv.org/abs/2410.10819) separates retrieval from streaming heads; [layer-wise eviction design](https://arxiv.org/abs/2606.15157) uses offline profiles. Nobody frames it as a test of LevAttention's heavy-attention premise on reasoning accuracy; that framing is open but modest.
- **H3 (non-monotone frontier) is partly pre-empted.** [The Quantization Trap](https://arxiv.org/abs/2602.13595) shows 8- and 4-bit can raise net energy and lower multi-hop accuracy across six GPU architectures. [Lian et al.](https://arxiv.org/abs/2606.25519) and [Lotfi et al.](https://arxiv.org/abs/2606.00206) (NeurIPS 2026 oral per the listing) show low-bit PTQ lengthens CoT and offsets speedups. [Jin, Wei and Brooks](https://arxiv.org/abs/2505.14733) report accuracy-per-energy trade-offs for test-time compute; [Husom et al.](https://arxiv.org/abs/2504.03360) measure energy, accuracy and latency for 28 quantised models on a Raspberry Pi including GSM8K. **The per-problem difficulty-aware selector was not found anywhere** and is the most publishable piece; the energy-per-correct-answer unit with measured token inflation ties it to the Lian and Lotfi results.
- **Adjacent methods, not competitors:** [R-KV](https://arxiv.org/abs/2505.24133) (eviction for reasoning), [QEvict](https://arxiv.org/abs/2608.05326) and [InfoKV](https://arxiv.org/abs/2606.26875) (hybrid quantise/evict), [SCBench](https://arxiv.org/abs/2412.10319) (KV-centric long-context benchmark), [Agrawal and Mayer](https://arxiv.org/abs/2607.05399) (compression ratio is a poor predictor of end-to-end performance; no reasoning, no energy), [long-context quantisation](https://arxiv.org/pdf/2505.20276) (EMNLP 2025).
- **Combination.** No found paper measures the three interventions head-to-head on one problem set with divergence position, difficulty-stratified length effects and energy per correct answer. **Run a Semantic Scholar sweep and record it here before writing "first".**

## 3. Design flaws (fixed)

1. **The error-propagation derivation was wrong.** It summed per-position noise, giving σ√t or σt. The attention output is a softmax-weighted average, so value-noise variance is σ² Σ a², which is at most σ² and shrinks as attention spreads. The compounding that matters is autoregressive (a flipped token corrupts everything after), and that applies to weight quantisation too. The real asymmetry, now in the README §4, is that a quantised KV entry is a persistent position-specific error while weight error is the same perturbation every step.
2. **Length confounded with difficulty.** Longer gold solutions are harder problems, and harder problems are more fragile under any noise. The tertile test now runs within FP16-difficulty strata (pass@k over 5 sampled decodes, labelled in I1), and the first-divergence position, which has no such confound, is the primary statistic.
3. **Per-layer was the wrong unit for H2.** Retrieval and streaming heads coexist in a layer. H2 is now per head on a 6-layer subset.
4. **Sparse-arm speed was going to be fictional.** Without FlashAttention on a T4, sparsification is a top-k mask in eager attention. Accuracy is faithful; speed and energy are simulated and now labelled so.

## 4. Feasibility on free Kaggle/Colab

Accuracy sweeps are realistic. HF `QuantizedCache` supports hqq (int2/4/8) and quanto (int2/4) backends with float16 examples ([HF docs](https://huggingface.co/docs/transformers/en/kv_cache)); it does not support `torch.compile` and adds per-step overhead, which is itself a cost-model finding. Smoke-test both backends on a T4 before planning around them: T4 has no bf16, fp16 Qwen can overflow (1.5B fits in fp32 at ~6 GB if needed). GPTQ/AWQ/bitsandbytes 4-bit all run on sm75. Energy is the weak link: `nvidia-smi`/pynvml power readings exist on a T4 but sample coarsely ([Part-time Power Measurements](https://arxiv.org/html/2312.02741v3): ~5 % per-sample error, up to 70 % naive energy error on A100/H100; T4 not tested), and free-tier GPUs are shared and throttled. Report medians over repeats with uncertainty, or fall back to the tokens/s cost model. The ~100 GPU-hour full budget is 3–4 Kaggle weeks (quota commonly reported as ~30 h/week, not official: [gpuperhour, Sep 2026](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)); the ~18-hour minimum version is one week. vLLM and llama.cpp are alternatives, but per-head sparsity and divergence-token logging are far easier in HF.

## 5. What changed (summary; details in the Amendments appendix)

1. H3 promoted to headline; cost unit changed to energy per correct answer with measured token inflation.
2. H1 reframed as a controlled measurement; first-divergence position primary; tertile test within difficulty strata; difficulty labelling added to I1.
3. Mechanism rewritten from accumulation to persistence.
4. H2 moved to per-head statistics on a 6-layer subset; 28-layer sweep dropped.
5. Sparse-arm speed/energy labelled simulated; energy protocol fixed (medians over 3 repeats with spread; tokens/s fallback); I8 energy repeats added.
6. Jetson, 7B and COAT arms moved to stretch.
7. Citations corrected; target readers reduced from eight faculty to Sanchari Sen (primary) and Yogesh Simmhan (secondary).

## 6. Open items

- [ ] Semantic Scholar sweep for "KV cache quantization chain-of-thought first divergence energy per correct answer" before any "first" claim.
- [ ] Smoke-test HF `QuantizedCache` hqq and quanto backends at int8/int4/int2 on a T4; record which work and their per-step overhead.
- [ ] Confirm pynvml power readings are exposed on Kaggle and Colab T4 sessions; if not, the tokens/s cost model is primary, not fallback.
- [ ] Verify or drop the Yuan et al. 2024 citation.
- [ ] Verify the Sanchari Sen NeurIPS 2022 citation or cite her page's modelling work instead.
- [ ] Check ES-FoMo / ENLSP / AdaptFM-type workshop deadlines for 2027.
