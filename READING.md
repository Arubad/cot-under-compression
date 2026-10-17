# Reading list

Ordered so that each item uses the one before it. For each paper write a half-page note in `paper/notes/`: the claim, the one experiment that supports it, and what would have refuted it. The notes become the related-work section.

## Foundations (shared across the three projects)

| Order | Item | Why it matters |
|---|---|---|
| 1 | Keshav, [How to Read a Paper](https://web.stanford.edu/class/ee384m/Handouts/HowtoReadPaper.pdf) | The three-pass method; use it on everything below |
| 2 | Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Attention, KV cache and the softmax, before any project makes sense |
| 3 | Lones, [How to avoid machine learning pitfalls](https://arxiv.org/abs/2108.02497) | The checklist the pre-registration template implements |
| 4 | Bouthillier et al., [Accounting for Variance in ML Benchmarks](https://arxiv.org/abs/2103.03098) | Why seeds, bootstrap CIs and paired comparisons are non-negotiable |
| 5 | Sutton and Barto, [Reinforcement Learning](http://incompleteideas.net/book/the-book-2nd.html), ch. 1–3, 9, 13 | Policy gradient and function approximation; the language the IISc theory groups speak |
| 6 | Peyton Jones, [How to write a great research paper](https://www.microsoft.com/en-us/research/academic-program/write-great-research-paper/) | Read before writing the 4-page version |

## This project

| Order | Item | Why it matters |
|---|---|---|
| 1 | Wei et al., [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903); Cobbe et al., [GSM8K](https://arxiv.org/abs/2110.14168) | The task and the benchmark, including its known contamination issues |
| 2 | Dettmers et al., [LLM.int8()](https://arxiv.org/abs/2208.07339); Dettmers and Zettlemoyer, [The case for 4-bit precision](https://arxiv.org/abs/2212.09720) | Outlier features and why 4-bit weights are usually safe |
| 3 | Frantar et al., [GPTQ](https://arxiv.org/abs/2210.17323); Lin et al., [AWQ](https://arxiv.org/abs/2306.00978) | The two weight-quantisation methods you will run |
| 4 | Hooper et al., [KVQuant](https://arxiv.org/abs/2401.18079); Liu et al., [KIVI](https://arxiv.org/abs/2402.02750) | Per-channel keys vs per-token values; the schemes in the KV arm |
| 5 | Xiao et al., [StreamingLLM / attention sinks](https://arxiv.org/abs/2309.17453); Zhang et al., [H2O](https://arxiv.org/abs/2306.14048) | Sinks and heavy hitters, the two sparsifiers |
| 6 | Xiao et al., [DuoAttention](https://arxiv.org/abs/2410.10819) | Retrieval vs streaming heads; why H2 is per head |
| 7 | Liu et al., [Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823) | The closest existing benchmark; position against it |
| 8 | [PM-KVQ](https://arxiv.org/abs/2505.18610); [KVarN](https://arxiv.org/abs/2606.03458) | Both already argue KV error builds over long CoT; H1 must cite and refine them |
| 9 | Kannan et al., [LevAttention](https://arxiv.org/abs/2410.05462) | The heavy-attention premise H2 tests; note the experiments are on vision |
| 10 | Arya and Simmhan, [LLM inferencing on edge accelerators](https://arxiv.org/abs/2506.09554); [Pagoda](https://arxiv.org/abs/2509.20189) | The measurement style and roofline model the cost section should mirror |
| 11 | [Part-time Power Measurements](https://arxiv.org/html/2312.02741v3) | Why coarse GPU power sampling misreports energy; shapes the energy protocol |

## Related, for the related-work section

[The Quantization Trap](https://arxiv.org/abs/2602.13595); Lian et al., [Quantization Inflates Reasoning](https://arxiv.org/abs/2606.25519); Lotfi et al., [Quantized Reasoning Models Think They Need to Think Longer](https://arxiv.org/abs/2606.00206); [EntroKV](https://proceedings.mlr.press/v306/gao26a.html) (ICML 2026); [When Reasoning Meets Compression](https://arxiv.org/abs/2504.02010) (ICLR 2026); [Semantic Integrity Matters](https://arxiv.org/abs/2502.01941); [SCBench](https://arxiv.org/abs/2412.10319); [Agrawal and Mayer](https://arxiv.org/abs/2607.05399); Li et al., [Quantization Meets Reasoning](https://arxiv.org/abs/2501.03035); [R-KV](https://arxiv.org/abs/2505.24133); [Jin, Wei and Brooks](https://arxiv.org/abs/2505.14733); [Husom et al.](https://arxiv.org/abs/2504.03360); Wang et al., [Self-Consistency](https://arxiv.org/abs/2203.11171); COAT ([ICML 2026 AdaptFM workshop](https://icml.cc/virtual/2026/75331)).
