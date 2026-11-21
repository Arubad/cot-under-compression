# Theory for H3: what a cheaper token actually saves

_Written 2026-11-21, before any sweep. Architecture numbers checked against the `config.json` of Qwen2.5-1.5B-Instruct on the Colab VM: 28 layers, 12 query heads, 2 KV heads (GQA), hidden size 1536, head_dim 128, intermediate size 8960, vocab 151,936, tied embeddings._

## 1. Energy per correct answer

For configuration $c$ on a problem set of size $N$, let $a_c$ be accuracy, $L_c$ the mean number of generated tokens, and $e_c$ the energy per generated token (prefill amortised in). Then

$$E_c^{\text{correct}} = \frac{N\,e_c\,L_c}{N\,a_c} = \frac{e_c\,L_c}{a_c}.$$

> **T3.2 (break-even condition).** $c$ beats the reference 0 on energy per correct answer iff
> $$\frac{e_c}{e_0} < \frac{a_c}{a_0}\cdot\frac{L_0}{L_c}.$$
> To first order, an $x$ % per-token saving is cancelled by $x$ % token inflation plus $x$ % relative accuracy loss. This is the quantity the frontier plots, and it makes the argument of [Lian et al.](https://arxiv.org/abs/2606.25519) about chain lengthening exact.

## 2. Per-token cost on a T4: a roofline

Decoding at batch size $B$ is memory-bound when the arithmetic intensity is below the ridge point. T4 peak fp32 is 8.1 TFLOP/s and memory bandwidth is 320 GB/s, so the ridge is about 25 FLOP/B. At B = 8, a step does about $2 \cdot 1.5\text{e}9 \cdot 8 \approx 24$ GFLOP against about 6.8 GB of traffic, roughly 3.5 FLOP/B. So decoding is firmly memory-bound, and

$$\text{latency per step} \approx \frac{W_\text{bytes} + B\cdot \text{KV}_\text{bytes}(t)}{\text{BW}} + c_0.$$

### 2.1 Bytes moved per step

| Quantity | fp32 reference | fp16 | Notes |
|---|---|---|---|
| Non-embedding weights (~1.31 B params) | 5.24 GB | 2.62 GB | |
| Tied embedding / LM head (233 M params) | 0.93 GB | 0.47 GB | read once per step by the LM head |
| **W_bytes total** | **6.17 GB** | **3.09 GB** | |
| KV per token: 28 × 2 (K,V) × 2 heads × 128 × bytes | 57.3 kB | 28.7 kB | GQA makes this 6× smaller than MHA |
| KV per sequence at t = 830 (end of 8-shot prompt) | 47.6 MB | 23.8 MB | prompt length measured in SMOKE-03 |
| KV per sequence at t = 1342 (prompt + 512) | 77.0 MB | 38.5 MB | |

**KV share of per-step bytes**, $B\,\text{KV}(t) / (W + B\,\text{KV}(t))$, at the end of a 512-token chain:

| B | fp32 | |
|---|---|---|
| 1 | 1.2 % | |
| 8 (configured) | 9.1 % | |
| 32 | 28.5 % | |

The crossover length where KV bytes equal weight bytes is $t^* = W/(B\cdot \text{kv/token})$: about 107 k tokens at B = 1 and about 13.4 k at B = 8. Both are far beyond GSM8K.

> **Prediction T3.1.** At the configured batch size, a *real* INT4 KV cache (not fake-quant) saves at most ~8 % of decode bytes against the fp32 reference, and INT8 saves ~7 %. 50 % attention sparsity saves ~4.5 %. The energy axis of the frontier is therefore set almost entirely by the **weight** axis. KV quantisation and sparsity can enter the Pareto set only where they are nearly free in accuracy *and* token inflation. With the current fake-quant implementation, their measured saving is exactly 0, and the simulated saving is bounded by the numbers above.
>
> **Consequence for H3.** H3's "mixed configuration (4-bit W, 8-bit KV, moderate sparsity) is optimal" is, on this hardware and model, nearly a statement about the weight axis alone. If the result is "the optimum is 4-bit W with fp32 KV and no sparsity", the theory expected it. That still refutes the mixed-configuration wording but does not refute non-monotonicity. The regime where H3's mixed optimum becomes plausible is large B or long context. A batch-size or context ablation, which costs no new decodes for the simulated arms, is the natural way to show the crossover.

### 2.2 Weight quantisation against an fp32 reference

The SMOKE-01 decision made fp32 the reference. bitsandbytes leaves embeddings and the LM head unquantised, so

- INT8 (LLM.int8): $W \approx 1.31 + 0.93 = 2.24$ GB → 2.75× fewer bytes than fp32.
- NF4: $W \approx 0.66\,(+\text{scales}) + 0.93 \approx 1.65$ GB → about 3.7× fewer bytes than fp32. The unquantised tied embedding is now **more than half** of the bytes moved.

In an fp16 deployment (fp16 reference, fp16 embeddings) the same quantisers give INT8: 3.09 → 1.78 GB (1.7×) and NF4: 3.09 → 1.13 GB (2.7×). **Every weight-quantisation saving reported against the fp32 reference is inflated by about 1.4× (NF4) to 1.6× (INT8) relative to the deployment-relevant fp16 comparison.** We should report both: measured against fp32, plus a roofline-projected column against fp16. See `open_questions.md` Q2.

### 2.3 Where the roofline will be wrong

> **Prediction T3.3.** bitsandbytes INT8 (LLM.int8 with outlier decomposition) and NF4 are known to decode *slower* than fp16 at small batch on older GPUs, because the dequantisation and outlier matmuls are not bandwidth-bound. On a T4 (sm75), measured tokens/s for INT8 may come out *below* the fp32 reference despite moving 2.75× fewer bytes. Energy per token follows latency, since the GPU sits near its 70 W cap while decoding: SMOKE-03 read 67 W against a 70 W limit. So the roofline is an upper bound on savings, and the weight arm is the one arm where measured and modelled cost can disagree by more than 2×. That disagreement is itself a reportable result for the systems audience.

## 3. Power model

The T4 is capped at 70 W and runs near the cap while decoding. Energy per step is therefore close to $P_\text{cap}\times$ latency, and energy is roughly proportional to wall-clock. NVML exposes a hardware cumulative energy counter on the Colab T4 (`nvmlDeviceGetTotalEnergyConsumption`, verified 2026-11-21). Differencing that counter over a run avoids the sampling error of 10 Hz power polling ([Part-time Power Measurements](https://arxiv.org/abs/2312.02741)). Proposal (Q5): use the counter as the primary energy measurement and keep polling as a cross-check.

## 4. When can a per-problem selector win?

Under greedy decoding each (problem, configuration) pair is deterministic: correct or not, with a known cost. For a cost budget, the oracle selector picks the cheapest correct configuration per problem. Its gain over the best fixed configuration is bounded by the fraction of problems on which the configurations **disagree**. If the cheap configurations fail on a set of problems that is predictable from cheap features (first-token entropy, prompt length, a short-draft agreement), a selector can capture part of the oracle gain. The ≥ 1-point threshold needs $\gtrsim$ 3 problems in 300 to be routed correctly *beyond* what the best fixed configuration already gets right. That is easy for the oracle and hard for a learned selector at n = 300 with 5-fold CV. Expect the minimum-scale selector result to be under-powered, which the pre-registration already allows for ("reported as preliminary").

## Changelog
- 2026-11-21 — first version, with byte counts from the verified model config and NVML capabilities verified on the Colab T4.
