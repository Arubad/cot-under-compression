# DIAG — decode cost model (LAT-01) and the KV INT4 per-token collapse (Q11)

**Status:** ✅ done 2026-11-21, Colab T4 `cot-t4`, commit `be82d77` (LAT-01) and `bea36fb` (Q11 scripts `tools/colab/q11_kv_recon.py` and `q11b_tail_ablation.py`)
**Hypotheses served:** H3 (cost model), H1 (I3 design) · **Theory:** [03 §2](../theory/03_cost_model_H3.md), [01](../theory/01_persistence_model_H1.md)

These are diagnostics, not pre-registered tests. Each had a prediction written down before it ran: T3.1 and the I1 latency row for LAT-01, and the Q11 entry in `open_questions.md` for Q11.

---

## A. LAT-01 — decode-step micro-benchmark

### Prediction (written before the run)
- T3.1: at GSM8K lengths, KV is ~1–10 % of per-sequence decode **bytes**, so the KV term is < 10 % of step time at t ≤ 1342 (I1 page, latency row).
- The weight term sits near the roofline. W/BW is 19 ms at 320 GB/s.

### Result
Fit of `step = α + γ·B + β·B·t` over B ∈ {1, 4, 8, 16} and t ∈ {256 … 8192}, fp32, eager attention, random-filled DynamicCache. **r² = 0.996**, max relative residual 24 %.

| Term | Value | Meaning |
|---|---|---|
| α | **35.1 ms** | weight-read floor. Effective weight BW 176 GB/s, which is 55 % of the T4's 320 GB/s |
| γ | 0.64 ms per sequence | per-sequence overhead (LM head, sampling) |
| β | **4.69 µs per token per sequence** | KV term. Effective KV BW **12 GB/s**, 14× worse per byte than the weight read |
| Power | 66.9 W | NVML, same as I1 |

KV share of step time, `βBt / step`:

| B \ context t | 500 | 900 (≈ GSM8K mean) | 1342 |
|---|---|---|---|
| 1 | 6 % | 11 % | 15 % |
| 8 | 32 % | **46 %** | 56 % |
| 16 | 45 % | 60 % | 69 % |

### Interpretation
**T3.1 is refuted for time. It still holds for bytes.** KV is ~9 % of the bytes at B = 8 but ~46 % of the step time, because eager attention moves KV ~14× less efficiently than the weight GEMV. Two things cause this. `repeat_kv` materialises each of the 2 KV heads 6× to match the 12 query heads, and the decode-time attention is a chain of small unfused kernels. The theory assumed that every byte costs the same, and that assumption is false for this implementation.

**This matters for the H3 headline, and the model alone cannot settle it.** The simulated energy for a KV-quantised cell scales β by `kv_bits / 32`, which assumes a low-bit cache makes this *eager* attention term proportionally cheaper. Two readings of "the KV saving" are both defensible:
1. **Measured-eager reference (current code).** INT4 KV cuts β by 8×, saving roughly 40 % of decode time at B = 8. KV compression then looks very attractive.
2. **Bandwidth-bound reference.** A fused decoding kernel (flash-decoding, no `repeat_kv`) would already make the fp32 KV term ~14× smaller, so quantising it buys only the byte share, about 9 % at B = 8.

Reading 1 credits KV quantisation with savings that come mostly from fixing an inefficient kernel. Reading 2 is the hardware-honest number. These differ by 4–5× in exactly the quantity H3's mixed-optimum claim depends on. → **Q14 (PI)**. My recommendation: report both, use reading 2 as primary, and state it in the amendment.

The I1 step (≈ 130 ms) is above the model's 74 ms at B = 8, t = 900. That gap is consistent with I1's per-batch wall time including the ~830-token prefill and left-padding, both of which add context tokens the model charges for.

---

## B. Q11 — is the KV INT4 per-token collapse real?

### Prediction (written before the run, `open_questions.md` Q11)
The collapse is plausibly real. Qwen2.5's QKV biases give key-outlier channels, so a per-token scale over head_dim is set by outliers and the remaining channels are crushed. If true, per-token reconstruction error on real keys should be much larger than per-channel.

### Result 1: reconstruction and first decode step, real fp32 prefill caches, 8 subset problems
| Scheme (INT4 unless noted) | rel. err K | rel. err V | first-step KL (nats, mean) | top-1 agrees |
|---|---|---|---|---|
| INT8 per-token | 0.009 | 0.007 | 0.0009 | 8/8 |
| INT4 per-token | 0.153 | 0.124 | **6.35** | **0/8** |
| INT2 per-token | 0.605 | 0.577 | 11.9 | 0/8 |
| INT4 per-channel (harness: one scale per channel over the whole block, 64-token fp tail) | 0.065 | 0.115 | 0.027 | 7/8 |
| INT4 per-channel, 64-token groups (as documented) | 0.045 | 0.082 | — | — |

Key outlier ratio (max ÷ median channel |K|): 68 at layer 0 and 22 at layer 1, then 2.4–13 elsewhere.

The per-token K error is only 2.3× the per-channel error, yet the KL is 240× larger. Reconstruction error alone does not explain the collapse.

### Result 2: ablation, injecting quantised copies of the real cache (8 problems, INT4)
| Variant | KL mean | KL median | top-1 agrees |
|---|---|---|---|
| per-token, no tail, no sinks | 9.79 | 10.2 | 0/8 |
| per-token, 4 fp sinks | 8.85 | 10.1 | 1/8 |
| **per-token, 64-token fp tail** | **0.087** | 0.021 | 7/8 |
| per-token, keys only | 9.00 | 10.2 | 1/8 |
| per-token, values only | 0.005 | 0.002 | 8/8 |
| per-channel, no tail | 0.032 | 0.007 | 7/8 |
| per-channel, 64-token tail (≈ harness) | 0.011 | 0.001 | 8/8 |
| per-channel, tail + 4 sinks | 0.0016 | 0.0002 | 8/8 |

### Interpretation
- **The collapse is real and is a key effect.** Values alone cost KL 0.005, keys alone cost KL 9.0.
- **Its mechanism is not what Q11 guessed.** Leaving the most recent 64 tokens' keys in full precision cuts per-token KL ~110×, from 9.8 to 0.09. Sinks barely matter. Per-token quantisation of the *recent* keys, which carry most of the first decode step's attention mass (the chat-template tokens `<|im_start|>assistant\n`), destroys the step. Old keys tolerate it.
- A consistent mechanism: within one token's key, a few outlier channels set the range, so the ordinary channels get ~1–2 levels. That wipes the content part of q·k for exactly the keys the query attends to most. This is theory/01's "anchor" case, where the error lands on high-attention positions, so a per-error-unit hazard is huge.
- Grouping still matters, but less. Without a tail, per-channel beats per-token by ~300× in KL (0.032 vs 9.8). With a tail, per-channel beats per-token by ~8× (0.011 vs 0.087).
- **Q10 is confirmed quantitatively.** The harness's per-channel cell has a 64-token full-precision tail that per-token lacks, and the tail alone is worth a ~110× KL reduction for per-token. The pre-registered "per-token vs per-channel" comparison therefore mostly measures *tail vs no tail*.
- The injected-cache KL (9.8) is higher than the in-harness one (6.35). One plausible cause is that the harness path quantises the prefill but hands the prefill pass fp keys. Both are complete collapses, so this does not change the conclusion.
- Side finding (code vs doc): the harness's per-channel scheme uses **one scale per channel over the whole completed block**, e.g. 768 prefill tokens, not 64-token groups as the docstring says. Its K error is 0.065 against 0.045 for true 64-token groups. → added to Q10.

### Consequence for I3 and H1 (→ Q13, PI)
As configured, the minimum-version I3 cell (INT4 per-token, no tail) will produce near-zero accuracy and T_div ≈ 1 on essentially every problem. H1's bit-width monotonicity would "pass" trivially, but the length-gap and hazard-shape tests have no dynamic range, because everything diverges at token 1. Options:
- (a) Add a 64-token full-precision residual tail to per-token, as KIVI and HF `QuantizedCache` both do. Recommended: it is standard practice and it gives a non-degenerate INT4 cell.
- (b) Switch the minimum version to KIVI-style per-channel K / per-token V, with tails matched across schemes.
- (c) Keep the pre-registered design and report the collapse as a finding, adding INT4 + tail as a secondary cell.

Whatever is chosen, matching the tail across schemes fixes the Q10 confound.
