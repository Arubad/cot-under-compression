# I1 — fp32 reference: accuracy, trajectory cache, difficulty strata, head statistics, latency fit

**Status:** ⏳ prediction written 2026-11-21 (frozen at launch)
**Hypotheses served:** all three. Every later number is relative to this run. · **Theory:** [01 §5](../theory/01_persistence_model_H1.md), [02 §3, §5](../theory/02_heavy_attention_H2.md), [03 §2–3](../theory/03_cost_model_H3.md)

## 1. Question
What does the unperturbed model do on the frozen 300-problem GSM8K subset? This covers the reference trajectories and top-16 logprobs every divergence is measured against, the difficulty strata the H1 tertile test conditions on, the per-head attention statistics H2 regresses on, and the latency/energy baseline H3 divides by.

## 2. Prediction (frozen at launch)
| Quantity | Prediction | Reasoning |
|---|---|---|
| Accuracy (8-shot CoT, greedy, 512 tokens) | **62–74 %** | The Qwen2.5 report gives ~73 % GSM8K for 1.5B-Instruct. The subset is stratified on length, so it is roughly representative; the frozen prompt may cost a few points |
| Truncated at 512 tokens | < 3 % of problems | GSM8K solutions are short; the model rarely loops at greedy |
| Mean generated tokens | 120–220 | |
| Difficulty strata (pass@4, T = 0.7) | easy (4/4) 40–60 %, medium (1–3) 25–40 %, hard (0/4) 8–20 % | A 1.5B model is unstable under sampling on a large minority of problems |
| Accuracy by gold-length tertile | short > mid > long, with a spread of ≥ 10 pt | The length/difficulty confound that motivated stratification (REVIEW §3.2) |
| Zero-change control (50 problems) | bit-exact on all three paths | SMOKE-03 passed on 12 |
| Head statistics | Many heads put > 50 % mass on position 0 (sinks) from layer ~2 on; layers 0–1 have the highest entropy; top-10 % mass > 0.9 for most mid-layer heads | Attention-sink literature; this sets up the leverage-vs-top-k gap (theory/02 §3) |
| Mean power while decoding | 60–70 W | SMOKE-03: 67 W against a 70 W cap |
| Decode step time at B = 8 | 40–120 ms | Roofline floor W/BW = 19 ms. Eager fp32 attention, padding and the 151 k-vocab fp32 LM head add 2–6× |
| Latency-model KV slope | KV term < 10 % of step time at t ≤ 1342 | theory/03 Table 2.1 |
| Energy per problem | 100–500 J | ~50–100 ms/step × 512 steps per batch × 65 W / 8 problems |

**What would surprise us:**
- Accuracy below 55 %. That would point to a prompt or answer-extraction bug, not the model, and should be checked against the extracted answers before anything else.
- Hard stratum below 5 %. The H1 within-stratum test would then have an empty cell, and strata would need merging. That needs an amendment.
- Decode step time near the 19 ms floor. Eager attention would then be far more efficient than expected.

## 3. Design
| Item | Value |
|---|---|
| Commands | `make_subset --config configs/baseline.yaml` once (writes and commits `configs/gsm8k_300_stratified.json`), then `baseline --config configs/baseline.yaml` |
| Model, dtype | Qwen2.5-1.5B-Instruct, fp32, eager attention |
| Problems | 300 (stratified by gold-length tertile); control on 50 |
| Generation | greedy, 512 new tokens, B = 8, top-16 logprobs per step; pass@4 at T = 0.7 for difficulty |
| Energy | 10 Hz NVML polling (current code). Pending Q5: the hardware energy counter |
| Est. GPU-h | 2–5 h. The greedy pass is ~1/5 of it; pass@4 is 4× more decodes |

## 4. Result
_(after run)_

## 5. Interpretation
_(after run)_

## 6. Next
I2–I4 read `results/cache/fp32 reference` artefacts, and I5 picks its layers from `head_stats.csv` using the rule fixed in theory/02 §5.
