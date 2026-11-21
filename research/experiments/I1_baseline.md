# I1 — fp32 reference: accuracy, trajectory cache, difficulty strata, head statistics, latency fit

**Status:** ✅ done. Ran 2026-11-21 08:16–10:03 UTC on Colab T4 at commit `30249b4`, where the prediction below is frozen. Artefacts: `results/archive/I1_baseline/`
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
| Latency-model KV slope | KV term < 10 % of step time at t ≤ 1342 | theory/03 Table 2.1. *Note (post-launch, D-004): the in-run fit cannot identify this; it is tested with the latency micro-benchmark instead. This note does not change the prediction* |
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
Config hash `90fe3aeb6018`. Wall-clock 1 h 47 min, including the model download and the pass@4 sampling.

| Quantity | Prediction | Observed | Verdict |
|---|---|---|---|
| Accuracy (300, greedy, 512 tok) | 62–74 % | **64.7 %** [59.3, 70.0] bootstrap 95 % CI | ✅ in range |
| Truncated at 512 | < 3 % | 3 / 300 = 1.0 % | ✅ |
| Mean generated tokens | 120–220 | 137 (median 115, IQR 85–167) | ✅ |
| Strata (pass@4, T = 0.7) | easy 40–60 / medium 25–40 / hard 8–20 % | easy 33.3 / **medium 48.3** / hard 18.3 % (100 / 145 / 55) | ⚠️ easy below and medium above the range; hard in range. No empty cell |
| Greedy accuracy per stratum | (not predicted) | easy 95.0 %, medium 62.8 %, hard 14.5 % | labels are coherent with greedy |
| Accuracy by gold-length tertile | short > mid > long, spread ≥ 10 pt | **84 / 63 / 47 %**, spread 37 pt | ✅ strongly |
| Zero-change control (50 problems) | bit-exact on all paths | 0/50 cache, 0/50 sparse mismatches, max \|Δlogprob\| = 0.0, no NaN | ✅ |
| Head statistics | top-10 % mass > 0.9 for most mid-layer heads; layers 0–1 highest entropy | top-10 % mass > 0.9 for 83 % of all heads and 97 % of heads in layers 8–20. Highest-entropy layers are **27 (4.37) and 1 (4.34)**, then 2 (3.53) and 26 (3.19). Layer 0 is ordinary (2.47) | ⚠️ partly. Mid-layer concentration holds. High entropy sits at *both ends* of the network, not only layers 0–1. The position-0 sink share was not recorded (the recorder logs only entropy and top-10 % mass), so that part is untested |
| Mean power while decoding | 60–70 W | 67.0 W (NVML polling) | ✅ |
| Decode step time at B = 8 | 40–120 ms | **≈ 130 ms** (median batch wall / longest sequence in the batch) | ❌ slightly above. An upper bound, because batch wall time includes prefill of the ~830-token prompt |
| Latency-model KV slope | KV < 10 % of step time | in-run fit: r² = 0.04, intercept −1.8 s → not identifiable, as D-004 predicted | ⏳ deferred to LAT-01 |
| Energy per problem | 100–500 J | **283 J** (32.4 tok/s aggregate) | ✅ |

Spot check of wrong answers: the first 12 errors all extract a clean numeric prediction (e.g. gold 125 vs pred 2971, gold 14 vs pred 23). There are 0 / 300 missing predictions, so the errors are reasoning errors, not extraction failures. The 54 % on the 50-problem control subset sits inside the sampling noise of the 300-problem estimate: a binomial SE of about 7 pt on n = 50 puts 54 % roughly 1.5 SE below 64.7 %.

## 5. Interpretation
**The reference is sound, and every later number divides by it.** Accuracy lands mid-prediction, the harness is numerically inert on 50 problems, and the answer extractor does not fail. The 64.7 % reference is lower than the ~73 % the Qwen report gives for this model, which is consistent with the frozen 8-shot prompt and greedy decoding. It is not a bug signal.

**The length/difficulty confound is large, so stratification was the right call.** Accuracy falls 37 pt from the short to the long tertile. Any raw "KV harm grows with length" result in H1 would be confounded by this, which justifies the within-stratum test (REVIEW §3.2). The cell sizes (hard × tertile has n = 10, 19, 26) are small. The H1 within-stratum test will therefore have wide intervals in the hard row, and pooling across strata with a stratified estimator (Mantel–Haenszel style) will carry most of the weight. This should be noted when Q3 is decided.

**Strata are skewed toward medium (48 %).** Under T = 0.7 sampling, a 1.5B model is less stable than assumed. Only a third of problems are solved 4/4. This is no threat to the design. It does mean "easy" is a stricter category than expected, and easy-stratum greedy accuracy is 95 %.

**H2 setup, the theory/02 §3 gap is real.** Top-10 % mass is saturated: the median is 0.971, and 97 % of mid-layer heads are above 0.9. Mass therefore has little variance to rank heads by in the middle of the network, whereas entropy still spreads (IQR 1.9–3.2 nats). T2.1 predicted |ρ_mass| ≥ |ρ_H|. That prediction is now at risk *for range-restriction reasons*, not mechanistic ones: a saturated predictor cannot correlate. I5 should report ρ for both and note this restriction in advance. The high entropy at layers 1 and 27 matters for the I5 layer rule (theory/02 §5). If the rule picks layers spanning the entropy range, it will select from both ends, which is what we want.

**H3, the cost side.** The model runs at 67 W and 283 J/problem, which is ~2.1 J/generated token at B = 8. The ≈ 130 ms per step is 6–7× the 19 ms roofline floor, so compute and overhead dominate, not bandwidth (eager fp32 attention, padding, the fp32 151 k-vocab LM head). This **strengthens T3.1**: if the step is not bandwidth-bound, shrinking KV bytes buys even less than the byte-share estimate. It also cuts the other way for weights. Weight quantisation saves energy only if it removes the dominant (non-bandwidth) cost, and bnb dequant adds compute (T3.3). LAT-01 will separate the terms.

## 5a. Corrections to the plan triggered by I1
- Record the attention-sink share (mass on position 0) in the head recorder before I5, so that the remaining I1 head-statistics prediction can be checked from the I4/I5 reference pass. Cost: one extra accumulator. → Q12.
- The "fp16" label is still in artefact names (`fp16_gsm8k.parquet`, `cell_name: "fp16"`, `accuracy_fp16`). The run was fp32 (D-001). Q8 is cosmetic but now visibly confusing.

## 6. Next
I2–I4 read `results/cache/fp32 reference` artefacts, and I5 picks its layers from `head_stats.csv` using the rule fixed in theory/02 §5.
