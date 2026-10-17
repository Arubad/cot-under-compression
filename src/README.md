# src/ — planned modules

- `harness/` — batched greedy generation with a fixed 8-shot CoT template; FP16 generation cache (token ids + logits, computed once); the bit-exact zero-change control that runs on every commit; one-row-per-configuration table of accuracy, tokens/s, peak memory and energy.
- `interventions/` — `weight_quant.py` (INT8/INT4, GPTQ/AWQ-style; COAT-style rotation is a stretch option), `kv_quant.py` (INT8/INT4/INT2, per-token vs per-channel, FP16 prefill then quantise, sink protection), `sparse_attention.py` (sliding window, sink + window, top-k heavy-hitter by accumulated attention, leverage-score sampling from LevAttention's description; single-head, single-layer and all-layer modes; on a T4 this is top-k masking in eager/SDPA attention, so its speed/energy is simulated from the latency model).
- `metrics/` — accuracy with answer extraction, accuracy by gold-solution-length tertile within FP16-difficulty strata, per-token KL to the FP16 trajectory, first-divergence token position (primary for H1), generated-token counts, paired bootstrap CIs.
- `cost/` — NVML power polling (medians over repeats with spread; tegrastats only in the Jetson stretch arm), decode-latency model latency(t) ≈ (W_bytes + KV_bytes(t))/BW + c with roofline check, energy-per-correct-answer frontier (token inflation included), difficulty-aware per-problem configuration selector.
- `analysis/` — per-head attention entropy and top-k mass on the FP16 pass, entropy-vs-drop regression, all figures.
- `utils/` — config hashing, deterministic generation settings, W&B wrappers.
- `reproduce.py` — `python -m src.reproduce --figure frontier` from cached logs.
