# Decision log

Append-only. Each entry records a protocol choice that a reviewer could question: what was decided, the alternatives, why, and what it costs us. Choices that change a pre-registered element also need a dated amendment in `PREREGISTRATION.md` §8.

Format: `## D-NNN — YYYY-MM-DD — title`, then *Decision / Alternatives / Reason / Consequence / Amendment needed?*

---

## D-001 — 2026-11-09 — fp32 is the reference precision for Qwen2.5-1.5B
- **Decision:** all 1.5B runs use `float32`.
- **Alternatives:** fp16, which overflows (NaN logprobs, SMOKE-01); bf16, which the T4 (sm75) doesn't support.
- **Reason:** the pre-registration already allows an fp32 fallback when NaNs appear (§5 precision note, 2026-10-18).
- **Consequence:** weight-quantisation savings are measured against fp32 and overstate fp16-deployment savings by about 1.4–1.6× ([theory/03 §2.2](theory/03_cost_model_H3.md)). Report a roofline-projected fp16 column alongside.
- **Amendment needed?** It is covered by the existing fallback clause, but it should be recorded as a dated amendment for clarity (Q1).

## D-002 — 2026-11-21 — compute platform: Google Colab via `colab` CLI (v0.8.0) instead of Kaggle
- **Decision:** run experiments on Colab T4 sessions driven from the local machine with the `colab` CLI. The Kaggle notebooks stay in the repo as an alternative path.
- **Alternatives:** wait for the Kaggle quota.
- **Reason:** Kaggle is occupied. The hardware class is unchanged (Tesla T4 16 GB, driver 580.82.07), so the pre-registered hardware assumption holds.
- **Consequence:** (a) Colab VMs are ephemeral, so every run ends by downloading its `results/` to the local repo (see [`colab_runbook.md`](colab_runbook.md)). (b) The free tier can reclaim the VM, so long sweeps must checkpoint per cell. (c) Colab and Kaggle T4s may differ in clocks and thermal state, so energy numbers from different platforms are not pooled. (d) NVML on Colab exposes a hardware energy counter.
- **Amendment needed?** Recommended: a short "platform" amendment. Hypotheses and tests are unchanged.

## D-003 — 2026-11-21 — library versions: use Colab's stock stack and record it, rather than pinning to 4.x
- **Decision:** run on Colab's preinstalled Python 3.13.16, torch 2.14.1+cu130, transformers 5.22.1 and NumPy 2.x, plus `bitsandbytes` and `nvidia-ml-py`. Versions are printed into every run's log.
- **Alternatives:** pin transformers 4.5x as `requirements.txt` implied.
- **Reason:** on this stack all 11 unit tests pass, including the CPU sparse-control test, and the GPU zero-change control is bit-exact (SMOKE-03). Downgrading torch/transformers on Colab costs minutes per session and risks CUDA mismatches.
- **Consequence:** one compatibility fix was needed (`np.trapz` was removed in NumPy 2). After the first successful baseline, freeze `pip freeze` into `results/env/` and pin the exact versions in `requirements.txt`, so all sweeps share one stack.
- **Amendment needed?** No. This is an implementation detail, and the zero-change control is the guard.

## D-004 — 2026-11-21 — cost model for simulated cells comes from a controlled decode micro-benchmark
- **Decision:** `python -m src.run latency` times decode steps on a (B, t) grid with a random-filled KV cache: B ∈ {1, 4, 8, 16}, t ∈ {256 … 8192}, B·t ≤ 65,536. It fits `step = α + γ·B + β·B·t`. Simulated energy per problem is `P · Σ_t (α/B + γ + β·s_kv·f·t)`, where s_kv = kv_bits / reference width (32) and f = effective kept fraction. f = 1 for single-head masking, because GQA shares KV heads. The old in-run fit is kept as a diagnostic (`latency_model_inrun.json`).
- **Alternatives:** keep regressing in-run per-token time on sequence length.
- **Reason:** SMOKE-04's in-run fit gave a bandwidth of 0.27 GB/s and an intercept of −23 s, which made simulated energies negative. Theory/03 §2.1 explains why: at GSM8K lengths the KV term is ~1–10 % of the bytes moved, so in-run length variation cannot identify it, and prefill time scaling with prompt length contaminates the slope. The old simulated-energy code also scaled bytes as if the reference were fp16 and did not split the batch-shared weight read over B.
- **Consequence:** simulated energies are now traceable to a measured fit with reported r² and residuals. A cell whose benchmark is missing gets `NaN` and `simulated: true`, never the measured wall-clock (fixes a mislabel). Combined weight-quantised + simulated cells are not modelled. The benchmark costs ~5–10 GPU-min.
- **Amendment needed?** No. The pre-registration specifies the model form ("latency(t) ≈ (W + KV(t))/BW + c"), not the fitting procedure. Worth one sentence in the next amendment for transparency.

## D-005 — 2026-11-21 — harness bugs found by SMOKE-04, fixed before any sweep
1. **Sparse path NaN with left padding** (`sparsify.py`). Pad-query rows have no valid key; masking them to −inf gave NaN, which entered the pad's K/V, and 0·NaN poisoned every real token. A *single* sparsified head therefore diverged at token 0. Fix: rows with no valid key are left as in eager. Regression test: 8 padded cases, which fail on the old code and pass now.
2. **Sink positions under left padding** (`sparsify.py`, `sink_window`): sinks were absolute positions 0..s-1, which are pads in a padded row. Fix: sinks are the first *valid* positions.
3. **Unmapped cell keys silently ignored** (`config.py`): `kv_backend`, `kv_sink_tokens` and others landed at the top level, so the smoke `quanto`/`hqq` cells actually ran the fake cache. Fix: all intervention keys mapped, and unknown keys raise. The planned minimum sweeps only use keys that were already mapped, so they were not affected.
4. NumPy-2 `np.trapz` removal (`power.py`); `None`-valued keys in `pareto.aggressiveness`.
- **Consequence:** I1 (launched at `30249b4`, before fixes 1–3) is unaffected. It uses neither masking nor cell overrides, and its zero-change control passed. Every sweep must run at or after the commit carrying these fixes.

## D-006 — 2026-11-21 — recommended answers to Q1–Q5, Q9, Q13, Q14 adopted; harness changed before I2–I8
- **Decision:** the PI asked for the recommended answers to be taken. They are recorded as amendments A1–A9 in `PREREGISTRATION.md` §8. Code changes: a 64-token full-precision tail on the KV cache for both schemes and true per-channel groups (`kv_quant.py`); NF4 with fp32 compute (`weight_quant.py`); the NVML energy counter as primary (`power.py`); KM / Nelson–Aalen / hazard-exponent statistics (`metrics/divergence.py`, `analysis/hypotheses.py`); cost ratios with bandwidth-bound and eager KV references (`latency_bench.py`, `run.py`); sink share, top-1 % mass and ‖W_O‖ head covariates (`sparsify.py`, `headstats` command); and a control that also requires the eager run to reproduce the cached I1 reference.
- **Alternatives:** keep the registered no-tail per-token cell (Q13c); keep the measured-eager KV cost as primary (Q14). Both were rejected for the reasons in the Q13 / Q14 rows.
- **Reason:** see each open-question row. Single-head sparsification now ranks only the selected heads. The mask is identical (unit test), but I5 runs about 3× cheaper in its sparsified layer.
- **Consequence:** config hashes change for every config. I1's results stay valid: the eager path is unchanged, and the control re-checks it against the I1 cache before any sweep. Q6 is answered: the Colab account is free tier (0 compute units). Sessions can be reclaimed, so the run queue checkpoints per cell (`--skip-existing`) and results are downloaded after every job.
- **Amendment needed?** Done (A1–A9).
