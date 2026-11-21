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
