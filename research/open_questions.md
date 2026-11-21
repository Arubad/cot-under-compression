# Open questions

Owner **PI** means it needs your decision. Owner **Claude** means I can resolve it without you. Resolved items move to the bottom with the date and resolution.

| ID | Question | Why it matters | Owner | Blocks |
|---|---|---|---|---|
| Q1 | Record the fp32 reference (D-001) and the Colab platform switch (D-002) as dated amendments in `PREREGISTRATION.md` §8? | The pre-registration is the contract, and changes to it should be your call | PI | nothing (recommended before I2) |
| Q2 | Should weight-quantisation costs also be reported against a *roofline-projected fp16* reference? | fp32 inflates weight-arm savings by 1.4–1.6× ([theory/03](theory/03_cost_model_H3.md) §2.2) | PI | I2 write-up |
| Q3 | Add pre-registered secondary analyses for H1 before I3 runs: (a) Kaplan–Meier median T_div (censoring-aware) as the statistic for the existing monotonicity test; (b) Nelson–Aalen hazard shape, KV vs weights; (c) fit exponent β in E[T_div] ∝ σ^-β instead of the fixed 1/σ² | (a) The current statistic is biased by censoring, and this is the **primary** H1 test. (b) It is the only statistic that separates persistence from exposure. (c) Theory predicts 1/σ, not 1/σ² ([theory/01](theory/01_persistence_model_H1.md)) | PI (amendment); Claude (code) | I3 |
| Q4 | Add continuous secondary outcomes (mean KL, KM-median T_div) and a partial correlation controlling for ‖W_O‖ to H2, keeping Δacc as the pass/fail test | Single-head Δacc is near the noise floor and attenuates ρ ([theory/02](theory/02_heavy_attention_H2.md) §4) | PI (amendment); Claude (code) | I5 |
| Q5 | Use the NVML hardware energy counter as the primary energy source, with 10 Hz polling as a cross-check? | More accurate than polling; verified available on the Colab T4 | PI | I1 energy, I8 |
| Q6 | Colab tier: is this a free or a paid (Pro / compute-unit) account? `colab usage` shows 0.00 units | Free-tier sessions can be reclaimed, and the full ~100 GPU-h version may not fit. It sets how aggressively runs are chunked | PI | planning I3/I4 |
| Q7 | `make_subset` must run once, and `configs/gsm8k_300_stratified.json` must be committed before the first sweep | The pre-registration requires the subset to be frozen before sweeps | Claude | I2–I5 |
| Q8 | Labels in harness output still say "fp16" (`accuracy_fp16`, "I1 fp16 greedy") although the reference is fp32 | Cosmetic, but it will confuse readers of logs and figures | Claude | write-up |

## Resolved
- 2026-11-21 — *Is NVML power readable on the T4 we use?* Yes, on Colab: instantaneous power and the cumulative energy counter (REVIEW §6 item 3).
- 2026-11-21 — *Does the sparse path pass the zero-change control after the SMOKE-02 mask fix?* Yes: 0/12 mismatches, max |Δlogprob| = 0.0 (SMOKE-03).
