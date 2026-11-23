# Open questions

Owner **PI** means it needs your decision. Owner **Claude** means I can resolve it without you. Resolved items move to the bottom with the date and resolution.

| ID | Question | Why it matters | Owner | Blocks |
|---|---|---|---|---|
| Q8 | Labels in harness output still say "fp16" (`accuracy_fp16`, cache file `fp16_<task>.parquet`) although the reference is fp32 | Cosmetic. Progress-bar labels and the control now say "reference", and the control adds `accuracy_reference`. File and key names are kept for compatibility | Claude | write-up |
| Q10b | With left padding, `kv_sink_tokens` and per-channel groups still use absolute positions, so pad K/V can enter a per-channel group's scales and the "sinks" can be pads. Per-token quantisation (the minimum version) is unaffected, because each token has its own scale, and its 64-token tail is always real tokens | Validity of the full-version per-channel comparison and the sink ablation | Claude (before any per-channel or sink run) | full-version I3, sink ablation |
| Q15 | NF4 weights diverge from the reference *earlier* than INT4 KV (KM median 5 vs 8 tokens) but lose less than half the accuracy (5.7 vs 12.3 pt). T_div counts harmless paraphrase, so H1's KV-vs-weights contrast is not visible in the pre-registered primary statistic. Report a secondary answer-level statistic (e.g. divergence that changes the final answer, or KL-weighted hazard)? | Changes what the H1 write-up can claim; must be decided before the KV/weights comparison is written | PI | H1 write-up |
| Q16 | On the primary (bandwidth) KV reading the frontier is flat (best config −3.6 % per correct answer, inside the accuracy CIs); on the eager reading INT8 KV saves 33 %. Headline the eager-reference frontier as the deployment-relevant result on this hardware, stated alongside the registered primary verdict (A8)? | Decides the paper's main figure and framing | PI | write-up |

## Resolved
- 2026-11-21 — **Q1, Q2, Q3, Q4, Q5, Q9, Q13, Q14: the recommended answers were adopted at the PI's instruction** and recorded as amendments A1–A8 (`PREREGISTRATION.md` §8, D-006). Q13 → (a) a 64-token tail for per-token, matched with per-channel. Q14 → bandwidth-bound primary, measured-eager secondary. Q9 → NF4 with fp32 compute. Q3 → KM median is primary, plus Nelson–Aalen and β. Q4 → continuous outcomes and partial ρ given ‖W_O‖ as secondary. Q5 → hardware energy counter. Q2 → fp16-projected secondary column. Q1 → fp32 / Colab amendment.
- 2026-11-21 — Q6: **free tier** (0 compute units, `colab usage`). Long sweeps checkpoint per cell, and results are downloaded after each job.
- 2026-11-21 — Q10 (part): per-channel now uses true 64-token groups (unit-tested). The padding part remains as Q10b.
- 2026-11-21 — Q12: the head recorder now logs the sink share (mass on the first valid key) and top-1 % mass. `python -m src.run headstats` adds them, with ‖W_O[:, head]‖, to `head_stats.csv` without touching I1's columns, which fixed the I5 layer subset.
- 2026-11-21 — Q11: **the per-token INT4 collapse is real, and it is a key effect concentrated on the recent tokens** (DIAG B). Values only: KL 0.005. Keys only: KL 9.0. A 64-token fp tail cuts KL from 9.8 to 0.09, and sinks barely matter. It is not a quantiser bug. It spawns Q13.
- 2026-11-21 — REVIEW §6 item 1 / D-004: the decode cost model is measured (LAT-01, r² 0.996): α = 35.1 ms, β = 4.69 µs per token per sequence.
- 2026-11-21 — *Is NVML power readable on the T4 we use?* Yes, on Colab: instantaneous power and the cumulative energy counter (REVIEW §6 item 3).
- 2026-11-21 — *Does the sparse path pass the zero-change control after the SMOKE-02 mask fix?* Yes: 0/12 mismatches, max |Δlogprob| = 0.0 (SMOKE-03).
- 2026-11-21 — Q7: subset generated and committed (`30249b4`). Tertile boundaries 94 / 140 gold tokens, 100 problems per tertile.
- 2026-11-21 — REVIEW §6 item 2 (HF `QuantizedCache` backends on T4): **answered by SMOKE-05.** Both raise ImportError on stock Colab (`optimum-quanto` and `hqq` are not installed). They are only a cross-check for the fake cache, so installing them is optional.
