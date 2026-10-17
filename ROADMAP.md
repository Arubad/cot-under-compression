# Roadmap and progress tracker

Ten weeks, one project at a time, minimum-viable version first. Tick boxes as you go; GitHub renders the counts in the file view. Every configuration run also gets an entry in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

**Start date:** _fill in_ · **Target venue:** an efficient-ML workshop at ICLR 2027 or NeurIPS 2027 (ES-FoMo, ENLSP, AdaptFM-type; 4 pages); IndoML / CODS-COMAD as the Indian venue.

## Milestone 0 — Pre-registration v0.2 (done 2026-10-18)
- [x] Independent review ([`REVIEW.md`](REVIEW.md))
- [x] H3 promoted to headline, H1 reframed, mechanism rewritten, H2 per head; amendments logged in [`PREREGISTRATION.md`](PREREGISTRATION.md)
- [x] Scaffold committed (configs, requirements, src layout, log template)

## Milestone 1 — Weeks 1–2: reading and ownership (no GPU)
- [ ] Foundations block of [`READING.md`](READING.md) read, with a half-page note per paper in `paper/notes/`
- [ ] Project block of [`READING.md`](READING.md) read, same notes
- [ ] Derive by hand why value noise enters as σ² Σ a² and why autoregressive flips compound (`paper/notes/derivation_h1.md`)
- [ ] Read PM-KVQ and KVarN; write one paragraph on what each already shows and what this study adds
- [ ] Rewrite the three hypotheses in your own words in a new amendment (same tests; your wording)
- [ ] Semantic Scholar sweep for prior head-to-head quantisation/sparsity studies with divergence position; record in `REVIEW.md` §6

## Milestone 2 — Weeks 3–4: harness and control
- [ ] `src/harness`: batched greedy generation, 8-shot CoT template, FP16 generation + logit cache (I1)
- [ ] Difficulty labels: 5 sampled FP16 decodes per problem, pass@k stored in `results/I1/difficulty_passk.parquet`
- [ ] 300-problem GSM8K subset stratified by solution length and difficulty, indices committed to `configs/gsm8k_300_stratified.json`
- [ ] `src/interventions`: weight quant (bitsandbytes/GPTQ), KV quant via HF `QuantizedCache` (hqq, quanto), top-k / sink+window masking in eager attention
- [ ] `src/metrics`: accuracy, tertiles within strata, per-token KL, first-divergence token, generated-token count; paired bootstrap
- [ ] `src/cost`: pynvml polling, latency model, energy per correct answer
- [ ] Smallest end-to-end run: 1 config, 50 problems
- [ ] **Expected-to-fail control passes:** zero-change run is bit-exact against the FP16 cache (logged)
- [ ] Smoke-test results recorded: which KV backends run on T4; whether pynvml power is exposed on Kaggle/Colab

## Milestone 3 — Weeks 5–7: minimum-viable sweeps (~18 GPU-h)
- [ ] I2 INT4 weights
- [ ] I3 KV INT8 and INT4 per-token
- [ ] I4 sink + window and top-k at 50 % (speed/energy simulated, labelled)
- [ ] I5 per-head sweep on the 6-layer subset
- [ ] I6 ARC-Challenge control at 200 problems
- [ ] I8 energy repeats (3×) on the headline configs
- [ ] First-divergence histogram per bit-width; within-stratum tertile plot
- [ ] Per-head entropy-vs-drop scatter with ρ
- [ ] Energy-per-correct-answer frontier (measured and simulated points marked)
- [ ] Every run has a complete `EXPERIMENT_LOG.md` entry

## Milestone 4 — Weeks 8–10: selector, analysis and write-up
- [ ] Difficulty-aware per-problem selector (cheap signals: first-token entropy, short-draft agreement); oracle vs learned selector table
- [ ] H1, H2, H3 each marked confirmed / refuted / under-powered against the pre-registered thresholds
- [ ] Negative-results section drafted
- [ ] 4-page workshop paper in `paper/`
- [ ] One-page summary for Sanchari Sen (frontier and selector); one for Yogesh Simmhan (latency model, measured-vs-simulated energy)
- [ ] `python -m src.reproduce --figure frontier` works from cached logs
- [ ] Decide whether the full version (full GSM8K, 3B, all KV schemes, self-consistency) is worth ~80 more GPU-hours

## Stretch (only after Milestone 4)
- [ ] Qwen2.5-3B replication of headline cells
- [ ] I7 self-consistency at equal energy
- [ ] Per-channel vs per-token KV comparison; INT2 KV
- [ ] 7B-in-4-bit headline configs
- [ ] COAT-style rotation vs plain per-channel weight quantisation
- [ ] Jetson Orin replication if hardware is borrowed
