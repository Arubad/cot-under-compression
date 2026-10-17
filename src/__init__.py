"""cot-under-compression — KV-cache, quantisation and sparse attention vs chain-of-thought accuracy.

Planned modules (see src/README.md):
    harness        — generation loop, FP16 cache, bit-exact zero-change control, per-config logging table
    interventions  — weight quantisation, KV-cache quantisation (per-token / per-channel), sparse attention
    metrics        — accuracy, length tertiles within difficulty strata, per-token KL to FP16, first-divergence token, token counts
    cost           — power polling, decode-latency model, roofline check, energy-per-correct-answer frontier, difficulty-aware selector
    analysis       — per-head attention entropy / top-k mass, figures
    reproduce      — one-command regeneration of the frontier figure
"""

__version__ = "0.0.0-prereg"
