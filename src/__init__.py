"""cot-under-compression: KV-cache quantisation, weight quantisation and sparse attention vs chain-of-thought accuracy.

Packages: common (config/seed/log/stats), harness (data, prompting, generation, difficulty, power, control),
interventions (weight_quant, kv_quant, sparsify), metrics (accuracy, divergence, entropy), frontier (pareto,
selector), analysis (pre-registered tests, figures).  Entry point: python -m src.run
"""

__version__ = "0.2.0-code"
