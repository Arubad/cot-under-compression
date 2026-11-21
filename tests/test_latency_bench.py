"""Cost model checks (theory/03 §2): the step-model fit recovers known coefficients, and simulated
energy scales the KV term with bits and kept fraction while splitting batch-shared cost over B."""
import numpy as np

from src.harness.latency_bench import fit_step_model, simulated_energy_per_problem


def test_fit_recovers_coefficients():
    a, g, b = 0.05, 0.002, 3e-7
    pts = [{"B": B, "t": t, "step_s": a + g * B + b * B * t} for B in (1, 4, 8, 16) for t in (256, 1024, 4096) if B * t <= 65536]
    fit = fit_step_model(pts)
    assert np.isclose(fit["alpha_s"], a) and np.isclose(fit["gamma_s"], g) and np.isclose(fit["beta_s"], b)
    assert fit["r2"] > 0.999999


def test_simulated_energy_scalings():
    lm = {"source": "latency_bench", "alpha_s": 0.05, "gamma_s": 0.0, "beta_s": 1e-6, "watts": 70.0, "ref_kv_bits": 32}
    ref = simulated_energy_per_problem(lm, 8, 800, 200)
    int8 = simulated_energy_per_problem(lm, 8, 800, 200, kv_bits=8)
    sparse = simulated_energy_per_problem(lm, 8, 800, 200, keep_frac_eff=0.5)
    kv_term = 70.0 * 1e-6 * sum(800 + i for i in range(200))
    assert np.isclose(ref - int8, 0.75 * kv_term)          # int8 of an fp32 cache reads 1/4 of the bytes
    assert np.isclose(ref - sparse, 0.5 * kv_term)
    assert np.isclose(simulated_energy_per_problem(lm, 1, 800, 200) - ref, 70.0 * 0.05 * 200 * (1 - 1 / 8))
    assert np.isnan(simulated_energy_per_problem({"source": "in_run_fit_diagnostic"}, 8, 800, 200))
