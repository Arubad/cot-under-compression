"""Regression tests for the 2026-11-21 amendments (Q3, Q9, Q10, Q12, Q13, Q14)."""
import numpy as np
import pytest

from src.harness.latency_bench import cost_ratio, decode_cost_s, fp16_projection_ratio
from src.metrics.divergence import exponential_rate, fit_sigma_exponent, km_median, km_median_ci, nelson_aalen_binned

LM = {"source": "latency_bench", "alpha_s": 0.0351, "gamma_s": 0.00064, "beta_s": 4.69e-6, "watts": 66.9, "ref_kv_bits": 32,
      "w_bytes": 6.17e9, "kv_bytes_per_token": 57344.0, "eff_weight_bw_bytes_per_s": 6.17e9 / 0.0351}


# ---------------------------------------------------------------- Q3: censoring-aware T_div

def test_km_median_handles_censoring():
    # 10 problems: 4 diverge early, 6 never diverge (censored at 300) -> survival never <= 0.5
    t = [5, 6, 7, 8] + [300] * 6
    e = [True] * 4 + [False] * 6
    assert km_median(t, e) == float("inf")
    # the naive median of min(T_div, len) would have said 300 / 154: biased
    t2 = [5, 6, 7, 8, 9, 10] + [300] * 4
    e2 = [True] * 6 + [False] * 4
    assert km_median(t2, e2) == 9.0
    ci = km_median_ci(t2, e2, n_boot=200)
    assert ci["lo"] <= 9.0 <= ci["hi"] and np.isclose(ci["frac_censored"], 0.4)


def test_km_equals_empirical_median_without_censoring():
    t = np.arange(1, 12)
    assert km_median(t, np.ones(11, bool)) == 6.0


def test_hazard_and_exponent():
    rng = np.random.default_rng(0)
    rates = {}
    for b, h in ((8, 0.001), (4, 0.017)):
        T = rng.geometric(h, size=4000) - 1
        cens = np.minimum(T, 512)
        rates[b] = exponential_rate(cens, T < 512)
        assert abs(rates[b] - h) / h < 0.1
    fit = fit_sigma_exponent(rates)
    # sigma_4 / sigma_8 = 255/15 = 17 and the hazards differ by 17x -> beta ~ 1
    assert abs(fit["beta"] - 1.0) < 0.1
    na = nelson_aalen_binned(np.minimum(rng.geometric(0.01, 2000) - 1, 512), np.ones(2000, bool), t_max=128)
    assert all(abs(h - 0.01) < 0.004 for h in na["hazard_per_token"][:3])


# ---------------------------------------------------------------- Q14 / Q2: cost ratios

def test_cost_ratio_reference_is_one_and_bandwidth_saving_is_byte_share():
    p, g = [830] * 50, [140] * 50
    assert np.isclose(cost_ratio(LM, 8, p, g, p, g), 1.0)
    bw = cost_ratio(LM, 8, p, g, p, g, kv_bits=4, residual_tokens=0, kv_ref="bandwidth")
    eager = cost_ratio(LM, 8, p, g, p, g, kv_bits=4, residual_tokens=0, kv_ref="eager")
    assert 0.97 < bw < 1.0          # a few % at most: the byte share
    assert eager < bw - 0.1         # measured-eager credits far more
    # token inflation enters linearly-ish: 20 % more tokens costs ~20 % more
    assert 1.15 < cost_ratio(LM, 8, p, [168] * 50, p, g) < 1.25


def test_residual_tail_and_sparsity_reduce_saving():
    p, g = [830] * 10, [140] * 10
    full = cost_ratio(LM, 8, p, g, p, g, kv_bits=4, residual_tokens=0, kv_ref="eager")
    tail = cost_ratio(LM, 8, p, g, p, g, kv_bits=4, residual_tokens=64, kv_ref="eager")
    assert full < tail < 1.0
    sp = cost_ratio(LM, 8, p, g, p, g, keep_frac_eff=0.5, kv_ref="eager")
    assert sp < 1.0
    assert np.isnan(cost_ratio({"source": "in_run"}, 8, p, g, p, g))


def test_fp16_projection_is_cheaper():
    r = fp16_projection_ratio(LM, 8, [830], [140])
    assert 0.45 < r < 0.6


# ---------------------------------------------------------------- Q13 / Q10: FakeQuantCache tail and groups

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _cache(**kw):
    from src.interventions.kv_quant import make_cache_factory

    return make_cache_factory(**kw)()


def _feed(cache, T0=130, steps=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(2, 2, T0, 16, generator=g)
    v = torch.randn(2, 2, T0, 16, generator=g)
    cache.update(k, v, 0)
    ks, vs = [k], [v]
    for _ in range(steps):
        kk, vv = torch.randn(2, 2, 1, 16, generator=g), torch.randn(2, 2, 1, 16, generator=g)
        cache.update(kk, vv, 0)
        ks.append(kk); vs.append(vv)
    return torch.cat(ks, -2), torch.cat(vs, -2)


def _stored(cache):
    return cache._get(0)


def test_per_token_tail_keeps_recent_positions_exact():
    c = _cache(bits=4, scheme="per_token", residual_tokens=64)
    K, V = _feed(c)
    k, v = _stored(c)
    T = K.shape[-2]
    assert torch.equal(k[:, :, T - 64:], K[:, :, T - 64:])          # tail untouched
    assert not torch.allclose(k[:, :, : T - 64], K[:, :, : T - 64])  # older positions quantised
    assert c._n_quantised[0] == T - 64


def test_per_token_without_tail_is_v02_cache():
    c = _cache(bits=4, scheme="per_token", residual_tokens=0)
    K, _ = _feed(c)
    k, _ = _stored(c)
    from src.interventions.kv_quant import fake_quant_torch

    assert torch.allclose(k, fake_quant_torch(K, 4, -1))


def test_per_channel_uses_true_groups():
    from src.interventions.kv_quant import fake_quant_torch

    c = _cache(bits=4, scheme="per_channel", group_size=32, residual_tokens=32)
    K, _ = _feed(c, T0=130, steps=0)
    k, _ = _stored(c)
    # 130 - 32 = 98 -> 3 complete groups of 32 quantised; each group has its own per-channel scale
    assert c._n_quantised[0] == 96
    for s in (0, 32, 64):
        assert torch.allclose(k[:, :, s:s + 32], fake_quant_torch(K[:, :, s:s + 32], 4, -2))
    assert torch.equal(k[:, :, 96:], K[:, :, 96:])


def test_sinks_and_16_bit_passthrough():
    c = _cache(bits=4, scheme="per_token", residual_tokens=0, sink_tokens=4)
    K, _ = _feed(c, steps=0)
    k, _ = _stored(c)
    assert torch.equal(k[:, :, :4], K[:, :, :4])
    from src.interventions.kv_quant import make_cache_factory

    assert make_cache_factory(bits=16) is None


# ---------------------------------------------------------------- Q9: NF4 compute dtype

def test_nf4_defaults_to_fp32_compute():
    pytest.importorskip("bitsandbytes")
    from src.interventions.weight_quant import make_quant_config

    assert make_quant_config(4).bnb_4bit_compute_dtype == torch.float32
    assert make_quant_config(4, compute_dtype="float16").bnb_4bit_compute_dtype == torch.float16


# ---------------------------------------------------------------- Q12 + subset-head masking

def test_single_head_mask_matches_all_head_mask_on_that_head():
    """Ranking only the selected head must give exactly the mask the all-head code gives on that head."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from src.interventions.sparsify import SparseConfig, apply_sparse

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager")
    m = Qwen2ForCausalLM(cfg).eval()
    ids = torch.randint(0, 128, (1, 20))
    attn = m.model.layers[0].self_attn
    store = {}

    def hook(mod, args, kwargs, out):
        store["w"] = out[1]

    h = attn.register_forward_hook(hook, with_kwargs=True)
    with torch.no_grad():
        apply_sparse(m, SparseConfig(method="topk_heavy_hitter", keep_frac=0.5))
        m(ids)
        w_all = store["w"].clone()
        apply_sparse(m, SparseConfig(method="topk_heavy_hitter", keep_frac=0.5, heads=[2]))
        m(ids)
        w_one = store["w"].clone()
        apply_sparse(m, SparseConfig(method="none", record_entropy=True))
        m(ids)
        w_none = store["w"].clone()
        from src.interventions import sparsify as sp

        summ = sp.SPARSE.summary()
    h.remove()
    apply_sparse(m, SparseConfig(method="none"))
    assert torch.equal(w_one[:, 2], w_all[:, 2])
    for hh in (0, 1, 3):
        assert torch.equal(w_one[:, hh], w_none[:, hh])
    rec = summ["0.0"]
    assert 0.0 <= rec["sink_share"] <= 1.0 and rec["top1pct_mass"] <= rec["topk_mass"] + 1e-6
