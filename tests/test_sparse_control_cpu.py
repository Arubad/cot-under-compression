"""CPU check that the custom "sparse" attention function with method=none is bit-exact with HF eager.

Uses a tiny randomly-initialised Qwen2 model (2 layers) with left padding, both for a full
forward pass and for greedy generate() with a KV cache (decode steps have q_len == 1, where the
sdpa mask path may hand a custom implementation None instead of a mask).  This is the CPU twin
of the zero-change control that failed on Kaggle (SMOKE-02, 2026-11-09).
"""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.interventions.sparsify import SparseConfig, apply_sparse  # noqa: E402


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512, attn_implementation="eager")
    model = Qwen2ForCausalLM(cfg).eval()
    return model


def _batch():
    ids = torch.tensor([[5, 6, 7, 8, 9, 10], [0, 0, 11, 12, 13, 14]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]])
    return ids, mask


def test_forward_bit_exact(tiny):
    ids, mask = _batch()
    apply_sparse(tiny, SparseConfig(method="none"))
    with torch.no_grad():
        ref = tiny(input_ids=ids, attention_mask=mask).logits
    apply_sparse(tiny, SparseConfig(method="none", record_entropy=True))
    assert tiny.config._attn_implementation == "sparse"
    with torch.no_grad():
        out = tiny(input_ids=ids, attention_mask=mask).logits
    apply_sparse(tiny, SparseConfig(method="none"))
    valid = mask.bool().unsqueeze(-1)
    assert torch.equal(ref.masked_fill(~valid, 0), out.masked_fill(~valid, 0))


def test_generate_bit_exact(tiny):
    ids, mask = _batch()
    kw = dict(max_new_tokens=8, do_sample=False, pad_token_id=0, output_scores=True, return_dict_in_generate=True)
    apply_sparse(tiny, SparseConfig(method="none"))
    with torch.no_grad():
        ref = tiny.generate(input_ids=ids, attention_mask=mask, **kw)
    apply_sparse(tiny, SparseConfig(method="none", record_entropy=True))
    with torch.no_grad():
        out = tiny.generate(input_ids=ids, attention_mask=mask, **kw)
    apply_sparse(tiny, SparseConfig(method="none"))
    assert torch.equal(ref.sequences, out.sequences)
    for a, b in zip(ref.scores, out.scores):
        assert torch.equal(a, b)


def test_sparse_actually_changes_something(tiny):
    ids, mask = _batch()
    apply_sparse(tiny, SparseConfig(method="none"))
    with torch.no_grad():
        ref = tiny(input_ids=ids, attention_mask=mask).logits
    apply_sparse(tiny, SparseConfig(method="sliding_window", keep_frac=0.5))
    with torch.no_grad():
        out = tiny(input_ids=ids, attention_mask=mask).logits
    apply_sparse(tiny, SparseConfig(method="none"))
    assert not torch.allclose(ref[:, -1], out[:, -1])


@pytest.mark.parametrize("method", ["sliding_window", "sink_window", "topk_heavy_hitter", "leverage"])
@pytest.mark.parametrize("heads", [None, 0])
def test_sparse_with_left_padding_is_finite_and_padding_invariant(tiny, method, heads):
    """SMOKE-04: masking a left-padded batch filled pad-query rows with -inf -> NaN -> 0*NaN poisoned
    every real token, so a single sparsified head diverged at token 0.  The sparsified output of a
    padded sequence must be finite and match the same sequence run without padding."""
    ids, mask = _batch()
    cfg = SparseConfig(method=method, keep_frac=0.5, layers=0 if heads is not None else None, heads=heads, sink_tokens=1)
    apply_sparse(tiny, cfg)
    with torch.no_grad():
        padded = tiny(input_ids=ids, attention_mask=mask).logits
        alone = tiny(input_ids=ids[1:, 2:], attention_mask=mask[1:, 2:]).logits
    apply_sparse(tiny, SparseConfig(method="none"))
    assert torch.isfinite(padded[mask.bool()]).all()
    torch.testing.assert_close(padded[1, 2:], alone[0], atol=1e-4, rtol=1e-4)
