"""Q11: is the KV INT4 per-token collapse real? (run on the VM after setup_vm.py)

For N real GSM8K prompts from the frozen subset, prefilled one at a time (no padding), it measures:
  1. relative reconstruction error ||X - Q(X)|| / ||X|| of cached K and V per layer, for
       per_token (one scale per token over head_dim), the harness's per_channel (one scale per channel
       over the *whole completed prefill block*, as FakeQuantCache does it today), and the documented
       per_channel with 64-token groups;
  2. key outlier structure: per layer, max_c |K_c| / median_c |K_c| over channels;
  3. end-to-end: KL(reference || quantised) of the first decode-step distribution, using the real
     FakeQuantCache code path (quantise_prefill=False, i.e. what I3 runs).
Writes results/q11_kv_recon.json and prints a summary.
"""
import json, os, sys

os.chdir("/content/cot")
sys.path.insert(0, "/content/cot")
import numpy as np
import torch

from src.harness.data import load_task
from src.harness.generate import load_model
from src.harness.prompting import build_prompt, chat_wrap
from src.interventions.kv_quant import fake_quant_torch, make_cache_factory

N = int(os.environ.get("N", "8"))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
dev = "cuda"
tok, model = load_model(MODEL, torch.float32, dev)
probs = load_task("gsm8k", "configs/gsm8k_300_stratified.json")[:N]


def kv_layers(cache):
    if hasattr(cache, "layers"):
        return [(l.keys, l.values) for l in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def rel(x, y):
    return float((x - y).float().norm() / x.float().norm())


def per_channel_groups(x, bits, g=64):
    T = x.shape[-2]
    out = x.clone()
    for s in range(0, T - T % g, g):
        out[:, :, s:s + g] = fake_quant_torch(x[:, :, s:s + g], bits, axis=-2)
    return out  # tail (< g tokens) left in full precision, as in the cache


def per_channel_block(x, bits, g=64):
    T = x.shape[-2]
    end = T - T % g
    out = x.clone()
    out[:, :, :end] = fake_quant_torch(x[:, :, :end], bits, axis=-2)
    return out


schemes = {
    "int8_per_token": lambda x: fake_quant_torch(x, 8, -1),
    "int4_per_token": lambda x: fake_quant_torch(x, 4, -1),
    "int2_per_token": lambda x: fake_quant_torch(x, 2, -1),
    "int4_per_channel_block(harness)": lambda x: per_channel_block(x, 4),
    "int4_per_channel_g64(doc)": lambda x: per_channel_groups(x, 4),
}
cache_cells = {
    "int8_per_token": dict(bits=8, scheme="per_token"),
    "int4_per_token": dict(bits=4, scheme="per_token"),
    "int2_per_token": dict(bits=2, scheme="per_token"),
    "int4_per_channel(harness)": dict(bits=4, scheme="per_channel"),
}

rec = {s: {"K": [], "V": []} for s in schemes}
outlier = []
kl = {c: [] for c in cache_cells}
argmax_agree = {c: [] for c in cache_cells}

with torch.no_grad():
    for p in probs:
        ids = tok(chat_wrap(tok, build_prompt("gsm8k", p)), return_tensors="pt").input_ids.to(dev)
        out = model(ids, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        layers = kv_layers(out.past_key_values)
        for s, f in schemes.items():
            rec[s]["K"].append([rel(k, f(k)) for k, _ in layers])
            rec[s]["V"].append([rel(v, f(v)) for _, v in layers])
        outlier.append([float(k.abs().amax(dim=(0, 1, 2)).max() / k.abs().amax(dim=(0, 1, 2)).median()) for k, _ in layers])
        # reference decode step
        ref = model(ids, use_cache=True)
        ref_step = model(nxt, past_key_values=ref.past_key_values, use_cache=True).logits[:, -1].float().log_softmax(-1)
        for c, kw in cache_cells.items():
            cache = make_cache_factory(**kw)()
            o = model(ids, past_key_values=cache, use_cache=True)
            q_step = model(nxt, past_key_values=o.past_key_values, use_cache=True).logits[:, -1].float().log_softmax(-1)
            kl[c].append(float((ref_step.exp() * (ref_step - q_step)).sum()))
            argmax_agree[c].append(bool(ref_step.argmax() == q_step.argmax()))
        print("done", p["id"], ids.shape[-1], "tokens", flush=True)

summary = {
    "n_prompts": N,
    "recon_rel_err_mean_over_layers": {s: {kv: float(np.mean(v)) for kv, v in d.items()} for s, d in rec.items()},
    "recon_rel_err_K_by_layer": {s: np.mean(d["K"], axis=0).round(4).tolist() for s, d in rec.items()},
    "key_outlier_ratio_by_layer(max/median channel absmax)": np.mean(outlier, axis=0).round(2).tolist(),
    "first_decode_step_KL_mean": {c: float(np.mean(v)) for c, v in kl.items()},
    "first_decode_step_KL_median": {c: float(np.median(v)) for c, v in kl.items()},
    "first_decode_step_argmax_agree": {c: float(np.mean(v)) for c, v in argmax_agree.items()},
}
os.makedirs("results", exist_ok=True)
json.dump(summary, open("results/q11_kv_recon.json", "w"), indent=1)
print(json.dumps({k: v for k, v in summary.items() if "by_layer" not in k}, indent=1))
print("K outlier ratio by layer:", summary["key_outlier_ratio_by_layer(max/median channel absmax)"])
print("int4 per_token K err by layer:", summary["recon_rel_err_K_by_layer"]["int4_per_token"])
