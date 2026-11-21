"""Q11b: separate quantiser grouping from the full-precision residual tail and sink tokens.
Injects quantised copies of the real fp32 prefill cache into a DynamicCache and measures the first
decode-step KL against the reference. Variants: {per_token, per_channel(whole block)} x {tail 0, tail 64} x {sinks 0, 4}."""
import copy, json, os, sys
os.chdir("/content/cot"); sys.path.insert(0, "/content/cot")
import numpy as np, torch
from src.harness.data import load_task
from src.harness.generate import load_model
from src.harness.prompting import build_prompt, chat_wrap
from src.interventions.kv_quant import fake_quant_torch

N = int(os.environ.get("N", "8"))
tok, model = load_model("Qwen/Qwen2.5-1.5B-Instruct", torch.float32, "cuda")
probs = load_task("gsm8k", "configs/gsm8k_300_stratified.json")[:N]

def q(x, bits, scheme, tail, sinks):
    T = x.shape[-2]; end = T - tail; s = sinks
    out = x.clone()
    if end > s:
        out[:, :, s:end] = fake_quant_torch(x[:, :, s:end], bits, -1 if scheme == "per_token" else -2)
    return out

variants = {}
for bits in (4,):
    for scheme in ("per_token", "per_channel"):
        for tail in (0, 64):
            for sinks in (0, 4):
                variants[f"int{bits}_{scheme}_tail{tail}_sink{sinks}"] = (bits, scheme, tail, sinks)
variants["int4_per_token_Konly"] = (4, "per_token", 0, 0)
variants["int4_per_token_Vonly"] = (4, "per_token", 0, 0)
kl = {k: [] for k in variants}; agree = {k: [] for k in variants}
with torch.no_grad():
    for p in probs:
        ids = tok(chat_wrap(tok, build_prompt("gsm8k", p)), return_tensors="pt").input_ids.cuda()
        out = model(ids, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        base = out.past_key_values
        ref = model(nxt, past_key_values=copy.deepcopy(base), use_cache=True).logits[:, -1].float().log_softmax(-1)
        for name, (b, sch, tail, sk) in variants.items():
            c = copy.deepcopy(base)
            for L in c.layers:
                if not name.endswith("Vonly"): L.keys = q(L.keys, b, sch, tail, sk)
                if not name.endswith("Konly"): L.values = q(L.values, b, sch, tail, sk)
            lp = model(nxt, past_key_values=c, use_cache=True).logits[:, -1].float().log_softmax(-1)
            kl[name].append(float((ref.exp() * (ref - lp)).sum())); agree[name].append(bool(ref.argmax() == lp.argmax()))
        print("done", p["id"], flush=True)
res = {k: {"KL_mean": round(float(np.mean(kl[k])), 4), "KL_median": round(float(np.median(kl[k])), 4), "argmax_agree": float(np.mean(agree[k]))} for k in variants}
json.dump(res, open("results/q11b_tail_ablation.json", "w"), indent=1)
for k, v in res.items(): print(f"{k:38s} {v}")
