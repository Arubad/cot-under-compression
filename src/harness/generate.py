"""Greedy chain-of-thought decoding with a per-step top-k logprob recorder.

``run_generation`` decodes a list of problems under one *intervention* (weights /
KV cache / sparsity, see interventions/) and returns one record per problem:
    id, prompt_tokens, gen_tokens (list[int]), text, pred, correct, n_gen,
    topk_ids (n_gen x K), topk_logprobs (n_gen x K), wall_s, first_token_entropy
The FP16 run's records are cached to ``results/cache/fp16_<task>.parquet`` and
every other configuration's first-divergence / KL are computed against them.

Top-k logprobs are recorded on the fly by a LogitsProcessor so the full
(steps x vocab) score tensor is never kept in memory (Qwen's vocab is 151k).
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm

from .prompting import build_prompt, chat_wrap, extract_answer, is_correct


class TopKRecorder:
    """LogitsProcessor that records top-k ids / logprobs per step for every row; scores pass through unchanged."""

    def __init__(self, k: int = 16):
        self.k = k
        self.ids: List[torch.Tensor] = []
        self.lps: List[torch.Tensor] = []
        self.first_entropy: Optional[torch.Tensor] = None

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(scores.float(), dim=-1)
        top = lp.topk(self.k, dim=-1)
        self.ids.append(top.indices.cpu())
        self.lps.append(top.values.cpu())
        if self.first_entropy is None:
            self.first_entropy = -(lp.exp() * lp).sum(-1).cpu()
        return scores

    def stacked(self):
        return torch.stack(self.ids, dim=1).numpy(), torch.stack(self.lps, dim=1).numpy()  # (batch, steps, k)


def load_model(name: str, dtype, device, attn_implementation: str = "eager", quant_config=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(torch_dtype=dtype, attn_implementation=attn_implementation)
    if quant_config is not None:
        kwargs["quantization_config"] = quant_config
        kwargs["device_map"] = {"": 0} if torch.cuda.is_available() else None
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if quant_config is None:
        model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def run_generation(
    model,
    tokenizer,
    problems: List[Dict],
    task: str,
    max_new_tokens: int = 512,
    batch_size: int = 8,
    topk: int = 16,
    cache_factory=None,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int = 0,
    system: Optional[str] = None,
    desc: str = "decode",
) -> List[Dict]:
    """``cache_factory``: callable returning a fresh Cache object (e.g. FakeQuantCache) for each batch, or None."""
    device = next(model.parameters()).device
    prompts = [chat_wrap(tokenizer, build_prompt(task, p), system) for p in problems]
    order = np.argsort([len(p) for p in prompts])  # length-sorted batches
    records: List[Optional[Dict]] = [None] * len(problems)
    from transformers import LogitsProcessorList

    for s in tqdm(range(0, len(order), batch_size), desc=desc):
        idx = order[s : s + batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True).to(device)
        rec = TopKRecorder(topk)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, logits_processor=LogitsProcessorList([rec]), use_cache=True)
        if do_sample:
            torch.manual_seed(seed + s)
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=1.0)
        else:
            gen_kwargs.update(do_sample=False)
        if isinstance(cache_factory, dict):      # HF QuantizedCache backend: generate() kwargs
            gen_kwargs.update(cache_factory)
        elif cache_factory is not None:
            gen_kwargs["past_key_values"] = cache_factory()
        t0 = time.time()
        out = model.generate(**enc, **gen_kwargs)
        wall = time.time() - t0
        new = out[:, enc["input_ids"].shape[1] :].cpu()
        ids_k, lps_k = rec.stacked()
        for j, i in enumerate(idx):
            toks = new[j].tolist()
            # cut at the first eos / pad
            n = len(toks)
            for t, tk in enumerate(toks):
                if tk in (tokenizer.eos_token_id, tokenizer.pad_token_id):
                    n = t
                    break
            toks = toks[:n]
            text = tokenizer.decode(toks, skip_special_tokens=True)
            pred = extract_answer(text, task)
            records[i] = {
                "id": problems[i]["id"],
                "gold": problems[i]["gold"],
                "prompt_tokens": int(enc["attention_mask"][j].sum().item()),
                "gen_tokens": toks,
                "n_gen": n,
                "text": text,
                "pred": pred,
                "correct": bool(is_correct(pred, problems[i]["gold"], task)),
                "topk_ids": ids_k[j, :n].astype(np.int32).tolist(),
                "topk_logprobs": lps_k[j, :n].astype(np.float32).tolist(),
                "first_token_entropy": float(rec.first_entropy[j].item()) if rec.first_entropy is not None else float("nan"),
                "wall_s_batch": wall,
                "batch_size": int(len(idx)),
            }
    return [r for r in records if r is not None]


def records_to_frame(records: List[Dict]):
    import pandas as pd

    return pd.DataFrame(records)


def accuracy(records: List[Dict]) -> float:
    return float(np.mean([r["correct"] for r in records])) if records else float("nan")
