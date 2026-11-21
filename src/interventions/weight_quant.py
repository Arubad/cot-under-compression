"""Weight quantisation via bitsandbytes (runs on a T4, sm75).

16 -> no quantisation (plain fp16).
8  -> LLM.int8() (BitsAndBytesConfig(load_in_8bit=True)).
4  -> NF4 (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")) with fp32 compute by
      default (amendment 2026-11-21, Q9): the reference is fp32, and fp16 compute would confound weight
      quantisation with fp16 activations, which overflow in Qwen2.5-1.5B (SMOKE-01).
      ``compute_dtype="float16"`` gives the fp16-compute variant.  LLM.int8 always casts activations
      to fp16, so the INT8 arm is full-version only and reported with that caveat.

GPTQ / AWQ pre-quantised checkpoints (e.g. Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4,
Qwen/Qwen2.5-1.5B-Instruct-AWQ) can be used instead by setting
``model.name`` to the checkpoint id and ``weight_method: checkpoint``; they need
optimum + auto-gptq / autoawq, which are not pinned in requirements.txt.
"""
from __future__ import annotations

from typing import Optional


def make_quant_config(bits: int, method: str = "bnb", compute_dtype: str = "float32") -> Optional[object]:
    if bits >= 16 or method == "checkpoint":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype={"float32": torch.float32, "float16": torch.float16}[compute_dtype], bnb_4bit_use_double_quant=False)
    raise ValueError(f"unsupported weight bits {bits}")


def weight_bytes(model) -> int:
    """Approximate resident weight bytes (for the latency model's W_bytes term)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return int(total)
