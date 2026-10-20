"""Weight quantisation via bitsandbytes (runs on a T4, sm75).

16 -> no quantisation (plain fp16).
8  -> LLM.int8() (BitsAndBytesConfig(load_in_8bit=True)).
4  -> NF4 with fp16 compute (BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")).

GPTQ / AWQ pre-quantised checkpoints (e.g. Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4,
Qwen/Qwen2.5-1.5B-Instruct-AWQ) can be used instead by setting
``model.name`` to the checkpoint id and ``weight_method: checkpoint``; they need
optimum + auto-gptq / autoawq, which are not pinned in requirements.txt.
"""
from __future__ import annotations

from typing import Optional


def make_quant_config(bits: int, method: str = "bnb") -> Optional[object]:
    if bits >= 16 or method == "checkpoint":
        return None
    import torch
    from transformers import BitsAndBytesConfig

    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    if bits == 4:
        return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
    raise ValueError(f"unsupported weight bits {bits}")


def weight_bytes(model) -> int:
    """Approximate resident weight bytes (for the latency model's W_bytes term)."""
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    return int(total)
