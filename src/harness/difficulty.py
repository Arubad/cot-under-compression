"""Difficulty labels and the cheap per-problem features used by the selector.

difficulty = FP16 pass@4 at temperature 0.7 (4 sampled decodes):
    easy = 4/4 correct, medium = 1-3/4, hard = 0/4.
Features (all available *before* committing to a configuration):
    q_tokens          question length in tokens
    n_numbers         count of numbers in the question
    first_entropy     entropy of the FP16 first-token distribution (from the greedy cache)
    fp16_len          FP16 greedy solution length (only for the oracle-ish upper bound; flagged)
"""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd


def pass_at_k_labels(model, tokenizer, problems: List[Dict], task: str, k: int = 4, temperature: float = 0.7, max_new_tokens: int = 512, batch_size: int = 8, seed: int = 0) -> pd.DataFrame:
    from .generate import run_generation

    hits = {p["id"]: 0 for p in problems}
    for s in range(k):
        recs = run_generation(model, tokenizer, problems, task, max_new_tokens=max_new_tokens, batch_size=batch_size, topk=1, do_sample=True, temperature=temperature, seed=seed * 1000 + s, desc=f"pass@k sample {s+1}/{k}")
        for r in recs:
            hits[r["id"]] += int(r["correct"])
    df = pd.DataFrame({"id": list(hits.keys()), "n_correct": list(hits.values()), "k": k})
    df["stratum"] = stratum_from_hits(df["n_correct"].values, k)
    return df


def stratum_from_hits(n_correct: np.ndarray, k: int) -> np.ndarray:
    n = np.asarray(n_correct)
    out = np.full(len(n), "medium", dtype=object)
    out[n == k] = "easy"
    out[n == 0] = "hard"
    return out


def cheap_features(problems: List[Dict], fp16_records: List[Dict], tokenizer) -> pd.DataFrame:
    by_id = {r["id"]: r for r in fp16_records}
    rows = []
    for p in problems:
        r = by_id.get(p["id"])
        rows.append(
            {
                "id": p["id"],
                "q_tokens": len(tokenizer(p["question"], add_special_tokens=False)["input_ids"]),
                "n_numbers": len(re.findall(r"\d+(?:\.\d+)?", p["question"])),
                "q_chars": len(p["question"]),
                "first_entropy": r["first_token_entropy"] if r else float("nan"),
                "fp16_len": r["n_gen"] if r else float("nan"),
            }
        )
    return pd.DataFrame(rows)
