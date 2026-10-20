"""Tasks: GSM8K (primary), MATH-500 and ARC-Challenge (controls).

A *problem* is ``{"id", "question", "gold", "gold_solution", "choices"?}``.  For
GSM8K the gold answer is the number after ``####``.  Tertile boundaries by
gold-solution token length are fixed once (``configs/gsm8k_tertiles.json``) and a
300-problem subset stratified by tertile is committed (``configs/gsm8k_300_stratified.json``)
so every configuration is scored on identical problems.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

import numpy as np

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")
TERTILES_PATH = os.path.join(CONFIG_DIR, "gsm8k_tertiles.json")
SUBSET_PATH = os.path.join(CONFIG_DIR, "gsm8k_300_stratified.json")


def parse_gsm8k_gold(answer_field: str) -> str:
    m = re.search(r"####\s*(-?[\d,\.]+)", answer_field)
    if not m:
        raise ValueError(f"no #### answer in: {answer_field[:80]}")
    return normalise_number(m.group(1))


def normalise_number(s: str) -> str:
    """'1,234.50' -> '1234.5'; '$18' -> '18'; keeps sign."""
    s = s.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return s
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def load_gsm8k(split: str = "test") -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for i, row in enumerate(ds):
        out.append({"id": f"gsm8k_{split}_{i}", "question": row["question"], "gold": parse_gsm8k_gold(row["answer"]), "gold_solution": row["answer"]})
    return out


def load_math500() -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [{"id": f"math500_{i}", "question": r["problem"], "gold": str(r["answer"]).strip(), "gold_solution": r["solution"]} for i, r in enumerate(ds)]


def load_arc(split: str = "test", n: Optional[int] = None) -> List[Dict]:
    from datasets import load_dataset

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    out = []
    for i, r in enumerate(ds):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        out.append({"id": f"arc_{split}_{i}", "question": r["question"], "choices": dict(zip(labels, texts)), "gold": r["answerKey"], "gold_solution": ""})
        if n and len(out) >= n:
            break
    return out


def load_task(name: str, subset_path: Optional[str] = None, n: Optional[int] = None) -> List[Dict]:
    if name == "gsm8k":
        probs = load_gsm8k("test")
        if subset_path:
            with open(subset_path) as f:
                keep = set(json.load(f).get("ids", []))
            if keep:
                probs = [p for p in probs if p["id"] in keep]
            else:
                print(f"WARNING: {subset_path} has no ids yet; run `python -m src.run make_subset` first. Using the full set.")
    elif name == "math500":
        probs = load_math500()
    elif name == "arc":
        probs = load_arc("test")
    else:
        raise ValueError(name)
    return probs[:n] if n else probs


# ---------------------------------------------------------------- tertiles and the stratified subset

def token_lengths(texts: List[str], tokenizer) -> np.ndarray:
    return np.array([len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts])


def tertile_boundaries(lengths: np.ndarray) -> List[float]:
    return [float(np.quantile(lengths, 1 / 3)), float(np.quantile(lengths, 2 / 3))]


def assign_tertile(lengths: np.ndarray, bounds: List[float]) -> np.ndarray:
    """0 = short, 1 = medium, 2 = long."""
    lengths = np.asarray(lengths, dtype=float)
    return (lengths > bounds[0]).astype(int) + (lengths > bounds[1]).astype(int)


def stratified_subset(ids: List[str], strata: np.ndarray, n_total: int, seed: int) -> List[str]:
    """Equal counts per stratum (as far as possible), seeded."""
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    levels = sorted(set(strata.tolist()))
    per = n_total // len(levels)
    chosen: List[str] = []
    for lv in levels:
        pool = [i for i, s in zip(ids, strata) if s == lv]
        take = min(per, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    # top up if some stratum was short
    if len(chosen) < n_total:
        rest = [i for i in ids if i not in set(chosen)]
        chosen.extend(rng.choice(rest, size=n_total - len(chosen), replace=False).tolist())
    return sorted(chosen, key=lambda s: ids.index(s))


def make_subset(tokenizer_name: str, n_total: int = 300, seed: int = 0) -> Dict:
    """Compute tertile boundaries on the full GSM8K test set and write the stratified subset. Idempotent."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    probs = load_gsm8k("test")
    lens = token_lengths([p["gold_solution"] for p in probs], tok)
    bounds = tertile_boundaries(lens)
    tert = assign_tertile(lens, bounds)
    ids = [p["id"] for p in probs]
    chosen = stratified_subset(ids, tert, n_total, seed)
    with open(TERTILES_PATH, "w") as f:
        json.dump({"tokenizer": tokenizer_name, "boundaries": bounds, "n": len(probs)}, f, indent=2)
    with open(SUBSET_PATH, "w") as f:
        json.dump({"tokenizer": tokenizer_name, "seed": seed, "n": len(chosen), "ids": chosen, "tertile": {i: int(t) for i, t in zip(ids, tert) if i in set(chosen)}}, f, indent=2)
    return {"boundaries": bounds, "n_subset": len(chosen), "counts": np.bincount(tert[[ids.index(c) for c in chosen]]).tolist()}


def load_tertiles() -> Dict:
    with open(TERTILES_PATH) as f:
        return json.load(f)
