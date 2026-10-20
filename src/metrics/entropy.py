"""Summaries of the recorded per-head attention statistics."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def stats_frame(summary: Dict) -> pd.DataFrame:
    """``summary`` = SparseConfig.summary(): {"l.h": {layer, head, entropy, topk_mass, n}} -> DataFrame."""
    df = pd.DataFrame(list(summary.values()))
    if df.empty:
        return df
    return df.sort_values(["layer", "head"]).reset_index(drop=True)


def layer_means(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("layer").agg(entropy=("entropy", "mean"), topk_mass=("topk_mass", "mean"), n_heads=("head", "count")).reset_index()
