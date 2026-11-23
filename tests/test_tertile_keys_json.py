"""Regression: metrics.json round-trips by_tertile keys to strings ('0', '1', '2'). The H1 tertile clauses
looked them up as ints and were silently skipped (ratio NaN, refutation clause never evaluated), 2026-11-23."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis import hypotheses
from src.metrics.accuracy import tertile_gap_ratio


def _ci(m, lo, hi):
    return {"mean": m, "lo": lo, "hi": hi, "n": 100}


def _acc(gaps):
    return json.loads(json.dumps({"by_tertile": {t: {"gap_to_fp16": _ci(g, g - 0.08, g + 0.08)} for t, g in enumerate(gaps)},
                                  "by_stratum_tertile": {}}))


def _kv(bits, km, acc):
    return {"_meta": {"cell": {"kv_bits": bits, "kv_scheme": "per_token"}}, "accuracy": acc,
            "divergence": {"km_median_first_divergence": km, "median_first_divergence": km, "exponential_hazard_per_token": 0.01 * (9 - bits)}}


def test_tertile_ratio_reads_string_keys():
    assert abs(tertile_gap_ratio(_acc([0.10, 0.12, 0.30])) - 3.0) < 1e-9


def test_h1_refutes_flat_tertile_gaps_after_json_round_trip():
    flat = _acc([0.11, 0.13, 0.13])
    kv = [_kv(8, _ci(1e9, 216, 1e9), _acc([0, 0, 0])), _kv(4, _ci(8, 7, 11), flat), _kv(2, _ci(3, 2, 3), _acc([0.6, 0.58, 0.46]))]
    w = [{"_meta": {"cell": {"weight_bits": 4}}, "accuracy": _acc([0.09, 0.02, 0.06]), "divergence": {"exponential_hazard_per_token": 0.09}}]
    r = hypotheses.test_h1(kv, w)
    assert r["monotone_non_overlapping"]
    assert abs(r["kv4_tertile_gap_ratio"] - 0.13 / 0.11) < 1e-9
    assert r["verdict"] == "refuted"
