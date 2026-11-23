"""Regression: degenerate low-bit outputs can contain digit runs too long for a float (q2, KV INT2 per-token
crashed with OverflowError in normalise_number after 30 min of decoding, 2026-11-23)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.harness.data import normalise_number
from src.harness.prompting import extract_answer, is_correct


def test_overflowing_number_does_not_crash():
    huge = "1" * 400
    assert normalise_number(huge) == huge
    assert normalise_number("-" + huge) == "-" + huge
    assert extract_answer("so we get " + huge) == huge
    assert extract_answer("The answer is " + huge) == huge
    assert not is_correct(huge, "18")
    assert is_correct(huge, huge)


def test_inf_and_nan_strings_stay_strings():
    for s in ("inf", "nan", "1e999"):
        assert normalise_number(s) == s
