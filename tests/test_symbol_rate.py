"""
test_symbol_rate.py
-------------------
Tests for src/features/symbol_rate.py against both synthetic BPSK files.

Ground-truth:
  test_bpsk.wav      : symbol_rate = 500  sym/s
  test_bpsk_2400.wav : symbol_rate = 2400 sym/s

The estimator auto-downconverts real passband signals to complex baseband
before applying the cyclostationary method, so the estimates should land
near the true symbol rate.  We allow ~15% tolerance for spectral leakage
and short signal duration.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.wav_loader import load_wav
from preprocessing.normalize import remove_dc_offset
from features.symbol_rate import estimate_symbol_rate

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")


@pytest.fixture(scope="module")
def bpsk_500():
    path = os.path.join(DATA_DIR, "test_bpsk.wav")
    if not os.path.isfile(path):
        pytest.skip("test_bpsk.wav not found — run make_test_bpsk.py first")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    return iq, fs


@pytest.fixture(scope="module")
def bpsk_2400():
    path = os.path.join(DATA_DIR, "test_bpsk_2400.wav")
    if not os.path.isfile(path):
        pytest.skip("test_bpsk_2400.wav not found — run make_test_bpsk_2400.py first")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    return iq, fs


def test_symbol_rate_500(bpsk_500):
    """Estimate should land within 15% of the true 500 sym/s."""
    iq, fs = bpsk_500
    est = estimate_symbol_rate(iq, fs)
    error_pct = abs(est - 500) / 500 * 100
    assert error_pct < 15, (
        f"Symbol rate estimate {est:.1f} Hz is {error_pct:.1f}% off from 500 sym/s"
    )


def test_symbol_rate_2400(bpsk_2400):
    """Estimate should land within 15% of the true 2400 sym/s."""
    iq, fs = bpsk_2400
    est = estimate_symbol_rate(iq, fs)
    error_pct = abs(est - 2400) / 2400 * 100
    assert error_pct < 15, (
        f"Symbol rate estimate {est:.1f} Hz is {error_pct:.1f}% off from 2400 sym/s"
    )


def test_symbol_rate_ordering(bpsk_500, bpsk_2400):
    """The 2400 sym/s estimate should be higher than the 500 sym/s estimate."""
    iq1, fs1 = bpsk_500
    iq2, fs2 = bpsk_2400
    est1 = estimate_symbol_rate(iq1, fs1)
    est2 = estimate_symbol_rate(iq2, fs2)
    assert est2 > est1, (
        f"2400 sym/s estimate ({est2:.1f} Hz) should exceed "
        f"500 sym/s estimate ({est1:.1f} Hz)"
    )

