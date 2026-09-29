"""
test_ingestion.py
-----------------
Basic pytest tests for the ingestion pipeline.

Requires the synthetic test file to have been generated first:
    python data/synthetic/make_test_bpsk.py
"""

import os
import sys
import numpy as np
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "synthetic"))

from ingestion.wav_loader import load_wav
from make_test_bpsk import SAMPLE_RATE as EXPECTED_SAMPLE_RATE


SYNTH_WAV = os.path.join(
    os.path.dirname(__file__), "..", "data", "synthetic", "test_bpsk.wav"
)


@pytest.fixture(scope="module")
def loaded_signal():
    """Load the synthetic WAV once for all tests in this module."""
    if not os.path.isfile(SYNTH_WAV):
        pytest.skip(
            "Synthetic test file not found. "
            "Run  python data/synthetic/make_test_bpsk.py  first."
        )
    return load_wav(SYNTH_WAV)


def test_sample_rate_matches(loaded_signal):
    """The loader should report the same sample rate the generator used."""
    _, sample_rate = loaded_signal
    assert sample_rate == EXPECTED_SAMPLE_RATE, (
        f"Expected {EXPECTED_SAMPLE_RATE} Hz, got {sample_rate} Hz"
    )


def test_array_is_non_empty(loaded_signal):
    """The returned array must contain data."""
    iq, _ = loaded_signal
    assert len(iq) > 0, "IQ array is empty"


def test_array_is_complex(loaded_signal):
    """wav_loader must return a complex-valued array."""
    iq, _ = loaded_signal
    assert np.iscomplexobj(iq), (
        f"Expected complex dtype, got {iq.dtype}"
    )
