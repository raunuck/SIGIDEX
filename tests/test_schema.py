"""
test_schema.py
--------------
Tests for src/schema.py — the Bucket 1 report builder.

Verifies report structure and that modulation type matches ground truth
for each synthetic file.
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.wav_loader import load_wav
from schema import build_bucket1_report

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bpsk_report():
    path = os.path.join(DATA_DIR, "test_bpsk.wav")
    if not os.path.isfile(path):
        pytest.skip("test_bpsk.wav not found")
    iq, fs = load_wav(path)
    return build_bucket1_report(iq, fs)


@pytest.fixture(scope="module")
def qpsk_report():
    path = os.path.join(DATA_DIR, "test_qpsk.wav")
    if not os.path.isfile(path):
        pytest.skip("test_qpsk.wav not found")
    iq, fs = load_wav(path)
    return build_bucket1_report(iq, fs)


@pytest.fixture(scope="module")
def fsk_report():
    path = os.path.join(DATA_DIR, "test_fsk.wav")
    if not os.path.isfile(path):
        pytest.skip("test_fsk.wav not found")
    iq, fs = load_wav(path)
    return build_bucket1_report(iq, fs)


def test_bucket1_report_structure(bpsk_report):
    """Verify that all required Bucket 1 schema keys are present."""
    for key in ("rf", "modulation", "coding"):
        assert key in bpsk_report, f"Missing top-level key: {key}"
    for key in ("sampling_rate_hz", "bandwidth_hz", "snr_db"):
        assert key in bpsk_report["rf"], f"Missing rf key: {key}"
    for key in ("type", "symbol_rate_sps", "confidence"):
        assert key in bpsk_report["modulation"], f"Missing modulation key: {key}"
    assert "interleaver" in bpsk_report["coding"]
    assert "fec" in bpsk_report["coding"]


def test_bucket1_report_values(bpsk_report):
    """Verify values match ground-truth for synthetic BPSK signal."""
    # BPSK ground truth: 44.1 kHz sample rate, 500 sym/s, PSK modulation
    assert bpsk_report["rf"]["sampling_rate_hz"] == 44100
    assert bpsk_report["modulation"]["type"] == "PSK"
    assert abs(bpsk_report["modulation"]["symbol_rate_sps"] - 500.0) < 25.0
    assert bpsk_report["modulation"]["confidence"] > 0.5
    assert bpsk_report["rf"]["snr_db"] > 10.0


def test_report_has_top_level_keys(bpsk_report):
    for key in ("rf", "modulation", "coding"):
        assert key in bpsk_report, f"Missing top-level key: {key}"


def test_rf_section_keys(bpsk_report):
    rf = bpsk_report["rf"]
    for key in ("sampling_rate_hz", "bandwidth_hz", "snr_db"):
        assert key in rf, f"Missing rf key: {key}"


def test_modulation_section_keys(bpsk_report):
    mod = bpsk_report["modulation"]
    for key in ("type", "symbol_rate_sps", "confidence"):
        assert key in mod, f"Missing modulation key: {key}"


def test_coding_section_is_unknown(bpsk_report):
    coding = bpsk_report["coding"]
    assert coding["interleaver"]["type"] == "unknown"
    assert coding["interleaver"]["confidence"] == 0.0
    assert coding["fec"]["type"] == "unknown"
    assert coding["fec"]["confidence"] == 0.0


# ── Modulation type correctness ──────────────────────────────────────────────

def test_bpsk_report_type(bpsk_report):
    assert bpsk_report["modulation"]["type"] == "PSK", (
        f"BPSK report: expected PSK, got {bpsk_report['modulation']['type']}"
    )


def test_qpsk_report_type(qpsk_report):
    assert qpsk_report["modulation"]["type"] == "PSK", (
        f"QPSK report: expected PSK, got {qpsk_report['modulation']['type']}"
    )


def test_fsk_report_type(fsk_report):
    assert fsk_report["modulation"]["type"] == "FSK", (
        f"FSK report: expected FSK, got {fsk_report['modulation']['type']}"
    )


# ── Sanity checks on values ─────────────────────────────────────────────────

def test_sampling_rate_correct(bpsk_report):
    assert bpsk_report["rf"]["sampling_rate_hz"] == 44100


def test_bandwidth_is_positive(bpsk_report, fsk_report):
    assert bpsk_report["rf"]["bandwidth_hz"] > 0
    assert fsk_report["rf"]["bandwidth_hz"] > 0


def test_snr_is_finite(bpsk_report, fsk_report):
    assert np.isfinite(bpsk_report["rf"]["snr_db"])
    assert np.isfinite(fsk_report["rf"]["snr_db"])


def test_confidence_in_range(bpsk_report, qpsk_report, fsk_report):
    for label, report in [("BPSK", bpsk_report), ("QPSK", qpsk_report),
                          ("FSK", fsk_report)]:
        conf = report["modulation"]["confidence"]
        assert 0.0 < conf <= 1.0, f"{label} confidence {conf} out of range"
