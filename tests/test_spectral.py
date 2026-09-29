"""
test_spectral.py
----------------
Tests for src/features/spectral.py against both synthetic BPSK files.

Ground-truth expectations:
  test_bpsk.wav      : carrier=5000 Hz, symbol_rate=500  -> BW ~ 2*500  = ~1000 Hz main lobe
  test_bpsk_2400.wav : carrier=8000 Hz, symbol_rate=2400 -> BW ~ 2*2400 = ~4800 Hz main lobe
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.wav_loader import load_wav
from preprocessing.normalize import remove_dc_offset
from features.spectral import compute_psd, estimate_bandwidth, estimate_snr

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bpsk_500():
    """Load the 500 sym/s BPSK file."""
    path = os.path.join(DATA_DIR, "test_bpsk.wav")
    if not os.path.isfile(path):
        pytest.skip("test_bpsk.wav not found — run make_test_bpsk.py first")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    return iq, fs


@pytest.fixture(scope="module")
def bpsk_2400():
    """Load the 2400 sym/s BPSK file."""
    path = os.path.join(DATA_DIR, "test_bpsk_2400.wav")
    if not os.path.isfile(path):
        pytest.skip("test_bpsk_2400.wav not found — run make_test_bpsk_2400.py first")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    return iq, fs


# ── compute_psd ──────────────────────────────────────────────────────────────

def test_psd_shape(bpsk_500):
    iq, fs = bpsk_500
    freqs, psd = compute_psd(iq, fs)
    assert len(freqs) == len(psd), "freqs and psd must have the same length"
    assert len(psd) > 0, "PSD is empty"


def test_psd_has_energy_at_carrier_500(bpsk_500):
    """PSD should peak near 5000 Hz for the 500 sym/s file."""
    iq, fs = bpsk_500
    freqs, psd = compute_psd(iq, fs)
    peak_freq = freqs[np.argmax(psd)]
    # The mono WAV is real-valued -> energy at +5000 and -5000 Hz.
    assert abs(abs(peak_freq) - 5000) < 500, (
        f"PSD peak at {peak_freq} Hz, expected near +/-5000 Hz"
    )


def test_psd_has_energy_at_carrier_2400(bpsk_2400):
    """PSD should peak near 8000 Hz for the 2400 sym/s file."""
    iq, fs = bpsk_2400
    freqs, psd = compute_psd(iq, fs)
    peak_freq = freqs[np.argmax(psd)]
    assert abs(abs(peak_freq) - 8000) < 1000, (
        f"PSD peak at {peak_freq} Hz, expected near +/-8000 Hz"
    )


# ── estimate_bandwidth ───────────────────────────────────────────────────────

def test_bandwidth_500(bpsk_500):
    """
    500 sym/s BPSK with rect pulses: main lobe ~1000 Hz.
    At -20 dB threshold we expect roughly 1000-3000 Hz (includes sidelobes).
    The signal is real-valued so there's energy at both +5000 and -5000,
    making the full occupied span larger.  We just check it's in a reasonable
    ballpark and not wildly off.
    """
    iq, fs = bpsk_500
    freqs, psd = compute_psd(iq, fs)
    bw = estimate_bandwidth(freqs, psd, threshold_db=-20)
    # Real-valued signal -> mirrored spectrum.  Total occupied span at -20 dB
    # will cover both mirror images.  Accept anything in [500, 15000].
    assert 500 < bw < 15000, f"Bandwidth {bw} Hz out of expected range"


def test_bandwidth_2400(bpsk_2400):
    """
    2400 sym/s BPSK: main lobe ~4800 Hz.  At -20 dB, wider due to sidelobes.
    """
    iq, fs = bpsk_2400
    freqs, psd = compute_psd(iq, fs)
    bw = estimate_bandwidth(freqs, psd, threshold_db=-20)
    assert 2000 < bw < 40000, f"Bandwidth {bw} Hz out of expected range"


def test_bandwidth_2400_wider_than_500(bpsk_500, bpsk_2400):
    """The higher symbol-rate signal should have wider bandwidth."""
    _, fs1 = bpsk_500
    iq1, _ = bpsk_500
    freqs1, psd1 = compute_psd(iq1, fs1)
    bw1 = estimate_bandwidth(freqs1, psd1, threshold_db=-20)

    iq2, fs2 = bpsk_2400
    freqs2, psd2 = compute_psd(iq2, fs2)
    bw2 = estimate_bandwidth(freqs2, psd2, threshold_db=-20)

    assert bw2 > bw1, (
        f"2400 sym/s BW ({bw2} Hz) should be wider than 500 sym/s BW ({bw1} Hz)"
    )


# ── estimate_snr ─────────────────────────────────────────────────────────────

def test_snr_500(bpsk_500):
    """
    The synthetic signal has no added noise, so SNR should be very high
    when the signal band is set around the carrier.
    """
    iq, fs = bpsk_500
    freqs, psd = compute_psd(iq, fs)
    snr = estimate_snr(freqs, psd, signal_band=(4000, 6000))
    assert snr > 5, f"SNR {snr:.1f} dB is unexpectedly low for a clean signal"


def test_snr_2400(bpsk_2400):
    iq, fs = bpsk_2400
    freqs, psd = compute_psd(iq, fs)
    snr = estimate_snr(freqs, psd, signal_band=(5000, 11000))
    assert snr > 5, f"SNR {snr:.1f} dB is unexpectedly low for a clean signal"


def test_bandwidth_default_threshold(bpsk_500, bpsk_2400):
    """Confirm estimate_bandwidth with default threshold_db (-10.0) yields clean estimates."""
    iq1, fs1 = bpsk_500
    freqs1, psd1 = compute_psd(iq1, fs1)
    bw1 = estimate_bandwidth(freqs1, psd1)
    assert 400 < bw1 < 2000, f"Default BW for 500 sym/s BPSK out of range: {bw1} Hz"

    iq2, fs2 = bpsk_2400
    freqs2, psd2 = compute_psd(iq2, fs2)
    bw2 = estimate_bandwidth(freqs2, psd2)
    assert 2000 < bw2 < 6000, f"Default BW for 2400 sym/s BPSK out of range: {bw2} Hz"
    assert bw2 > bw1


def test_snr_auto_default_threshold(bpsk_500):
    """Confirm estimate_snr auto-detect with default threshold_db (-10.0) yields high SNR on clean signal."""
    iq, fs = bpsk_500
    freqs, psd = compute_psd(iq, fs)
    snr = estimate_snr(freqs, psd)  # uses default threshold_db=-10.0
    assert snr > 15, f"Auto-detected SNR {snr:.1f} dB unexpectedly low"
    assert np.isfinite(snr), "Auto-detected SNR should be finite on clean signal"
