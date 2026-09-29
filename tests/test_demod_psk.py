"""
test_demod_psk.py
------------------
Tests for Week 4 BPSK demodulation (src/demod/psk.py):
  - Costas loop carrier phase and frequency synchronization
  - Symbol timing extraction
  - Hard bit decisions
  - End-to-end demodulation against synthetic test_bpsk.wav and ground truth bits
"""

import os
import sys
import numpy as np
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.wav_loader import load_wav
from demod.psk import (
    costas_loop_bpsk,
    recover_symbol_timing,
    bits_from_symbols,
    demodulate_bpsk,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")


# ── Helper: BER computation accounting for 180° phase ambiguity ──────────────

def compute_ber(
    recovered_bits: np.ndarray,
    ground_truth_bits: np.ndarray,
) -> tuple[float, int, int, str, bool]:
    """
    Compute Bit Error Rate (BER) allowing for BPSK 180-degree phase ambiguity.

    Returns
    -------
    ber : float
        Bit error rate (0.0 to 1.0).
    num_errors : int
        Number of mismatched bits.
    total_bits : int
        Total compared bits.
    lock_state : str
        Human-readable description of the lock state:
        'in-phase (direct)' or '180° inverted (phase-flipped)'.
    is_inverted : bool
        True if 180-degree phase inversion occurred.
    """
    n = min(len(recovered_bits), len(ground_truth_bits))
    if n == 0:
        return 0.0, 0, 0, "empty", False

    rec = recovered_bits[:n]
    gt = ground_truth_bits[:n]

    err_direct = int(np.sum(rec != gt))
    err_inverted = int(np.sum((1 - rec) != gt))

    if err_direct <= err_inverted:
        ber = err_direct / n
        return ber, err_direct, n, "in-phase (direct)", False
    else:
        ber = err_inverted / n
        return ber, err_inverted, n, "180° inverted (phase-flipped)", True


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bpsk_clean():
    wav_path = os.path.join(DATA_DIR, "test_bpsk.wav")
    bits_path = os.path.join(DATA_DIR, "test_bpsk_bits.npy")
    if not os.path.isfile(wav_path):
        pytest.skip("test_bpsk.wav not found")
    if not os.path.isfile(bits_path):
        pytest.skip("test_bpsk_bits.npy not found")

    iq, fs = load_wav(wav_path)
    bits = np.load(bits_path)
    return iq, fs, bits


@pytest.fixture(scope="module")
def bpsk_noisy():
    wav_path = os.path.join(DATA_DIR, "test_bpsk_noisy.wav")
    bits_path = os.path.join(DATA_DIR, "test_bpsk_bits.npy")
    if not os.path.isfile(wav_path) or not os.path.isfile(bits_path):
        pytest.skip("test_bpsk_noisy.wav or test_bpsk_bits.npy not found")

    iq, fs = load_wav(wav_path)
    bits = np.load(bits_path)
    return iq, fs, bits


# ── End-to-End Demodulation Tests ────────────────────────────────────────────

def test_demodulate_bpsk_auto_clean(bpsk_clean):
    """
    End-to-end demodulation on clean test_bpsk.wav with auto-estimated
    carrier frequency and symbol rate.
    """
    iq, fs, ground_truth = bpsk_clean

    recovered = demodulate_bpsk(iq, fs)

    ber, errors, total, lock_state, inverted = compute_ber(recovered, ground_truth)

    print("\n" + "=" * 60)
    print("  BPSK Demodulation Test (Auto-Estimated Parameters)")
    print("=" * 60)
    print(f"  Recovered bits count : {len(recovered)}")
    print(f"  Ground-truth count   : {len(ground_truth)}")
    print(f"  Lock polarity        : {lock_state}")
    print(f"  Bit errors           : {errors} / {total}")
    print(f"  Bit Error Rate (BER) : {ber * 100:.2f}%")
    print("=" * 60)

    # Recovered bits count should match ground-truth (200 symbols)
    assert len(recovered) == len(ground_truth)
    # Validation target: 0% BER or very close (<= 2% allowing settling)
    assert ber <= 0.02


def test_demodulate_bpsk_explicit_params(bpsk_clean):
    """
    End-to-end demodulation on clean test_bpsk.wav with known ground-truth
    carrier frequency (5000 Hz) and symbol rate (500 sym/s).
    """
    iq, fs, ground_truth = bpsk_clean

    recovered = demodulate_bpsk(
        iq,
        fs,
        carrier_freq_estimate=5000.0,
        symbol_rate=500.0,
    )

    ber, errors, total, lock_state, inverted = compute_ber(recovered, ground_truth)

    print("\n" + "=" * 60)
    print("  BPSK Demodulation Test (Explicit Parameters)")
    print("=" * 60)
    print(f"  Recovered bits count : {len(recovered)}")
    print(f"  Ground-truth count   : {len(ground_truth)}")
    print(f"  Lock polarity        : {lock_state}")
    print(f"  Bit errors           : {errors} / {total}")
    print(f"  Bit Error Rate (BER) : {ber * 100:.2f}%")
    print("=" * 60)

    assert len(recovered) == len(ground_truth)
    assert ber <= 0.02


def test_demodulate_bpsk_noisy(bpsk_noisy):
    """
    End-to-end demodulation on test_bpsk_noisy.wav (10 dB SNR).
    """
    iq, fs, ground_truth = bpsk_noisy

    recovered = demodulate_bpsk(iq, fs)

    ber, errors, total, lock_state, inverted = compute_ber(recovered, ground_truth)

    print("\n" + "=" * 60)
    print("  BPSK Demodulation Test (Noisy AWGN 10 dB SNR)")
    print("=" * 60)
    print(f"  Recovered bits count : {len(recovered)}")
    print(f"  Ground-truth count   : {len(ground_truth)}")
    print(f"  Lock polarity        : {lock_state}")
    print(f"  Bit errors           : {errors} / {total}")
    print(f"  Bit Error Rate (BER) : {ber * 100:.2f}%")
    print("=" * 60)

    assert len(recovered) == len(ground_truth)
    assert ber <= 0.05


# ── Unit Tests for Individual Pipeline Components ─────────────────────────────

def test_costas_loop_bpsk_carrier_tracking():
    """
    Test Costas loop directly on a synthetic BPSK signal with known frequency
    and phase offset.
    """
    fs = 20_000
    fc = 2_000.0
    symbol_rate = 200.0
    sps = int(fs / symbol_rate)
    num_symbols = 50

    rng = np.random.default_rng(123)
    bits = rng.integers(0, 2, size=num_symbols)
    symbols = 2 * bits - 1
    baseband = np.repeat(symbols, sps).astype(np.float64)

    t = np.arange(len(baseband)) / fs
    # Add initial carrier frequency offset (+30 Hz) and phase offset (+0.5 rad)
    carrier_offset = 30.0
    phase_offset = 0.5
    iq = baseband * np.exp(1j * (2 * np.pi * (fc + carrier_offset) * t + phase_offset))

    # Run Costas loop with nominal fc estimate (without knowing +30 Hz offset)
    corrected = costas_loop_bpsk(iq, fs=fs, carrier_freq_estimate=fc, loop_bandwidth=0.02)

    assert len(corrected) == len(iq)
    assert corrected.dtype == np.complex128

    # After initial settling (first 10 symbols), the constellation points
    # should be collapsed onto the real axis: variance of Q should be much
    # smaller than variance of I.
    settled = corrected[10 * sps :]
    var_i = np.var(settled.real)
    var_q = np.var(settled.imag)
    assert var_i > 0.5
    assert var_q < 0.05, f"Quadrature energy ({var_q}) should be near 0 after Costas lock"


def test_recover_symbol_timing():
    """Test center-point symbol timing recovery extracts correct symbol count."""
    fs = 10_000
    symbol_rate = 500.0
    sps = int(fs / symbol_rate)  # 20
    num_symbols = 40

    symbols_in = np.ones(num_symbols, dtype=np.complex128)
    baseband = np.repeat(symbols_in, sps)

    recovered_symbols = recover_symbol_timing(baseband, fs=fs, symbol_rate=symbol_rate)

    assert len(recovered_symbols) == num_symbols
    assert np.allclose(recovered_symbols, 1.0)


def test_recover_symbol_timing_invalid():
    """Test recover_symbol_timing input validation."""
    dummy = np.ones(100, dtype=np.complex128)
    with pytest.raises(ValueError):
        recover_symbol_timing(dummy, fs=10000, symbol_rate=0.0)
    with pytest.raises(ValueError):
        recover_symbol_timing(dummy, fs=0, symbol_rate=100.0)


def test_bits_from_symbols():
    """Test hard bit decision mapping: positive real -> 1, negative real -> 0."""
    symbols = np.array([1.2 + 0.1j, -0.9 - 0.2j, 0.4 + 0.0j, -1.5 + 0.3j])
    bits = bits_from_symbols(symbols)
    np.testing.assert_array_equal(bits, [1, 0, 1, 0])

    # Empty array
    empty_bits = bits_from_symbols(np.array([]))
    assert len(empty_bits) == 0


def test_demodulate_bpsk_empty():
    """Test demodulate_bpsk on empty input."""
    empty = demodulate_bpsk(np.array([]), fs=44100)
    assert len(empty) == 0
