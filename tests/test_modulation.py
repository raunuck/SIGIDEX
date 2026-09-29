"""
test_modulation.py
------------------
Tests for src/features/modulation.py against synthetic BPSK, QPSK, and FSK files,
including noise-robustness checks across 20 dB, 10 dB, and 0 dB SNR files.

Expected classifications:
  test_bpsk.wav                 -> "PSK"
  test_bpsk_snr20 / snr10       -> "PSK"
  test_qpsk.wav                 -> "PSK"
  test_qpsk_snr20 / snr10       -> "PSK"
  test_fsk.wav                  -> "FSK"
  test_fsk_snr20 / snr10        -> "FSK"
  test_*_snr0 (0 dB signals)    -> "unknown" (never a confident wrong answer)
  pure noise (complex / real)   -> "unknown"
"""

import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.wav_loader import load_wav
from ingestion.iq_loader import load_iq
from preprocessing.normalize import remove_dc_offset
from features.modulation import instantaneous_features, classify_modulation

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bpsk_signal():
    path = os.path.join(DATA_DIR, "test_bpsk.wav")
    if not os.path.isfile(path):
        pytest.skip("test_bpsk.wav not found")
    iq, fs = load_wav(path)
    return remove_dc_offset(iq), fs


@pytest.fixture(scope="module")
def qpsk_signal():
    path = os.path.join(DATA_DIR, "test_qpsk.wav")
    if not os.path.isfile(path):
        pytest.skip("test_qpsk.wav not found")
    iq, fs = load_wav(path)
    return remove_dc_offset(iq), fs


@pytest.fixture(scope="module")
def fsk_signal():
    path = os.path.join(DATA_DIR, "test_fsk.wav")
    if not os.path.isfile(path):
        pytest.skip("test_fsk.wav not found")
    iq, fs = load_wav(path)
    return remove_dc_offset(iq), fs


# ── instantaneous_features ──────────────────────────────────────────────────

def test_features_returns_required_keys(bpsk_signal):
    iq, fs = bpsk_signal
    feats = instantaneous_features(iq, fs)
    for key in (
        "amp_var",
        "amp_cv",
        "freq_var",
        "freq_cv",
        "freq_iqr_norm",
        "freq_iqr_norm_raw",
        "kurtosis",
        "papr_db",
    ):
        assert key in feats, f"Missing key: {key}"


def test_features_values_are_finite(bpsk_signal):
    iq, fs = bpsk_signal
    feats = instantaneous_features(iq, fs)
    for key, val in feats.items():
        assert np.isfinite(val), f"Feature '{key}' is not finite: {val}"


def test_smoothing_attenuates_noise_iqr():
    """Verify that moving-average smoothing attenuates high-frequency noise spikes in inst_freq."""
    path = os.path.join(DATA_DIR, "test_bpsk_snr10.wav")
    if not os.path.isfile(path):
        pytest.skip("test_bpsk_snr10.wav not found")
    iq, fs = load_wav(path)
    feats = instantaneous_features(iq, fs)
    # On 10 dB BPSK, raw IQR is ~0.47 while smoothed IQR is ~0.068
    assert feats["freq_iqr_norm"] < feats["freq_iqr_norm_raw"], (
        f"Smoothed IQR ({feats['freq_iqr_norm']}) should be smaller than raw ({feats['freq_iqr_norm_raw']})"
    )


# ── classify_modulation ─────────────────────────────────────────────────────

def test_bpsk_classified_as_psk(bpsk_signal):
    iq, fs = bpsk_signal
    result = classify_modulation(iq, fs)
    assert result["type"] == "PSK", (
        f"BPSK should be classified as PSK, got '{result['type']}' "
        f"(conf={result['confidence']:.3f})"
    )


def test_qpsk_classified_as_psk(qpsk_signal):
    iq, fs = qpsk_signal
    result = classify_modulation(iq, fs)
    assert result["type"] == "PSK", (
        f"QPSK should be classified as PSK, got '{result['type']}' "
        f"(conf={result['confidence']:.3f})"
    )


def test_fsk_classified_as_fsk(fsk_signal):
    iq, fs = fsk_signal
    result = classify_modulation(iq, fs)
    assert result["type"] == "FSK", (
        f"FSK should be classified as FSK, got '{result['type']}' "
        f"(conf={result['confidence']:.3f})"
    )


# ── Target Tests: 20 dB & 10 dB Noise Robustness ─────────────────────────────

@pytest.mark.parametrize("snr", [20, 10])
def test_noisy_bpsk_at_20_and_10_db_classified_as_psk(snr):
    path = os.path.join(DATA_DIR, f"test_bpsk_snr{snr}.wav")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    result = classify_modulation(iq, fs)
    assert result["type"] == "PSK", (
        f"BPSK at {snr} dB SNR must be classified as 'PSK', got '{result['type']}' (conf={result['confidence']})"
    )


@pytest.mark.parametrize("snr", [20, 10])
def test_noisy_qpsk_at_20_and_10_db_classified_as_psk(snr):
    path = os.path.join(DATA_DIR, f"test_qpsk_snr{snr}.wav")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    result = classify_modulation(iq, fs)
    assert result["type"] == "PSK", (
        f"QPSK at {snr} dB SNR must be classified as 'PSK', got '{result['type']}' (conf={result['confidence']})"
    )


@pytest.mark.parametrize("snr", [20, 10])
def test_noisy_fsk_at_20_and_10_db_classified_as_fsk(snr):
    path = os.path.join(DATA_DIR, f"test_fsk_snr{snr}.wav")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    result = classify_modulation(iq, fs)
    assert result["type"] == "FSK", (
        f"FSK at {snr} dB SNR must be classified as 'FSK', got '{result['type']}' (conf={result['confidence']})"
    )


@pytest.mark.parametrize("prefix,expected_mod", [
    ("test_bpsk", "PSK"),
    ("test_qpsk", "PSK"),
    ("test_fsk", "FSK"),
])
def test_zero_db_snr_not_confident_wrong_answer(prefix, expected_mod):
    """0 dB SNR signals may stay 'unknown'; nothing at any SNR may be a confident wrong answer."""
    path = os.path.join(DATA_DIR, f"{prefix}_snr0.wav")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    result = classify_modulation(iq, fs)
    # Permitted: "unknown" (preferred for brutal 0 dB noise) or expected_mod
    assert result["type"] in ("unknown", expected_mod), (
        f"{prefix} at 0 dB SNR was confidently misclassified as '{result['type']}' (expected 'unknown' or '{expected_mod}')"
    )
    if result["type"] == expected_mod:
        assert result["confidence"] <= 0.85, f"Confidence should not be inflated at 0 dB: {result['confidence']}"


def test_confidence_is_plausible(bpsk_signal, qpsk_signal, fsk_signal):
    """Confidence should be a float in (0, 1] for all test signals."""
    for label, (iq, fs) in [("BPSK", bpsk_signal), ("QPSK", qpsk_signal), ("FSK", fsk_signal)]:
        result = classify_modulation(iq, fs)
        conf = result["confidence"]
        assert 0.0 < conf <= 1.0, f"{label} confidence {conf} is outside (0, 1]"


def test_result_has_features_dict(bpsk_signal):
    iq, fs = bpsk_signal
    result = classify_modulation(iq, fs)
    assert "features" in result
    assert isinstance(result["features"], dict)


def test_pure_noise_returns_unknown():
    """Pure random noise lacks coherent modulation and should return 'unknown' with low confidence."""
    rng = np.random.default_rng(42)
    fs = 44100
    # Test with complex Gaussian noise (typical unmodulated IQ)
    noise_complex = rng.normal(0, 1, fs) + 1j * rng.normal(0, 1, fs)
    result = classify_modulation(noise_complex, fs)

    assert result["type"] == "unknown", (
        f"Pure complex noise should be classified as 'unknown', got '{result['type']}'"
    )
    assert 0.0 < result["confidence"] <= 0.3, (
        f"Confidence should be low (<= 0.3), got {result['confidence']}"
    )
    assert "reason" in result
    assert "features" in result

    # Also test with real Gaussian noise (unmodulated mono WAV)
    noise_real = rng.normal(0, 1, fs)
    result_real = classify_modulation(noise_real, fs)
    assert result_real["type"] == "unknown", (
        f"Pure real noise should be classified as 'unknown', got '{result_real['type']}'"
    )
    assert 0.0 < result_real["confidence"] <= 0.3


def test_runner_up_always_present_for_category_decisions(bpsk_signal, qpsk_signal, fsk_signal):
    """Whenever a category decision (FSK/PSK/QAM) is made, runner_up must be present."""
    for label, (iq, fs) in [("BPSK", bpsk_signal), ("QPSK", qpsk_signal), ("FSK", fsk_signal)]:
        result = classify_modulation(iq, fs)
        assert result["type"] in ("FSK", "PSK", "QAM")
        assert "runner_up" in result, f"runner_up missing for {label}"
        ru = result["runner_up"]
        assert "type" in ru and "confidence" in ru
        assert ru["type"] in ("FSK", "PSK", "QAM")
        assert ru["type"] != result["type"]
        assert ru["confidence"] <= result["confidence"]
        assert 0.0 <= ru["confidence"] <= 1.0


def test_runner_up_on_ambiguous_blend(bpsk_signal, fsk_signal):
    """A 50/50 blend of FSK and BPSK should always include runner_up with competitive confidence."""
    iq_bpsk, fs = bpsk_signal
    iq_fsk, _ = fsk_signal
    L = min(len(iq_bpsk), len(iq_fsk))
    blend = 0.5 * iq_bpsk[:L] + 0.5 * iq_fsk[:L]
    result = classify_modulation(blend, fs)

    assert result["type"] in ("FSK", "PSK", "QAM")
    assert "runner_up" in result, "runner_up must be present for ambiguous blend signal"
    ru = result["runner_up"]
    assert ru["type"] != result["type"]
    assert ru["confidence"] > 0.10, f"Expected non-trivial runner-up confidence, got {ru['confidence']}"


def test_runner_up_omitted_for_unknown():
    """When the coherence gate rejects signal as unknown, runner_up should not be present."""
    rng = np.random.default_rng(42)
    fs = 44100
    noise = rng.normal(0, 1, fs) + 1j * rng.normal(0, 1, fs)
    result = classify_modulation(noise, fs)
    assert result["type"] == "unknown"
    assert "runner_up" not in result


# ── Expanded FSK Set Tests (Narrow Shift & Complex Baseband IQ) ─────────────

@pytest.mark.parametrize("suffix", ["", "_snr20", "_snr10"])
def test_narrow_shift_fsk_classified_as_fsk(suffix):
    """Narrow-shift FSK (200 Hz shift on 5 kHz carrier) must classify as FSK at clean, 20 dB, 10 dB."""
    path = os.path.join(DATA_DIR, f"test_fsk_narrow{suffix}.wav")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    iq, fs = load_wav(path)
    iq = remove_dc_offset(iq)
    result = classify_modulation(iq, fs)
    assert result["type"] == "FSK", (
        f"Narrow-shift FSK ({suffix or 'clean'}) misclassified as {result['type']} (conf={result['confidence']})"
    )


@pytest.mark.parametrize("suffix", ["", "_snr20", "_snr10"])
def test_baseband_iq_fsk_classified_as_fsk(suffix):
    """Complex baseband IQ FSK (carrier at 0 Hz) must classify as FSK at clean, 20 dB, 10 dB."""
    path = os.path.join(DATA_DIR, f"test_fsk_baseband{suffix}.iq")
    if not os.path.isfile(path):
        pytest.skip(f"{path} not found")
    iq, fs = load_iq(path, 44100, dtype="float32")
    result = classify_modulation(iq, fs)
    assert result["type"] == "FSK", (
        f"Complex baseband IQ FSK ({suffix or 'clean'}) misclassified as {result['type']} (conf={result['confidence']})"
    )


def test_expanded_zero_db_fsk_not_confident_wrong_answer():
    """0 dB FSK signals on narrow-shift and baseband IQ must not be a confident wrong answer."""
    # Narrow shift at 0 dB
    path_narrow = os.path.join(DATA_DIR, "test_fsk_narrow_snr0.wav")
    if os.path.isfile(path_narrow):
        iq, fs = load_wav(path_narrow)
        res = classify_modulation(iq, fs)
        assert res["type"] in ("unknown", "FSK")

    # Baseband IQ at 0 dB
    path_bb = os.path.join(DATA_DIR, "test_fsk_baseband_snr0.iq")
    if os.path.isfile(path_bb):
        iq, fs = load_iq(path_bb, 44100, dtype="float32")
        res = classify_modulation(iq, fs)
        assert res["type"] in ("unknown", "FSK")

