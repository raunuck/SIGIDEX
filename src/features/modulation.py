"""
modulation.py
-------------
Classical (non-ML) modulation classifier using instantaneous signal features.

Classification logic:
  - FSK  : high frequency variation (inst. frequency jumps between tones)
  - PSK  : constant envelope + stable frequency (phase-shift keying)
  - QAM  : varying envelope + stable frequency (amplitude–phase modulation)

All thresholds are named constants at the top for easy tuning.
"""

import numpy as np
# scipy.signal.welch is imported locally inside _to_baseband

try:
    from features.spectral import compute_psd, estimate_bandwidth, estimate_snr
except ModuleNotFoundError:
    from src.features.spectral import compute_psd, estimate_bandwidth, estimate_snr


# ── Tunable threshold constants ──────────────────────────────────────────────
# These control the classification boundaries. Adjusted against measured synthetic
# test files (clean, 20 dB, 10 dB, 0 dB for BPSK, QPSK, 4k/6k FSK, 200 Hz narrow FSK,
# complex baseband IQ FSK, and pure noise).

# ── Smoothing Window Constants ───────────────────────────────────────────────
# Moving-average filter applied to instantaneous frequency to attenuate sample-by-sample
# phase jitter under noise while preserving steady-state symbol tones.
# Window size must stay well under half a symbol (e.g., at 1000 sym/s and 44.1 kHz,
# half a symbol = 22 samples; window = 11 samples).
INST_FREQ_SMOOTH_MIN_WINDOW = 5
INST_FREQ_SMOOTH_FS_DIVISOR = 4000

# ── FSK vs. PSK Decision Threshold ──────────────────────────────────────────
# Robust normalized frequency spread (freq_iqr_norm = freq_iqr_hz / bandwidth)
# above which the signal is classified as FSK.
#
# EMPIRICAL MEASUREMENTS (Smoothed freq_iqr_norm across expanded set):
#   - test_bpsk.wav (clean)             : 0.0239
#   - test_bpsk_snr20.wav (20 dB)       : 0.0726
#   - test_bpsk_snr10.wav (10 dB)       : 0.2042
#   - test_qpsk.wav (clean)             : 0.0338
#   - test_qpsk_snr20.wav (20 dB)       : 0.1496
#   - test_qpsk_snr10.wav (10 dB)       : 0.4516
#   -> Maximum PSK smoothed norm at >= 10 dB SNR is 0.4516.
#
#   - test_fsk.wav (clean 4k/6k)        : 0.9672
#   - test_fsk_snr20.wav (20 dB)        : 0.9556
#   - test_fsk_snr10.wav (10 dB)        : 0.9312
#   - test_fsk_narrow.wav (clean 200Hz) : 0.7740
#   - test_fsk_narrow_snr20 (20 dB)     : 0.7707
#   - test_fsk_narrow_snr10 (10 dB)     : 1.1728
#   - test_fsk_baseband.iq (clean IQ)   : 0.9675
#   - test_fsk_baseband_snr20 (20 dB)   : 0.9559
#   - test_fsk_baseband_snr10 (10 dB)   : 0.9334
#   -> Minimum FSK smoothed norm at >= 10 dB SNR is 0.7707.
#
# The measured gap between PSK (<= 0.4516) and FSK (>= 0.7707) spans 0.452 to 0.771.
# 0.60 sits squarely in this gap, providing comfortable margin for both PSK and FSK.
FREQ_IQR_FSK_THRESHOLD = 0.60
FREQ_CV_FSK_THRESHOLD = FREQ_IQR_FSK_THRESHOLD  # Backward compatibility alias

# ── Coherence Gate Thresholds (Two-Tier Check) ────────────────────────────────
# Catches non-signal input (pure noise) and ambiguous low-SNR signals before classification.

# Tier 1: Hard upper bound on smoothed freq_iqr_hz / fs.
# EMPIRICAL MEASUREMENTS:
#   - Pure complex Gaussian noise : freq_iqr_hz / fs = 0.1202 (5,296 Hz at 44.1 kHz)
#   - Pure real Gaussian noise    : freq_iqr_hz / fs = 0.0922 (4,064 Hz at 44.1 kHz)
#   - All coherent communication signals (even at 0 dB SNR): freq_iqr_hz / fs <= 0.0492 (<= 2,173 Hz).
# 0.0800 (8% of sampling rate, or 3,528 Hz at 44.1 kHz) cleanly rejects pure noise while retaining signals.
FREQ_IQR_FS_HARD_MAX = 0.08
FREQ_IQR_HARD_MAX = FREQ_IQR_FS_HARD_MAX  # Backward compatibility alias
FREQ_IQR_COHERENCE_MAX = FREQ_IQR_HARD_MAX

# Tier 2: Ambiguous middle range bounds for freq_iqr_norm (IQR / BW).
# At 0 dB SNR (brutal noise):
#   - test_bpsk_snr0 (0 dB): smoothed norm = 1.0770, carrier band SNR = 7.55 dB
#   - test_qpsk_snr0 (0 dB): smoothed norm = 1.7110, carrier band SNR = 7.58 dB
#   - test_fsk_snr0  (0 dB): smoothed norm = 1.0168, carrier band SNR = 7.26 dB
#   - test_fsk_baseband_snr0 (0 dB): smoothed norm = 1.0513, carrier band SNR = 7.25 dB
#   - test_fsk_narrow_snr0 (0 dB): smoothed norm = 4.2232, carrier band SNR = 9.87 dB
# These 0 dB signals have noise-elevated frequency spread falling in the [0.50, 5.00] band.
FREQ_IQR_AMBIGUOUS_MIN = 0.50
FREQ_IQR_AMBIGUOUS_MAX = 5.00

# Tier 2: Low-SNR cutoff threshold in dB.
# EMPIRICAL MEASUREMENTS:
# Measured via carrier-centered band SNR (estimate_snr with signal_band=(f_low, f_high) as evaluated in Tier 2):
#   - All 0 dB SNR files  : carrier band SNR = 7.25 dB to 9.87 dB
#   - All 10 dB SNR files : carrier band SNR = 14.74 dB to 15.91 dB (and narrow FSK at 29.78 dB)
# The measured gap is 9.87 dB to 14.74 dB (4.87 dB wide).
# 11.5 dB is centered in this gap.
# Signals with ambiguous frequency spread [0.50, 5.00] and SNR < 11.5 dB safely return "unknown".
LOW_SNR_THRESHOLD_DB = 11.5

# ── Amplitude Threshold for PSK vs QAM ───────────────────────────────────────
# Amplitude coefficient-of-variation above which the signal looks like QAM
# rather than PSK.
# EMPIRICAL MEASUREMENTS:
#   - Clean BPSK/QPSK : amp_cv ≈ 0.058 - 0.061
#   - 20 dB BPSK/QPSK : amp_cv ≈ 0.094 - 0.109
#   - 10 dB BPSK/QPSK : amp_cv ≈ 0.203 - 0.224 (due to Ricean noise envelope on constant amplitude)
#   - Clean 16-QAM    : amp_cv ≈ 0.35
# Setting threshold to 0.28 correctly identifies BPSK/QPSK at 10 dB as PSK while cleanly
# separating multi-amplitude QAM (> 0.35).
AMP_CV_QAM_THRESHOLD = 0.28


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_baseband(iq: np.ndarray, fs: int) -> np.ndarray:
    """
    Convert a signal to complex baseband for feature extraction.

    For real-valued passband signals: detect the carrier from the PSD,
    mix down by exp(-j*2*pi*fc*t), and lowpass-filter to remove the
    -2*fc image.  This gives a clean complex baseband where:
      - PSK has constant amplitude and phase-only changes
      - FSK has frequency variation around DC
      - QAM has amplitude variation

    For already-complex baseband signals: return as-is.
    """
    if np.max(np.abs(iq.imag)) < 1e-10 * np.max(np.abs(iq.real)):
        # Real-valued passband — find carrier and downconvert
        from scipy.signal import welch as _welch, firwin, filtfilt
        freqs, psd = _welch(iq.real, fs=fs, nperseg=min(1024, len(iq)),
                            return_onesided=True)
        carrier_freq = float(freqs[np.argmax(psd)])

        # Mix down to baseband
        t = np.arange(len(iq)) / fs
        iq_bb = iq * np.exp(-1j * 2 * np.pi * carrier_freq * t)

        # Proper FIR lowpass to remove the -2*fc image.
        # Cutoff well below 2*fc but above the signal bandwidth.
        cutoff_hz = max(carrier_freq * 0.8, 200.0)
        nyquist = fs / 2.0
        numtaps = min(101, len(iq_bb) // 3)
        if numtaps % 2 == 0:
            numtaps += 1
        numtaps = max(numtaps, 5)
        fir = firwin(numtaps, cutoff_hz / nyquist)
        # Apply zero-phase filter to both real and imag parts
        iq_bb = (filtfilt(fir, 1.0, iq_bb.real)
                 + 1j * filtfilt(fir, 1.0, iq_bb.imag))

        return iq_bb

    return iq


def _excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition), computed without scipy.stats."""
    mu = np.mean(x)
    m2 = np.mean((x - mu) ** 2)
    m4 = np.mean((x - mu) ** 4)
    if m2 < 1e-30:
        return 0.0
    return float(m4 / (m2 ** 2) - 3.0)


# ── Public API ───────────────────────────────────────────────────────────────

def instantaneous_features(iq: np.ndarray, fs: int = 44100) -> dict:
    """
    Compute instantaneous signal features useful for modulation classification.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued IQ samples (passband or baseband).
    fs : int
        Sample rate in Hz (needed to convert to Hz and compute PSD/bandwidth).

    Returns
    -------
    dict with keys:
        amp_var           – variance of the instantaneous amplitude
        amp_cv            – coefficient of variation of amplitude (std / mean)
        freq_var          – variance of the instantaneous frequency (rad/sample)
        freq_cv           – coefficient of variation of inst. frequency
        freq_iqr_norm     – robust frequency spread normalized by occupied bandwidth (IQR_hz / BW)
        freq_iqr_norm_raw – raw unsmoothed frequency spread normalized by bandwidth (for debugging)
        freq_iqr_hz       – smoothed instantaneous frequency IQR in Hz
        freq_iqr_hz_raw   – raw unsmoothed instantaneous frequency IQR in Hz
        bandwidth         – estimated occupied bandwidth in Hz
        kurtosis          – excess kurtosis of the amplitude distribution
        papr_db           – peak-to-average power ratio in dB
    """
    if np.isrealobj(iq) or np.max(np.abs(iq.imag)) < 1e-10 * np.max(np.abs(iq.real)):
        from scipy.signal import hilbert
        iq = hilbert(iq.real)

    amp = np.abs(iq)
    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase)  # rad / sample

    # Amplitude statistics
    amp_mean = np.mean(amp)
    amp_var = float(np.var(amp))
    amp_cv = float(np.std(amp) / (amp_mean + 1e-30))

    # Frequency statistics — raw variance / CV
    freq_var = float(np.var(inst_freq))
    freq_mean_abs = np.mean(np.abs(inst_freq))
    freq_cv = float(np.std(inst_freq) / (freq_mean_abs + 1e-30))

    # Instantaneous frequency in Hz (carrier-offset invariant)
    inst_freq_hz = inst_freq * fs / (2 * np.pi)
    q75_raw, q25_raw = np.percentile(inst_freq_hz, [75, 25])
    freq_iqr_hz_raw = float(q75_raw - q25_raw)

    # Smoothed instantaneous frequency:
    # Moving-average filter (window = max(5, int(fs / 4000)) samples) to attenuate
    # sample-by-sample phase jitter while preserving steady-state symbol frequency tones.
    smooth_window = max(INST_FREQ_SMOOTH_MIN_WINDOW, int(fs / INST_FREQ_SMOOTH_FS_DIVISOR))
    if len(inst_freq_hz) >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        inst_freq_sm_hz = np.convolve(inst_freq_hz, kernel, mode="same")
    else:
        inst_freq_sm_hz = inst_freq_hz

    q75, q25 = np.percentile(inst_freq_sm_hz, [75, 25])
    freq_iqr_hz = float(q75 - q25)

    # Occupied bandwidth for carrier-independent normalization
    freqs, psd = compute_psd(iq, fs)
    bw = estimate_bandwidth(freqs, psd)
    effective_bw = max(float(bw), 1.0)

    # Robust normalized frequency spread: IQR relative to occupied bandwidth
    freq_iqr_norm = float(freq_iqr_hz / effective_bw)
    freq_iqr_norm_raw = float(freq_iqr_hz_raw / effective_bw)

    # Higher-order statistics
    kurtosis = _excess_kurtosis(amp)

    # Peak-to-average power ratio
    peak_power = np.max(amp ** 2)
    avg_power = np.mean(amp ** 2)
    papr_db = float(10.0 * np.log10(peak_power / (avg_power + 1e-30)))

    return {
        "amp_var": amp_var,
        "amp_cv": amp_cv,
        "freq_var": freq_var,
        "freq_cv": freq_cv,
        "freq_iqr_norm": freq_iqr_norm,
        "freq_iqr_norm_raw": freq_iqr_norm_raw,
        "freq_iqr_hz": freq_iqr_hz,
        "freq_iqr_hz_raw": freq_iqr_hz_raw,
        "bandwidth": bw,
        "kurtosis": kurtosis,
        "papr_db": papr_db,
    }


def classify_modulation(iq: np.ndarray, fs: int = 44100) -> dict:
    """
    Classify the modulation type of an IQ signal using threshold logic on
    instantaneous features.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued IQ samples.
    fs : int
        Sample rate in Hz.

    Returns
    -------
    dict with keys:
        type       – "FSK", "PSK", "QAM", or "unknown"
        confidence – float in [0, 1]
        features   – the full instantaneous_features dict
        runner_up  – dict with "type" and "confidence" for the second-highest-scoring
                     category (always present whenever a category decision is made)
        reason     – explanation string when type is "unknown"
    """
    features = instantaneous_features(iq, fs)
    amp_cv = features["amp_cv"]
    # Primary discriminator for FSK vs PSK/QAM:
    # Use robust normalized IQR of SMOOTHED instantaneous frequency relative to bandwidth
    # to avoid false FSK classification from noise jitter or carrier offset dependence.
    freq_spread = features["freq_iqr_norm"]
    freq_iqr_hz = features["freq_iqr_hz"]

    # ── Coherence Gate (Two-Tier Check) ──────────────────────────────────
    # Tier 1: Hard upper bound (pure noise rejection)
    # Catches pure / near-pure random noise (freq_iqr_hz / fs >= 0.09) regardless of SNR.
    if freq_iqr_hz / fs > FREQ_IQR_FS_HARD_MAX:
        return {
            "type": "unknown",
            "confidence": 0.1,
            "reason": "no coherent modulation structure detected",
            "features": features,
        }

    # ── Compute SNR for Tier 2 Ambiguous-Zone Check ──────────────────────
    freqs, psd = compute_psd(iq, fs)
    peak_idx = int(np.argmax(psd))
    peak_freq = abs(float(freqs[peak_idx]))
    band_half_width = max(2000.0, fs * 0.05)
    f_low = max(0.0, peak_freq - band_half_width)
    f_high = min(fs / 2.0, peak_freq + band_half_width)
    snr_db = estimate_snr(freqs, psd, signal_band=(f_low, f_high))
    if np.isnan(snr_db):
        snr_db = 0.0

    # Tier 2: Ambiguous-zone check (noisy real signals)
    # Catches brutal noise signals (e.g. 0 dB SNR ≈ 7.5 dB in-band) whose noise-induced
    # frequency spread falls into the ambiguous region [FREQ_IQR_AMBIGUOUS_MIN, FREQ_IQR_AMBIGUOUS_MAX]
    # and SNR is below LOW_SNR_THRESHOLD_DB (11.5 dB).
    if (FREQ_IQR_AMBIGUOUS_MIN <= freq_spread <= FREQ_IQR_AMBIGUOUS_MAX
            and snr_db < LOW_SNR_THRESHOLD_DB):
        return {
            "type": "unknown",
            "confidence": 0.2,
            "reason": "ambiguous frequency spread at low SNR",
            "features": features,
        }

    # ── Tier 3: Score each modulation family ─────────────────────────────
    # Scores are in [0, 1]; higher = more likely.
    # We use squared ratios for sharper decision boundaries.

    # Ratio of frequency spread to threshold.  Values > 1 → likely FSK.
    freq_ratio = freq_spread / FREQ_IQR_FSK_THRESHOLD
    amp_ratio = amp_cv / AMP_CV_QAM_THRESHOLD

    # FSK:  frequency spread is the primary indicator.
    #        Score rises steeply when freq_spread exceeds the threshold.
    fsk_score = freq_ratio ** 2 / (1.0 + freq_ratio ** 2)

    # QAM:  amplitude variation high, frequency spread low.
    qam_amp = amp_ratio ** 2 / (1.0 + amp_ratio ** 2)
    qam_freq_penalty = 1.0 / (1.0 + freq_ratio ** 2)
    qam_score = qam_amp * qam_freq_penalty

    # PSK:  both amplitude and frequency variation low (constant envelope,
    #        stable carrier).
    psk_amp = 1.0 / (1.0 + amp_ratio ** 2)
    psk_freq = 1.0 / (1.0 + freq_ratio ** 2)
    psk_score = psk_amp * psk_freq

    # ── Normalize to probabilities ───────────────────────────────────────
    raw_scores = {"FSK": fsk_score, "PSK": psk_score, "QAM": qam_score}
    total = sum(raw_scores.values())
    if total < 1e-30:
        probs = {k: 1.0 / len(raw_scores) for k in raw_scores}
    else:
        probs = {k: v / total for k, v in raw_scores.items()}

    # ── Rank candidates ──────────────────────────────────────────────────
    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    winner_type, winner_conf = ranked[0]
    runner_type, runner_conf = ranked[1]

    result = {
        "type": winner_type,
        "confidence": round(winner_conf, 4),
        "features": features,
        "runner_up": {
            "type": runner_type,
            "confidence": round(runner_conf, 4),
        },
    }

    return result
