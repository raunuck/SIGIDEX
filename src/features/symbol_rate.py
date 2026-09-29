"""
symbol_rate.py
--------------
Symbol-rate estimation via the cyclostationary (squared-magnitude FFT) method.

The key insight: for most linear digital modulations, squaring |x(t)|^2
introduces periodicity at the symbol rate.  The FFT of the squared envelope
therefore shows a spectral peak at f = symbol_rate.

For real-valued (passband) signals the carrier must be removed first,
otherwise |x|^2 = cos^2(wc*t) dominates and the peak lands at 2*fc instead
of the symbol rate.  This module auto-detects the carrier and downconverts
to complex baseband before applying the estimator.
"""

import numpy as np
from scipy.signal import welch


def _downconvert_to_baseband(iq: np.ndarray, fs: int) -> np.ndarray:
    """
    If the signal is real-valued (or has negligible imaginary part),
    detect the carrier frequency from the PSD and shift the signal to
    complex baseband centered on 0 Hz, then lowpass-filter to remove
    the image at -2*fc.

    If the signal is already complex baseband (significant energy in
    both positive and negative frequencies around DC), return it as-is.
    """
    # Check if signal is passband (carrier away from DC) or real-valued
    freqs, psd = welch(iq.real, fs=fs, nperseg=min(1024, len(iq)),
                       return_onesided=True)
    carrier_freq = float(freqs[np.argmax(psd)])

    if carrier_freq > 200.0 or np.max(np.abs(iq.imag)) < 1e-10 * np.max(np.abs(iq.real)):
        # Passband signal — mix down to baseband using carrier from real part
        t = np.arange(len(iq)) / fs
        iq_baseband = iq.real * np.exp(-1j * 2 * np.pi * carrier_freq * t)

        # Lowpass filter to remove the -2*fc image.
        # Cutoff at carrier_freq / 2 — well below 2*fc but well above
        # any symbol-rate bandwidth we'd expect.
        cutoff = carrier_freq / 2
        # Simple moving-average LPF: window length sized to give
        # roughly the right cutoff.  For a boxcar of length L,
        # first null is at fs/L, so L ≈ fs/cutoff.
        L = max(int(fs / cutoff), 3)
        # Make L odd for symmetry
        if L % 2 == 0:
            L += 1
        kernel = np.ones(L) / L
        # Apply filter
        iq_baseband = np.convolve(iq_baseband, kernel, mode="same")

        return iq_baseband

    return iq


def estimate_symbol_rate(
    iq: np.ndarray,
    fs: int,
    min_freq: float = 10.0,
) -> float:
    """
    Estimate the symbol rate of a digitally-modulated IQ signal.

    Method: downconvert to complex baseband if needed, compute |iq|^2,
    take its FFT, find the strongest peak above `min_freq` Hz.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued IQ samples (passband or baseband).
    fs : int
        Sample rate in Hz.
    min_freq : float
        Ignore spectral peaks below this frequency (Hz) to skip the DC /
        near-DC energy that is always present in |x|^2.

    Returns
    -------
    float
        Estimated symbol rate in Hz (the frequency of the strongest
        cyclostationary peak).
    """
    # Downconvert real passband signals so the carrier doesn't dominate
    iq_bb = _downconvert_to_baseband(iq, fs)

    # Squared magnitude — exposes symbol-rate periodicity
    envelope_sq = np.abs(iq_bb) ** 2

    # Remove DC from the squared envelope to sharpen the peak
    envelope_sq = envelope_sq - np.mean(envelope_sq)

    # FFT of the squared envelope
    N = len(envelope_sq)
    fft_vals = np.fft.rfft(envelope_sq)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Mask out near-DC bins
    valid = freqs >= min_freq
    if not np.any(valid):
        return 0.0

    fft_mag_valid = fft_mag[valid]
    freqs_valid = freqs[valid]

    peak_idx = np.argmax(fft_mag_valid)
    symbol_rate = float(freqs_valid[peak_idx])

    return symbol_rate

# add to src/features/symbol_rate.py

def estimate_symbol_rate_fsk(iq, fs):
    """
    FSK-specific symbol rate estimator via tone-transition detection.
    Instantaneous frequency is smoothed first to suppress sample-level
    noise, since raw per-sample frequency estimates are too jittery to
    threshold directly — without smoothing, noise creates many spurious
    edges and the timing estimate locks onto noise spacing instead of
    the true symbol period.
    """
    phase = np.unwrap(np.angle(iq))
    inst_freq = np.diff(phase) * fs / (2 * np.pi)

    # Smooth with a moving average roughly 1/8th of a plausible symbol
    # period, to suppress sample noise without blurring real transitions.
    # A small fixed window works reasonably across typical symbol rates.
    window = max(3, int(fs / 8000))  # tune if needed; ~5 samples at 44.1kHz
    kernel = np.ones(window) / window
    smoothed = np.convolve(inst_freq, kernel, mode="same")

    threshold = np.median(smoothed)
    binary = (smoothed > threshold).astype(int)

    # Find actual transition *positions* (sample indices), not just
    # a raw edge train to autocorrelate blindly
    transitions = np.where(np.diff(binary) != 0)[0]

    if len(transitions) < 2:
        return 0.0

    # Median gap between consecutive real transitions = one symbol period.
    # Median (not mean) so a handful of leftover noisy edges don't skew it.
    gaps = np.diff(transitions)
    symbol_period_samples = np.median(gaps)

    if symbol_period_samples <= 0:
        return 0.0

    return float(fs / symbol_period_samples)