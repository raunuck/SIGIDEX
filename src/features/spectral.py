"""
spectral.py
-----------
Spectral analysis utilities for IQ signals:
  - Power spectral density (centered on 0 Hz)
  - Occupied bandwidth estimation
  - In-band / out-of-band SNR estimation
"""

import numpy as np
from scipy.signal import welch


def compute_psd(
    iq: np.ndarray,
    fs: int,
    nperseg: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the power spectral density of a complex IQ signal, centered on 0 Hz.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued IQ samples.
    fs : int
        Sample rate in Hz.
    nperseg : int
        FFT length per Welch segment.

    Returns
    -------
    freqs : np.ndarray
        Frequency axis in Hz, centered on 0 (range [-fs/2, +fs/2)).
    psd : np.ndarray
        Power spectral density (linear scale), same length as freqs.
    """
    freqs, psd = welch(iq, fs=fs, nperseg=nperseg, return_onesided=False)

    # welch with return_onesided=False gives frequencies [0, fs).
    # Shift to [-fs/2, +fs/2) so DC is in the center.
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    # Wrap frequencies > fs/2 to negative (welch returns [0..fs), fftshift
    # moves the second half to the front, but the values are still 0..fs
    # rather than -fs/2..+fs/2).
    freqs = freqs - fs * (freqs >= fs / 2)

    return freqs, psd


# Default threshold changed from -20.0 to -10.0 dB based on testing against
# synthetic noisy BPSK (e.g. 0 dB SNR): -20.0 dB captures noise-floor energy
# on realistically noisy signals, causing near-full-spectrum bandwidth reports
# and downstream false "inf" SNR results.
def estimate_bandwidth(
    freqs: np.ndarray,
    psd: np.ndarray,
    threshold_db: float = -10.0,
) -> float:
    """
    Estimate occupied bandwidth from a PSD.

    Converts the PSD to dB relative to its peak, then finds the contiguous
    frequency span where power stays above `threshold_db`.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis (Hz), as returned by compute_psd (centered).
    psd : np.ndarray
        Power spectral density (linear scale).
    threshold_db : float
        dB threshold below the peak.  Frequencies where the PSD is within
        this many dB of the peak are considered "occupied".
        Default is -10.0 dB (chosen over -20.0 dB to prevent near-full-spectrum
        bandwidth reports and downstream false "inf" SNR results on noisy signals).

    Returns
    -------
    float
        Estimated bandwidth in Hz.
    """
    psd_db = 10.0 * np.log10(psd + 1e-30)
    peak_db = np.max(psd_db)
    mask = psd_db >= (peak_db + threshold_db)  # threshold_db is negative

    if not np.any(mask):
        return 0.0

    occupied_freqs = freqs[mask]
    bandwidth = float(np.max(occupied_freqs) - np.min(occupied_freqs))
    return bandwidth

# estimate_snr saturates above ~18-20dB true SNR for unshaped test signals due to spectral leakage; treat values in that range as 'high/clean' rather than precise.
# Default threshold_db is -10.0 dB (changed from -20.0 dB) to avoid auto-detecting a signal band that spans the full spectrum on noisy signals (which causes false "inf" SNR).
def estimate_snr(
    freqs: np.ndarray,
    psd: np.ndarray,
    signal_band: tuple[float, float] | None = None,
    threshold_db: float = -10.0,
    mirror: bool = True,
) -> float:
    """
    Estimate SNR by comparing mean power inside a signal band to mean power
    outside it.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency axis (Hz), centered on 0.
    psd : np.ndarray
        Power spectral density (linear scale).
    signal_band : tuple[float, float] or None
        (f_low, f_high) defining the signal band in Hz.  If None, the band
        is auto-detected using the same dB-threshold approach as
        estimate_bandwidth (all frequencies within `threshold_db` of the
        PSD peak are treated as "signal").
    threshold_db : float
        Only used when signal_band is None.  dB below the PSD peak to
        define the signal region (default -10.0).  Chosen over -20.0 dB based
        on testing against synthetic noisy BPSK: -20.0 dB encompasses the
        entire noise floor on noisy signals (e.g. 0 dB SNR), leaving no
        out-of-band samples and falsely returning inf.
    mirror : bool
        When True and signal_band is provided, also include the mirror band
        (-f_high, -f_low) so that real-valued signals don't have their
        symmetric spectral image counted as noise.  Has no effect on the
        auto-detect path (which is already mirror-safe by construction).
        Default True.

    Returns
    -------
    float
        SNR in dB.  Returns np.inf if noise power is effectively zero.
    """
    if signal_band is None:
        # Auto-detect: per-bin threshold naturally captures both mirror
        # peaks (they're the same height), so no mirror issue here.
        psd_db = 10.0 * np.log10(psd + 1e-30)
        peak_db = np.max(psd_db)
        in_band = psd_db >= (peak_db + threshold_db)
    else:
        f_low, f_high = signal_band
        in_band = (freqs >= f_low) & (freqs <= f_high)
        if mirror:
            # Include the symmetric image around 0 Hz
            mirror_band = (freqs >= -f_high) & (freqs <= -f_low)
            in_band = in_band | mirror_band

    out_band = ~in_band

    if not np.any(in_band) or not np.any(out_band):
        return float("inf")

    signal_power = np.mean(psd[in_band])
    noise_power = np.mean(psd[out_band])

    if noise_power < 1e-30:
        return float("inf")

    snr_db = 10.0 * np.log10(signal_power / noise_power)
    return float(snr_db)


