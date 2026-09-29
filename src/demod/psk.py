"""
psk.py
------
BPSK carrier recovery, symbol timing, and demodulation.

Week 4 Implementation:
  - costas_loop_bpsk: Second-order Costas loop for carrier phase/frequency synchronization.
  - recover_symbol_timing: Symbol period center-point sampling.
  - bits_from_symbols: Hard decision (sign of real component) to binary bits.
  - demodulate_bpsk: High-level end-to-end BPSK demodulation pipeline.
"""

from __future__ import annotations
import numpy as np

try:
    from features.spectral import compute_psd
    from features.symbol_rate import estimate_symbol_rate
except ModuleNotFoundError:
    from src.features.spectral import compute_psd
    from src.features.symbol_rate import estimate_symbol_rate


def costas_loop_bpsk(
    iq: np.ndarray,
    fs: int,
    carrier_freq_estimate: float,
    loop_bandwidth: float = 0.01,
) -> np.ndarray:
    """
    Second-order Costas loop for BPSK carrier phase and frequency synchronization.

    Mathematical Architecture:
    =========================
    A received passband or analytic BPSK signal can be modeled as:
        r[n] = A[n] * m[n] * exp(j * (2 * pi * fc * n / fs + phi[n])) + w[n]
    where:
        m[n] in {-1, +1} is the antipodal BPSK symbol modulation.
        fc is the nominal carrier frequency in Hz.
        phi[n] is the carrier phase offset.
        w[n] is complex additive channel noise.

    1. Downconversion & Phase Rotation (NCO):
       The Numerically Controlled Oscillator maintains an instantaneous phase estimate:
           theta[n]
       The incoming sample r[n] is rotated by -theta[n]:
           y[n] = r[n] * exp(-j * theta[n])
           I[n] = Re(y[n]),  Q[n] = Im(y[n])

       Near convergence (theta[n] ~ 2*pi*fc*n/fs + phi[n] + phi_err):
           y[n] ~ A[n] * m[n] * exp(j * phi_err)
           I[n] ~ A[n] * m[n] * cos(phi_err)
           Q[n] ~ A[n] * m[n] * sin(phi_err)

    2. Phase Error Detector (PED):
       Because m[n] in {-1, +1}, the modulation causes periodic 180-degree phase inversions.
       The decision-directed BPSK phase error detector cancels m[n] via:
           e[n] = sign(I[n]) * Q[n] / |y[n]|
       Since sign(I[n]) = sign(m[n] * cos(phi_err)) = m[n] (for |phi_err| < pi/2):
           e[n] ~ (m[n]) * (A[n] * m[n] * sin(phi_err)) / A[n]
                = m[n]^2 * sin(phi_err)
                = sin(phi_err) ~ phi_err (for small angles).

       Normalizing by the instantaneous envelope |y[n]| makes the loop detector gain
       Kd = 1 rad^-1 independent of input signal amplitude, ensuring stable loop
       dynamics across varying signal levels.

       Inherent 180-Degree Ambiguity:
       Notice that when phi_err is near +/- pi, sign(I[n]) reverses polarity, producing
       a second stable equilibrium at phi_err = pi. This 180-degree phase ambiguity
       is mathematically fundamental to all suppressed-carrier BPSK loops.

    3. Second-Order Loop Filter (Proportional-Integral):
       For a continuous-time 2nd-order PLL with natural frequency omega_n and damping
       ratio zeta = 1 / sqrt(2) ~ 0.7071 (critically damped / maximally flat response):
           omega_n = 4 * loop_bandwidth / (zeta + 1 / (4 * zeta))
           Kp = 2 * zeta * omega_n      (proportional gain: instantaneous phase tracking)
           Ki = omega_n^2              (integral gain: frequency offset tracking)

    4. State and NCO Update:
       At each discrete sample step n:
           freq_offset[n] = freq_offset[n-1] + Ki * e[n]
           phase_adjust   = Kp * e[n]
           theta[n+1]     = theta[n] + omega_carrier + freq_offset[n] + phase_adjust
       where omega_carrier = 2 * pi * carrier_freq_estimate / fs.
       theta is wrapped to [-pi, pi) to prevent numerical overflow.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued input signal samples.
    fs : int
        Sample rate in Hz.
    carrier_freq_estimate : float
        Initial carrier frequency estimate in Hz.
    loop_bandwidth : float, optional
        Normalized loop filter bandwidth (default 0.01).

    Returns
    -------
    phase_corrected_iq : np.ndarray
        Complex array of phase-corrected baseband samples with carrier removed
        and constellation points aligned to the real axis.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    n_samples = len(iq)
    if n_samples == 0:
        return np.array([], dtype=np.complex128)

    # Carrier phase increment per sample (rad/sample)
    carrier_step = 2.0 * np.pi * carrier_freq_estimate / fs

    # 2nd-order loop filter coefficients (damping ratio zeta = 1/sqrt(2) ~ 0.7071)
    zeta = 0.70710678
    denom = zeta + 1.0 / (4.0 * zeta)
    omega_n = (4.0 * loop_bandwidth) / denom
    kp = 2.0 * zeta * omega_n
    ki = omega_n * omega_n

    # Output buffer and loop state variables
    phase_corrected_iq = np.zeros(n_samples, dtype=np.complex128)
    phase = 0.0
    freq_offset = 0.0

    for n in range(n_samples):
        # 1. NCO downconversion / phase rotation
        # y[n] = r[n] * exp(-j * phase)
        rot = np.exp(-1j * phase)
        y = iq[n] * rot
        phase_corrected_iq[n] = y

        i_comp = y.real
        q_comp = y.imag
        mag = np.hypot(i_comp, q_comp)

        # 2. BPSK Phase Error Detector (normalized by envelope for amplitude invariance)
        if mag > 1e-12:
            sign_i = 1.0 if i_comp >= 0.0 else -1.0
            error = (sign_i * q_comp) / mag
        else:
            error = 0.0

        # 3. Proportional + Integral Loop Filter
        freq_offset += ki * error
        phase_adjust = kp * error

        # 4. Advance NCO phase for sample n+1
        phase += carrier_step + freq_offset + phase_adjust

        # Wrap phase to [-pi, pi)
        phase = (phase + np.pi) % (2.0 * np.pi) - np.pi

    return phase_corrected_iq


def recover_symbol_timing(
    phase_corrected_iq: np.ndarray,
    fs: int,
    symbol_rate: float,
) -> np.ndarray:
    """
    Extract symbol samples at the center of each estimated symbol period.

    Given the sample rate and symbol rate, calculates samples per symbol:
        sps = fs / symbol_rate
    and samples the signal at:
        n_k = floor((k + 0.5) * sps) for k = 0, 1, ..., num_symbols - 1.

    Parameters
    ----------
    phase_corrected_iq : np.ndarray
        Complex-valued baseband IQ samples after Costas carrier phase recovery.
    fs : int
        Sample rate in Hz.
    symbol_rate : float
        Symbol rate in symbols per second (baud).

    Returns
    -------
    symbol_samples : np.ndarray
        1D complex array of symbol-spaced samples taken at the center
        of each symbol interval.
    """
    if symbol_rate <= 0:
        raise ValueError(f"symbol_rate must be positive, got {symbol_rate}")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    phase_corrected_iq = np.asarray(phase_corrected_iq, dtype=np.complex128)
    n_samples = len(phase_corrected_iq)
    if n_samples == 0:
        return np.array([], dtype=np.complex128)

    # Samples per symbol
    sps = fs / symbol_rate

    # If sps is very close to an integer (e.g. 88.0000003), snap to integer
    # to avoid rounding creep over long symbol sequences.
    if abs(sps - round(sps)) < 0.25:
        sps = float(round(sps))

    num_symbols = int(n_samples // sps)
    if num_symbols == 0:
        return np.array([], dtype=np.complex128)

    # Optimal center-point sampling indices: (k + 0.5) * sps
    indices = np.array(
        [int(np.floor((k + 0.5) * sps)) for k in range(num_symbols)],
        dtype=int,
    )
    valid_indices = indices[indices < n_samples]

    return phase_corrected_iq[valid_indices]


def bits_from_symbols(symbol_samples: np.ndarray) -> np.ndarray:
    """
    Make hard decisions on symbol samples to produce binary bits (0 or 1).

    In standard antipodal BPSK modulation:
        Bit 0 maps to constellation point -1 (Re < 0)
        Bit 1 maps to constellation point +1 (Re > 0)

    Hard decision rule:
        bit = 1 if Re(symbol) > 0.0 else 0

    Parameters
    ----------
    symbol_samples : np.ndarray
        1D complex array of symbol samples.

    Returns
    -------
    bit_array : np.ndarray
        1D array of recovered bits (dtype int, values 0 or 1).
    """
    symbol_samples = np.asarray(symbol_samples)
    if len(symbol_samples) == 0:
        return np.array([], dtype=int)

    return (np.real(symbol_samples) > 0.0).astype(int)


def demodulate_bpsk(
    iq: np.ndarray,
    fs: int,
    carrier_freq_estimate: float | None = None,
    symbol_rate: float | None = None,
    loop_bandwidth: float = 0.01,
) -> np.ndarray:
    """
    End-to-end BPSK demodulation pipeline recovering binary bits from an IQ signal.

    Integrates:
      1. Carrier frequency and symbol rate estimation (if not provided).
      2. Second-order Costas loop carrier phase and frequency tracking.
      3. Symbol timing recovery via center-point sampling.
      4. Hard decision bit slicing.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued input signal samples (e.g. from load_wav).
    fs : int
        Sample rate in Hz.
    carrier_freq_estimate : float, optional
        Carrier frequency in Hz. If None, estimated automatically from the PSD peak.
    symbol_rate : float, optional
        Symbol rate in symbols per second (baud). If None, estimated automatically
        via src.features.symbol_rate.estimate_symbol_rate.
    loop_bandwidth : float, optional
        Normalized Costas loop bandwidth (default 0.01).

    Returns
    -------
    bit_array : np.ndarray
        Recovered bit sequence (0 and 1). Note that BPSK carrier recovery has
        an inherent 180-degree phase ambiguity; the recovered bits may be either
        the true bits or their bitwise inversion (1 - bits).
    """
    iq = np.asarray(iq, dtype=np.complex128)
    if len(iq) == 0:
        return np.array([], dtype=int)

    # 1. Estimate carrier frequency if not explicitly provided
    if carrier_freq_estimate is None:
        nperseg = min(len(iq), 2048)
        freqs, psd = compute_psd(iq, fs, nperseg=nperseg)
        carrier_freq_estimate = float(freqs[np.argmax(psd)])

    # 2. Estimate symbol rate if not explicitly provided
    if symbol_rate is None:
        symbol_rate = float(estimate_symbol_rate(iq, fs))

    # 3. Carrier recovery via Costas loop
    phase_corrected = costas_loop_bpsk(
        iq,
        fs=fs,
        carrier_freq_estimate=carrier_freq_estimate,
        loop_bandwidth=loop_bandwidth,
    )

    # 4. Symbol timing recovery
    symbols = recover_symbol_timing(
        phase_corrected,
        fs=fs,
        symbol_rate=symbol_rate,
    )

    # 5. Hard bit decisions
    bits = bits_from_symbols(symbols)

    return bits


def demodulate_bpsk_full(
    iq: np.ndarray,
    fs: int,
    carrier_freq_estimate: float | None = None,
    symbol_rate: float | None = None,
    loop_bandwidth: float = 0.01,
) -> dict:
    """
    Thin wrapper over BPSK demodulation components returning intermediate
    DSP products (symbols, estimated carrier, estimated symbol rate, EVM)
    for GUI visualization.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued input signal samples.
    fs : int
        Sample rate in Hz.
    carrier_freq_estimate : float, optional
        Carrier frequency in Hz. If None, estimated from PSD peak.
    symbol_rate : float, optional
        Symbol rate in sym/s. If None, estimated via estimate_symbol_rate.
    loop_bandwidth : float, optional
        Normalized Costas loop bandwidth (default 0.01).

    Returns
    -------
    result : dict
        {
            "bits": np.ndarray,
            "symbols": np.ndarray,
            "carrier_freq": float,
            "symbol_rate": float,
            "evm_rms": float,  # Percentage, e.g. 1.74
            "phase_corrected": np.ndarray,
        }
    """
    iq = np.asarray(iq, dtype=np.complex128)
    if len(iq) == 0:
        return {
            "bits": np.array([], dtype=int),
            "symbols": np.array([], dtype=np.complex128),
            "carrier_freq": 0.0,
            "symbol_rate": 0.0,
            "evm_rms": 0.0,
            "phase_corrected": np.array([], dtype=np.complex128),
        }

    # 1. Estimate carrier frequency if needed
    if carrier_freq_estimate is None:
        nperseg = min(len(iq), 2048)
        freqs, psd = compute_psd(iq, fs, nperseg=nperseg)
        carrier_freq_estimate = float(freqs[np.argmax(psd)])

    # 2. Estimate symbol rate if needed
    if symbol_rate is None:
        symbol_rate = float(estimate_symbol_rate(iq, fs))

    # 3. Carrier recovery via Costas loop
    phase_corrected = costas_loop_bpsk(
        iq,
        fs=fs,
        carrier_freq_estimate=carrier_freq_estimate,
        loop_bandwidth=loop_bandwidth,
    )

    # 4. Symbol timing recovery
    symbols = recover_symbol_timing(
        phase_corrected,
        fs=fs,
        symbol_rate=symbol_rate,
    )

    # 5. Hard bit decisions
    bits = bits_from_symbols(symbols)

    # 6. Compute EVM (Error Vector Magnitude) against BPSK decision points
    if len(symbols) > 0:
        ref_scale = float(np.median(np.abs(np.real(symbols))))
        if ref_scale > 1e-12:
            norm_symbols = symbols / ref_scale
        else:
            norm_symbols = symbols
        decisions = np.where(np.real(norm_symbols) >= 0.0, 1.0, -1.0)
        errors = norm_symbols - decisions
        evm_rms = float(np.sqrt(np.mean(np.abs(errors) ** 2)) * 100.0)
    else:
        evm_rms = 0.0

    return {
        "bits": bits,
        "symbols": symbols,
        "carrier_freq": carrier_freq_estimate,
        "symbol_rate": symbol_rate,
        "evm_rms": evm_rms,
        "phase_corrected": phase_corrected,
    }

