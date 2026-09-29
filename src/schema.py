"""
schema.py
---------
Assemble the full Bucket 1 diagnostic report by combining Week 2's RF feature
extractors with Week 3's modulation classifier.

The "coding" section is hardcoded to "unknown" / 0.0 — that's Week 5's job.
"""

import numpy as np

try:
    from features.spectral import compute_psd, estimate_bandwidth, estimate_snr
    from features.symbol_rate import estimate_symbol_rate, estimate_symbol_rate_fsk
    from features.modulation import classify_modulation
    from preprocessing.normalize import remove_dc_offset
except ModuleNotFoundError:
    from src.features.spectral import compute_psd, estimate_bandwidth, estimate_snr
    from src.features.symbol_rate import estimate_symbol_rate, estimate_symbol_rate_fsk
    from src.features.modulation import classify_modulation
    from src.preprocessing.normalize import remove_dc_offset


def build_bucket1_report(iq: np.ndarray, fs: int) -> dict:
    """
    Build a complete Bucket 1 diagnostic report for a loaded IQ signal.

    Applies DC-offset removal internally, then runs all Week 2 + Week 3
    extractors and returns a single structured dict.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued IQ samples (raw, as returned by a loader).
    fs : int
        Sample rate in Hz.

    Returns
    -------
    dict matching the Bucket 1 schema:
        {
          "rf": {"sampling_rate_hz", "bandwidth_hz", "snr_db"},
          "modulation": {"type", "symbol_rate_sps", "confidence"},
          "coding": {
            "interleaver": {"type": "unknown", "confidence": 0.0},
            "fec":         {"type": "unknown", "confidence": 0.0}
          }
        }
    """
    # ── Preprocessing ────────────────────────────────────────────────────
    iq_clean = remove_dc_offset(iq)

    # ── RF features (Week 2) ─────────────────────────────────────────────
    freqs, psd = compute_psd(iq_clean, fs)
    bandwidth_hz = estimate_bandwidth(freqs, psd)

    # SNR: auto-detect signal band from the PSD (no manual band required)
    snr_db = estimate_snr(freqs, psd)  # uses per-bin threshold auto-detect

    # ── Modulation classification (Week 3) ───────────────────────────────
    mod_result = classify_modulation(iq_clean, fs)
    
    # Symbol rate
    if mod_result["type"] == "FSK":
        symbol_rate_sps = estimate_symbol_rate_fsk(iq, fs)
    else:
        symbol_rate_sps = estimate_symbol_rate(iq, fs)
    # symbol_rate_sps = estimate_symbol_rate(iq_clean, fs)


    # ── Assemble report ──────────────────────────────────────────────────
    report = {
        "rf": {
            "sampling_rate_hz": fs,
            "bandwidth_hz": round(bandwidth_hz, 2),
            "snr_db": round(snr_db, 2),
        },
        "modulation": {
            "type": mod_result["type"],
            "symbol_rate_sps": round(symbol_rate_sps, 2),
            "confidence": mod_result["confidence"],
        },
        "coding": {
            "interleaver": {"type": "unknown", "confidence": 0.0},
            "fec": {"type": "unknown", "confidence": 0.0},
        },
    }

    return report
