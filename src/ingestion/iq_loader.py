"""
iq_loader.py
------------
Load raw binary .IQ files into a complex numpy array.

Raw IQ files have no header — the user must supply (or guess) the sample rate,
data type, and I/Q channel ordering.
"""

import numpy as np
from scipy.signal import welch


# Supported raw dtypes and their numpy equivalents
_DTYPE_MAP = {
    "int8":    np.int8,
    "int16":   np.int16,
    "float32": np.float32,
}


def load_iq(
    filepath: str,
    sample_rate: int,
    dtype: str = "float32",
    iq_order: str = "iq",
) -> tuple[np.ndarray, int]:
    """
    Load a raw binary IQ file and return (iq_array, sample_rate).

    Parameters
    ----------
    filepath : str
        Path to the raw .IQ file.
    sample_rate : int
        Sample rate in Hz (must be provided; raw files carry no metadata).
    dtype : str
        Sample data type: "int8", "int16", or "float32".
    iq_order : str
        Channel interleaving order: "iq" (default) or "qi".

    Returns
    -------
    iq_array : np.ndarray (complex128)
        Complex-valued sample array.
    sample_rate : int
        Echo of the supplied sample rate.
    """
    if dtype not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype}'. Choose from: {list(_DTYPE_MAP)}"
        )
    if iq_order not in ("iq", "qi"):
        raise ValueError(f"iq_order must be 'iq' or 'qi', got '{iq_order}'")

    np_dtype = _DTYPE_MAP[dtype]
    raw = np.fromfile(filepath, dtype=np_dtype)

    # Samples are interleaved: I0, Q0, I1, Q1, ...
    # Trim to even length
    if len(raw) % 2 != 0:
        raw = raw[:-1]

    if iq_order == "iq":
        i_samples = raw[0::2].astype(np.float64)
        q_samples = raw[1::2].astype(np.float64)
    else:  # "qi"
        q_samples = raw[0::2].astype(np.float64)
        i_samples = raw[1::2].astype(np.float64)

    iq_array = i_samples + 1j * q_samples
    return iq_array, sample_rate


def guess_iq_format(
    filepath: str,
    sample_rate: int,
) -> dict:
    """
    Best-effort heuristic to guess the dtype of a raw IQ file.

    Strategy: try each supported dtype, compute the Welch PSD, and pick the
    dtype whose PSD has the highest peak-to-median ratio (i.e. the "peakiest"
    spectrum, indicating a coherent signal rather than noise-like garbage from
    a wrong dtype interpretation).

    Parameters
    ----------
    filepath : str
        Path to the raw .IQ file.
    sample_rate : int
        Assumed sample rate.

    Returns
    -------
    dict
        {"dtype": best_dtype, "peak_to_median": score}
    """
    best_dtype = None
    best_score = -np.inf
    all_too_short = True

    for dtype_name in _DTYPE_MAP:
        try:
            iq, _ = load_iq(filepath, sample_rate, dtype=dtype_name, iq_order="iq")
            if len(iq) < 256:
                continue
            all_too_short = False
            freqs, psd = welch(iq, fs=sample_rate, nperseg=min(1024, len(iq)))
            psd_db = 10 * np.log10(psd + 1e-30)
            peak = np.max(psd_db)
            median = np.median(psd_db)
            score = peak - median  # higher = peakier
            if score > best_score:
                best_score = score
                best_dtype = dtype_name
        except Exception:
            continue

    if all_too_short:
        return {
            "dtype": None,
            "reason": "file too short for reliable PSD (<256 complex samples for every dtype)",
        }

    return {"dtype": best_dtype, "peak_to_median": float(best_score)}
