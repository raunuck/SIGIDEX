"""
normalize.py
------------
Basic signal preprocessing: DC offset removal and power normalization.
Pure NumPy, no dependencies beyond numpy.
"""

import numpy as np


def remove_dc_offset(iq: np.ndarray) -> np.ndarray:
    """
    Remove the DC offset (mean) from a complex IQ signal.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued sample array.

    Returns
    -------
    np.ndarray
        Signal with DC component removed.
    """
    return iq - np.mean(iq)


def normalize_power(iq: np.ndarray) -> np.ndarray:
    """
    Normalize the signal to unit average power.

    The signal is scaled so that  mean(|iq|^2) = 1.

    Parameters
    ----------
    iq : np.ndarray
        Complex-valued sample array.

    Returns
    -------
    np.ndarray
        Power-normalized signal.
    """
    avg_power = np.mean(np.abs(iq) ** 2)
    if avg_power < 1e-30:
        return iq  # avoid division by zero for silence
    return iq / np.sqrt(avg_power)
