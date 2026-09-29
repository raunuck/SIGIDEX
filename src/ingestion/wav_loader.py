"""
wav_loader.py
-------------
Load .wav files into a common complex IQ array format.

- Mono WAV  → treated as real-valued; returned as complex with zero imaginary part.
- Stereo WAV → channel 0 = I, channel 1 = Q; returned as complex I + jQ.
"""

import numpy as np
import soundfile as sf
from scipy.signal import hilbert






def to_analytic_signal(real_waveform):
    """Converts a real-valued carrier-modulated signal into a true 
    complex analytic signal, so instantaneous amplitude/phase/frequency 
    are actually meaningful."""
    return hilbert(real_waveform)


def load_wav(filepath: str) -> tuple[np.ndarray, int]:
    """
    Load a WAV file and return (iq_array, sample_rate).

    Parameters
    ----------
    filepath : str
        Path to a .wav file.

    Returns
    -------
    iq_array : np.ndarray (complex128)
        Complex-valued sample array. stereo files are returned as channel0 + j*channel1.
    sample_rate : int
        Sample rate in Hz.
    """
    data, sample_rate = sf.read(filepath, dtype="float64", always_2d=True)
    # data shape: (num_samples, num_channels)
    
    

    num_channels = data.shape[1]

    if num_channels == 1:
        # Mono → real signal, imaginary part = 0
        iq_array = to_analytic_signal(data[:, 0])
    elif num_channels >= 2:
        # Stereo (or more) → first two channels as I / Q
        iq_array = data[:, 0] + 1j * data[:, 1]
    else:
        raise ValueError(f"Unexpected number of channels: {num_channels}")

    return iq_array, int(sample_rate)
