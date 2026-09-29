"""
metadata.py
------------
A lightweight metadata container for tracking file-level information about
loaded signals.  This is intentionally simple — just a dataclass for now.

Full SigMF support is out of scope for Week 1.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SignalMetadata:
    """
    Container for signal file metadata.

    Attributes
    ----------
    source_path : str
        Original file path the signal was loaded from.
    sample_rate : int
        Sample rate in Hz.
    dtype : str
        Original data type of the file (e.g. "float32", "int16", "wav").
    num_samples : int
        Number of complex samples after loading.
    description : str
        Free-form description / notes.
    extra : dict
        Arbitrary additional metadata.
    """

    source_path: str = ""
    sample_rate: int = 0
    dtype: str = ""
    num_samples: int = 0
    description: str = ""
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a short human-readable summary string."""
        return (
            f"SignalMetadata(\n"
            f"  source      = {self.source_path}\n"
            f"  sample_rate = {self.sample_rate} Hz\n"
            f"  dtype       = {self.dtype}\n"
            f"  num_samples = {self.num_samples}\n"
            f"  description = {self.description}\n"
            f")"
        )
