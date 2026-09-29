"""
src/demod
---------
Demodulation package for digital communications signals.
Week 4: BPSK demodulation.
"""

from .psk import (
    costas_loop_bpsk,
    recover_symbol_timing,
    bits_from_symbols,
    demodulate_bpsk,
    demodulate_bpsk_full,
)

__all__ = [
    "costas_loop_bpsk",
    "recover_symbol_timing",
    "bits_from_symbols",
    "demodulate_bpsk",
    "demodulate_bpsk_full",
]
