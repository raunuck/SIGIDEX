"""
src/correlation
---------------
Bitstream correlation, frame synchronization, and header/payload extraction.
Week 6 of signal-analyzer project.
"""

from .framing import (
    SYNC_WORD_16,
    SYNC_WORD_32,
    AsciiDecoded,
    find_sync_word,
    extract_frame,
    bits_to_ascii,
    decode_header_bytes,
    decode_header_bits,
    default_header_decoder,
)

__all__ = [
    "SYNC_WORD_16",
    "SYNC_WORD_32",
    "AsciiDecoded",
    "find_sync_word",
    "extract_frame",
    "bits_to_ascii",
    "decode_header_bytes",
    "decode_header_bits",
    "default_header_decoder",
]
