"""
framing.py
----------
Bitstream correlation, frame synchronization, and header/payload extraction.

Week 6:
  - find_sync_word: sliding correlation to locate frame boundaries.
  - extract_frame: header decoding and payload separation with truncation handling.
  - bits_to_ascii: convert bit arrays back to ASCII strings.

Frame Format Convention:
  [Sync Word: M bits] [Header: K bits] [Payload: N * 8 bits]
"""

from __future__ import annotations

import warnings
from typing import Callable
import numpy as np


# ── Standard Sync Words ──────────────────────────────────────────────────────
# 16-bit aerospace / telemetry sync word (0xEB90)
# Peak = 16, max sidelobe = 4
SYNC_WORD_16 = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=int)

# 32-bit CCSDS standard sync word (0x1ACFFC1D)
SYNC_WORD_32 = np.array(
    [(0x1ACFFC1D >> (31 - i)) & 1 for i in range(32)], dtype=int
)


class AsciiDecoded(str):
    """
    Subclass of str returned by bits_to_ascii.

    Behaves in all respects as a standard str, but carries metadata attributes:
      - warning: Optional[str] explaining any dropped bits or non-ASCII characters.
      - has_leftover: bool, True if leftover bits were dropped.
      - leftover_count: int, number of partial bits dropped (< 8).
    """

    warning: str | None = None
    has_leftover: bool = False
    leftover_count: int = 0

    def __new__(
        cls,
        content: str,
        warning: str | None = None,
        has_leftover: bool = False,
        leftover_count: int = 0,
    ):
        instance = super().__new__(cls, content)
        instance.warning = warning
        instance.has_leftover = has_leftover
        instance.leftover_count = leftover_count
        return instance


# ── Header Decoders ──────────────────────────────────────────────────────────

def decode_header_bytes(header_bits: np.ndarray) -> int:
    """
    Decode header bits as an unsigned big-endian integer representing payload
    length in BYTES, and return the corresponding number of payload BITS.

    Parameters
    ----------
    header_bits : np.ndarray
        1D array of header bits.

    Returns
    -------
    payload_bits_count : int
        Number of bits in the payload (byte_count * 8).
    """
    val = 0
    for b in np.asarray(header_bits, dtype=int):
        val = (val << 1) | int(b)
    return val * 8


def decode_header_bits(header_bits: np.ndarray) -> int:
    """
    Decode header bits as an unsigned big-endian integer representing payload
    length in BITS directly.

    Parameters
    ----------
    header_bits : np.ndarray
        1D array of header bits.

    Returns
    -------
    payload_bits_count : int
        Number of bits in the payload.
    """
    val = 0
    for b in np.asarray(header_bits, dtype=int):
        val = (val << 1) | int(b)
    return val


default_header_decoder = decode_header_bytes


# ── 1. Sliding Correlation Sync-Word Finder ──────────────────────────────────

def find_sync_word(
    bits: np.ndarray,
    sync_word: np.ndarray,
    threshold: float = 0.9,
) -> list[int]:
    """
    Slide the sync_word pattern across bits and compute a normalized
    correlation score at each candidate position.

    The score is the fraction of matching bits:
        score = sum(window == sync_word) / len(sync_word)

    This ensures the threshold is meaningful regardless of sync word length
    (e.g., threshold=0.9 requires at least 90% matching bits).

    Parameters
    ----------
    bits : np.ndarray
        1D array of received bits (0s and 1s).
    sync_word : np.ndarray
        1D array of sync word bits (0s and 1s).
    threshold : float, optional
        Normalized correlation score threshold in [0.0, 1.0] (default 0.9).

    Returns
    -------
    positions : list[int]
        Indices in `bits` where the normalized correlation meets or exceeds
        `threshold`. Returns an empty list [] if no match is found.
    """
    bits = np.asarray(bits, dtype=int)
    sync_word = np.asarray(sync_word, dtype=int)

    n_bits = len(bits)
    n_sync = len(sync_word)

    if n_bits == 0 or n_sync == 0 or n_bits < n_sync:
        return []

    # Slide across all valid window positions
    max_start = n_bits - n_sync + 1
    matches: list[int] = []

    for i in range(max_start):
        window = bits[i : i + n_sync]
        score = float(np.mean(window == sync_word))
        if score >= threshold:
            matches.append(i)

    return matches


# ── 2. Frame & Header / Payload Extractor ────────────────────────────────────

def extract_frame(
    bits: np.ndarray,
    sync_position: int,
    sync_word_length: int,
    header_length: int,
    header_decoder: Callable[[np.ndarray], int] = default_header_decoder,
) -> dict:
    """
    Given a confirmed sync word position, extract header bits immediately
    following the sync word, decode the header to determine the expected payload
    length in bits, then extract the payload bits.

    Handles truncated streams gracefully: if fewer bits remain than claimed by
    the header, returns whatever payload bits are available and sets
    truncated=True.

    Parameters
    ----------
    bits : np.ndarray
        1D array of bits containing the frame.
    sync_position : int
        Start index of the sync word in `bits`.
    sync_word_length : int
        Length of the sync word in bits.
    header_length : int
        Length of the header in bits.
    header_decoder : callable
        Function converting `header_bits` to an integer payload length in BITS.
        Defaults to `decode_header_bytes` (interprets header as byte count * 8).

    Returns
    -------
    result : dict
        {
            "sync_position": int,
            "header_value": int,           # decoded payload bit count
            "payload_bits": np.ndarray,    # extracted payload bits
            "truncated": bool,             # True if stream ended prematurely
            "expected_payload_bits": int,  # requested payload bit count
            "actual_payload_bits": int,    # count of bits actually extracted
        }
    """
    bits = np.asarray(bits, dtype=int)
    total_bits = len(bits)

    header_start = sync_position + sync_word_length
    header_end = header_start + header_length

    # Case 1: Stream truncated before or within header
    if total_bits < header_end:
        available_header = bits[header_start:total_bits] if header_start < total_bits else np.array([], dtype=int)
        return {
            "sync_position": sync_position,
            "header_value": 0,
            "payload_bits": np.array([], dtype=int),
            "truncated": True,
            "expected_payload_bits": 0,
            "actual_payload_bits": 0,
        }

    header_bits = bits[header_start:header_end]
    expected_payload_bits = int(header_decoder(header_bits))

    payload_start = header_end
    payload_end = payload_start + expected_payload_bits

    # Case 2: Stream truncated within payload
    if total_bits < payload_end:
        payload_bits = bits[payload_start:total_bits] if payload_start < total_bits else np.array([], dtype=int)
        truncated = True
    else:
        payload_bits = bits[payload_start:payload_end]
        truncated = False

    return {
        "sync_position": sync_position,
        "header_value": expected_payload_bits,
        "payload_bits": payload_bits,
        "truncated": truncated,
        "expected_payload_bits": expected_payload_bits,
        "actual_payload_bits": len(payload_bits),
    }


# ── 3. Bits to ASCII Converter ───────────────────────────────────────────────

def bits_to_ascii(
    bits: np.ndarray,
    return_warning: bool = False,
) -> str | tuple[str, str | None]:
    """
    Convert a 1D bit array (8 bits per byte, MSB first) into an ASCII string.

    If len(bits) is not an exact multiple of 8, leftover partial bits are
    dropped, and a warning is noted in the returned AsciiDecoded string
    (or as the second element of a tuple if return_warning=True).

    Parameters
    ----------
    bits : np.ndarray
        1D array of bits (0s and 1s).
    return_warning : bool, optional
        If True, returns a (text, warning_str_or_None) tuple.
        If False, returns an `AsciiDecoded` string (which subclasses `str` and
        carries `.warning`, `.has_leftover`, `.leftover_count` attributes).

    Returns
    -------
    text : str or tuple[str, str | None]
        The recovered ASCII string.
    """
    bits = np.asarray(bits, dtype=int)
    n_bits = len(bits)

    if n_bits == 0:
        empty_res = AsciiDecoded("", warning=None, has_leftover=False, leftover_count=0)
        return (empty_res, None) if return_warning else empty_res

    n_bytes = n_bits // 8
    leftover = n_bits % 8

    warning_msg: str | None = None
    has_leftover = leftover > 0

    if has_leftover:
        warning_msg = (
            f"Bit count {n_bits} is not a multiple of 8; "
            f"dropped {leftover} leftover bit(s)."
        )
        if not return_warning:
            warnings.warn(warning_msg, UserWarning, stacklevel=2)

    chars = []
    for i in range(n_bytes):
        byte_bits = bits[i * 8 : (i + 1) * 8]
        val = 0
        for b in byte_bits:
            val = (val << 1) | int(b)
        # Use ascii-safe conversion (replace non-ASCII/unprintable bytes if any)
        try:
            char = bytes([val]).decode("ascii", errors="replace")
        except Exception:
            char = "?"
        chars.append(char)

    text_content = "".join(chars)
    res = AsciiDecoded(
        text_content,
        warning=warning_msg,
        has_leftover=has_leftover,
        leftover_count=leftover,
    )

    if return_warning:
        return res, warning_msg
    return res
