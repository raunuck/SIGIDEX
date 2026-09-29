"""
fec.py
------
Forward Error Correction decoding utilities.

Week 5: Viterbi decoding for convolutional codes, and a decode quality
scoring function for the search-and-validate loop.

Uses commpy.channelcoding.convcode for the Trellis/Viterbi implementation.
"""

from __future__ import annotations

import numpy as np
import commpy.channelcoding.convcode as cc


# ── Pre-built Trellis cache ─────────────────────────────────────────────────
# Avoid re-constructing the same Trellis on every call. Keyed by (rate, K).
_TRELLIS_CACHE: dict[tuple[str, int], cc.Trellis] = {}

# Generator polynomial lookup: (rate_str, constraint_length) -> (memory, g_matrix)
# Only the common cases needed for the candidate search right now.
_GENERATOR_POLYS: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {
    ("1/2", 7): (np.array([6]), np.array([[0o171, 0o133]])),    # NASA/CCSDS standard
    ("1/2", 5): (np.array([4]), np.array([[0o23,  0o35]])),     # common short code
    ("1/2", 3): (np.array([2]), np.array([[0o7,   0o5]])),      # textbook example
}


def _get_trellis(rate: str = "1/2", constraint_length: int = 7) -> cc.Trellis:
    """
    Get (or create and cache) a commpy Trellis for the given code parameters.

    Parameters
    ----------
    rate : str
        Code rate as a fraction string, e.g. "1/2".
    constraint_length : int
        Constraint length K of the convolutional code.

    Returns
    -------
    trellis : cc.Trellis

    Raises
    ------
    ValueError
        If the (rate, constraint_length) pair is not in the generator lookup.
    """
    key = (rate, constraint_length)
    if key not in _TRELLIS_CACHE:
        if key not in _GENERATOR_POLYS:
            raise ValueError(
                f"No generator polynomials defined for rate={rate}, K={constraint_length}. "
                f"Known configs: {list(_GENERATOR_POLYS.keys())}"
            )
        memory, g_matrix = _GENERATOR_POLYS[key]
        _TRELLIS_CACHE[key] = cc.Trellis(memory, g_matrix)
    return _TRELLIS_CACHE[key]


def viterbi_decode(
    bits: np.ndarray,
    constraint_length: int = 7,
    rate: str = "1/2",
) -> np.ndarray:
    """
    Viterbi-decode a convolutionally encoded bitstream.

    Wraps commpy's conv_encode/viterbi_decode with sensible defaults for the
    NASA-standard rate 1/2, K=7 code, but accepts other configs for the
    search loop to try.

    Parameters
    ----------
    bits : np.ndarray
        1D array of encoded bits (hard decisions: 0 or 1).
    constraint_length : int, optional
        Constraint length K (default 7).
    rate : str, optional
        Code rate as a fraction string (default "1/2").

    Returns
    -------
    decoded_bits : np.ndarray
        1D array of decoded information bits.
    """
    bits = np.asarray(bits, dtype=int)
    if len(bits) == 0:
        return np.array([], dtype=int)

    trellis = _get_trellis(rate, constraint_length)

    # Traceback depth: standard rule of thumb is 5 * K
    tb_depth = 5 * constraint_length

    # commpy's viterbi_decode with 'hard' decoding type expects integer-valued
    # bits (0/1). Returns decoded message bits.
    decoded = cc.viterbi_decode(
        bits.astype(float),
        trellis,
        tb_depth,
        decoding_type="hard",
    )

    return decoded.astype(int)


def _conv_encode(
    info_bits: np.ndarray,
    constraint_length: int = 7,
    rate: str = "1/2",
) -> np.ndarray:
    """
    Convolutional-encode information bits (used internally for re-encoding
    during quality scoring).

    Parameters
    ----------
    info_bits : np.ndarray
        1D array of information bits.
    constraint_length : int
        Constraint length K.
    rate : str
        Code rate.

    Returns
    -------
    coded_bits : np.ndarray
        Encoded bit sequence.
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if len(info_bits) == 0:
        return np.array([], dtype=int)

    trellis = _get_trellis(rate, constraint_length)
    coded = cc.conv_encode(info_bits, trellis, termination="term")
    return coded.astype(int)


def decode_quality_score(
    original_encoded_bits: np.ndarray,
    decoded_info_bits: np.ndarray,
    constraint_length: int = 7,
    rate: str = "1/2",
) -> float:
    """
    Score how well a Viterbi decode "worked" by re-encoding the decoded output
    and comparing it to the original encoded bit sequence.

    Rationale:
      A correct decode followed by re-encode should closely reproduce the
      original encoded bits (before interleaving), because the convolutional
      code is deterministic. If the de-interleaving depth or FEC config was
      wrong, the Viterbi decoder produces garbage, and re-encoding that
      garbage will look very different from the input.

    Parameters
    ----------
    original_encoded_bits : np.ndarray
        The encoded bits that were fed into the Viterbi decoder (after
        de-interleaving, before decoding).
    decoded_info_bits : np.ndarray
        The Viterbi decoder's output (information bits).
    constraint_length : int
        Constraint length K used for decoding (and re-encoding).
    rate : str
        Code rate used for decoding (and re-encoding).

    Returns
    -------
    score : float
        Fraction of matching bits between the re-encoded sequence and the
        original encoded sequence (0.0 = no match, 1.0 = perfect match).
    """
    original_encoded_bits = np.asarray(original_encoded_bits, dtype=int)
    decoded_info_bits = np.asarray(decoded_info_bits, dtype=int)

    if len(original_encoded_bits) == 0 or len(decoded_info_bits) == 0:
        return 0.0

    # Re-encode the decoded information bits with the same FEC config
    re_encoded = _conv_encode(decoded_info_bits, constraint_length, rate)

    # Compare over the shorter of the two sequences
    n = min(len(re_encoded), len(original_encoded_bits))
    if n == 0:
        return 0.0

    matching = np.sum(re_encoded[:n] == original_encoded_bits[:n])
    return float(matching / n)
