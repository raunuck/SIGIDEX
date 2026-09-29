"""
search.py
---------
Search-and-validate loop for blindly identifying the correct de-interleaving
depth and FEC configuration from a recovered bitstream.

Strategy:
  For each candidate (interleaver_depth, fec_config) combination:
    1. Block de-interleave the raw bits with the candidate depth.
    2. Viterbi-decode with the candidate FEC config.
    3. Re-encode the decoded output and compare to the de-interleaved input
       using decode_quality_score.
    4. The candidate with the highest score wins.

A correct depth + FEC config will produce a quality score very close to 1.0,
while wrong configs will score around 0.5 (random chance).
"""

from __future__ import annotations

import numpy as np

try:
    from coding.deinterleave import block_deinterleave
    from coding.fec import viterbi_decode, decode_quality_score
except ModuleNotFoundError:
    from src.coding.deinterleave import block_deinterleave
    from src.coding.fec import viterbi_decode, decode_quality_score


def search_and_validate(
    raw_bits: np.ndarray,
    candidate_depths: tuple[int, ...] = (4, 8, 16, 32),
    candidate_fec_configs: tuple[tuple[str, int], ...] = (("1/2", 7),),
) -> dict:
    """
    Try all candidate de-interleaver depths and FEC configurations, score
    each one, and return the best-scoring combination.

    Parameters
    ----------
    raw_bits : np.ndarray
        1D array of recovered bits (after demodulation, before de-interleaving
        or FEC decoding). May include the BPSK 180-degree phase ambiguity
        (all bits inverted); both polarities are tried automatically.
    candidate_depths : tuple of int
        Interleaver depths to try.
    candidate_fec_configs : tuple of (rate_str, constraint_length)
        FEC configurations to try, e.g. (("1/2", 7),).

    Returns
    -------
    result : dict
        {
          "winner": {
            "depth": int,
            "fec_rate": str,
            "fec_constraint_length": int,
            "score": float,
            "decoded_bits": np.ndarray,
            "polarity": str,  # "direct" or "inverted"
          },
          "all_candidates": [
            {
              "depth": int,
              "fec_rate": str,
              "fec_constraint_length": int,
              "score": float,
              "polarity": str,
            },
            ...
          ],
          "confidence_margin": float,  # gap between 1st and 2nd best scores
        }
    """
    raw_bits = np.asarray(raw_bits, dtype=int)
    if len(raw_bits) == 0:
        return {
            "winner": None,
            "all_candidates": [],
            "confidence_margin": 0.0,
        }

    all_candidates = []
    best_score = -1.0
    best_result = None

    # Try both BPSK polarities (direct and 180-degree inverted)
    polarities = {
        "direct": raw_bits,
        "inverted": 1 - raw_bits,
    }

    for polarity_name, bits in polarities.items():
        for depth in candidate_depths:
            for fec_rate, fec_k in candidate_fec_configs:
                # 1. De-interleave
                deinterleaved = block_deinterleave(bits, depth)

                # 2. Viterbi decode
                try:
                    decoded = viterbi_decode(
                        deinterleaved,
                        constraint_length=fec_k,
                        rate=fec_rate,
                    )
                except Exception:
                    # If decoding fails (e.g. too few bits for the trellis),
                    # score as zero and move on.
                    decoded = np.array([], dtype=int)

                # 3. Score: re-encode decoded output, compare to deinterleaved
                score = decode_quality_score(
                    deinterleaved,
                    decoded,
                    constraint_length=fec_k,
                    rate=fec_rate,
                )

                candidate_entry = {
                    "depth": depth,
                    "fec_rate": fec_rate,
                    "fec_constraint_length": fec_k,
                    "score": score,
                    "polarity": polarity_name,
                }
                all_candidates.append(candidate_entry)

                if score > best_score:
                    best_score = score
                    best_result = {
                        "depth": depth,
                        "fec_rate": fec_rate,
                        "fec_constraint_length": fec_k,
                        "score": score,
                        "decoded_bits": decoded,
                        "polarity": polarity_name,
                    }

    # Sort all candidates by score descending
    all_candidates.sort(key=lambda c: c["score"], reverse=True)

    # Confidence margin: gap between 1st and 2nd best scores
    if len(all_candidates) >= 2:
        margin = all_candidates[0]["score"] - all_candidates[1]["score"]
    else:
        margin = all_candidates[0]["score"] if all_candidates else 0.0

    return {
        "winner": best_result,
        "all_candidates": all_candidates,
        "confidence_margin": margin,
    }
