"""
src/coding
----------
FEC decoding and de-interleaving package.
Week 5: Block de-interleaving and Viterbi convolutional decoding.
"""

from .deinterleave import block_deinterleave
from .fec import viterbi_decode, decode_quality_score
from .search import search_and_validate

__all__ = [
    "block_deinterleave",
    "viterbi_decode",
    "decode_quality_score",
    "search_and_validate",
]
