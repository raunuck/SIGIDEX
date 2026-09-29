"""
deinterleave.py
---------------
Block de-interleaver for reversing a standard block interleaver.

Standard block interleaver convention (ENCODE / interleave direction):
  - Write bits into a (depth x cols) grid ROW by ROW.
  - Read them out COLUMN by COLUMN.

Block de-interleaver (DECODE / de-interleave direction -- this module):
  - Write received bits into a (cols x depth) grid ROW by ROW
    (which is equivalent to filling the interleaver's grid COLUMN by COLUMN,
    reversing the column-by-column readout).
  - Read them back out COLUMN by COLUMN
    (which reconstructs the original row-by-row write order).

In matrix terms:
  Interleave:    vec -> reshape(depth, cols) -> transpose -> flatten
  De-interleave: vec -> reshape(cols, depth) -> transpose -> flatten
"""

import numpy as np


def block_deinterleave(bits: np.ndarray, depth: int) -> np.ndarray:
    """
    Standard block de-interleaver (inverse of block_interleave).

    Given a bit sequence that was interleaved by writing into a (depth x cols)
    grid row-by-row and reading column-by-column, this function reverses that
    operation to recover the original bit ordering.

    Parameters
    ----------
    bits : np.ndarray
        1D array of interleaved bits.
    depth : int
        Number of rows in the interleaver grid (the interleaver "depth").
        Must be a positive integer.

    Returns
    -------
    deinterleaved : np.ndarray
        1D array of de-interleaved bits in the original ordering.

    Raises
    ------
    ValueError
        If depth is not a positive integer.

    Notes
    -----
    If len(bits) is not an exact multiple of depth, the input is zero-padded
    to the next multiple of depth before de-interleaving. The returned array
    will be the full padded length. Callers that know the original un-padded
    length should truncate accordingly.
    """
    if depth <= 0:
        raise ValueError(f"depth must be a positive integer, got {depth}")

    bits = np.asarray(bits)
    n = len(bits)
    if n == 0:
        return np.array([], dtype=bits.dtype)

    cols = int(np.ceil(n / depth))
    total = depth * cols

    # Zero-pad to exact grid size if needed
    padded = np.zeros(total, dtype=bits.dtype)
    padded[:n] = bits

    # Reshape into (cols x depth) grid: bits fill row-by-row into the
    # transposed interleaver grid, then reading column-by-column restores
    # the original order.
    matrix = padded.reshape(cols, depth)
    return matrix.T.flatten()
