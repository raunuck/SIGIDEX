"""
make_test_bpsk_fec_interleaved.py
---------------------------------
Generate a synthetic BPSK signal that has been:
  1. FEC-encoded (rate 1/2, constraint-length 7 convolutional code, NASA standard)
  2. Block-interleaved with a known depth

Then BPSK-modulated using the same parameters as make_test_bpsk.py.

Ground-truth metadata is saved to a companion .npz file and printed to stdout.
"""

import numpy as np
import soundfile as sf
import os
import commpy.channelcoding.convcode as cc


# ── Ground-truth parameters ─────────────────────────────────────────────────
SAMPLE_RATE     = 44_100        # Hz  (match make_test_bpsk.py)
CARRIER_FREQ    = 5_000         # Hz  (match make_test_bpsk.py)
SYMBOL_RATE     = 500           # symbols/sec  (match make_test_bpsk.py)
NUM_INFO_BITS   = 200           # number of information bits before FEC encoding
BIT_SEED        = 99            # different seed from original test_bpsk to avoid confusion
INTERLEAVER_DEPTH = 8           # block interleaver depth (rows)

# FEC: rate 1/2, K=7 convolutional code (NASA/CCSDS standard)
# Generator polynomials: 171, 133 (octal) → memory = K-1 = 6
CONSTRAINT_LENGTH = 7
FEC_RATE          = "1/2"
MEMORY            = np.array([CONSTRAINT_LENGTH - 1])       # [6]
G_MATRIX          = np.array([[0o171, 0o133]])               # NASA standard


def block_interleave(bits: np.ndarray, depth: int) -> np.ndarray:
    """
    Standard block interleaver.

    ENCODE direction (interleave):
      - Write bits into a (depth × cols) grid ROW by ROW.
      - Read them out COLUMN by COLUMN.
      - Pads with zeros to the next full grid if len(bits) is not a multiple of depth.

    This is the standard convention used in communications textbooks.
    The inverse operation (de-interleave) is in src/coding/deinterleave.py.
    """
    n = len(bits)
    cols = int(np.ceil(n / depth))
    total = depth * cols
    padded = np.zeros(total, dtype=bits.dtype)
    padded[:n] = bits
    matrix = padded.reshape(depth, cols)
    return matrix.T.flatten()


def main():
    # ── 1. Generate random information bits ──────────────────────────────
    rng = np.random.default_rng(BIT_SEED)
    info_bits = rng.integers(0, 2, size=NUM_INFO_BITS).astype(int)

    # ── 2. FEC encode (rate 1/2, K=7 convolutional) ─────────────────────
    trellis = cc.Trellis(MEMORY, G_MATRIX)
    coded_bits = cc.conv_encode(info_bits, trellis, termination="term")
    coded_bits = coded_bits.astype(int)
    # With termination, coded_length = 2 * (NUM_INFO_BITS + memory) = 2 * 206 = 412

    # ── 3. Block interleave ──────────────────────────────────────────────
    interleaved_bits = block_interleave(coded_bits, INTERLEAVER_DEPTH)
    # Padded to next multiple of INTERLEAVER_DEPTH if needed

    # ── 4. BPSK modulate (same approach as make_test_bpsk.py) ────────────
    # The interleaved bits become the transmitted symbols
    symbols = 2 * interleaved_bits - 1   # 0 → -1,  1 → +1

    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE
    baseband = np.repeat(symbols, samples_per_symbol).astype(np.float64)

    t = np.arange(len(baseband)) / SAMPLE_RATE
    carrier = np.cos(2 * np.pi * CARRIER_FREQ * t)
    signal = baseband * carrier

    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal))

    # ── 5. Write output files ────────────────────────────────────────────
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    wav_path = os.path.join(out_dir, "test_bpsk_fec_interleaved.wav")
    sf.write(wav_path, signal, SAMPLE_RATE, subtype="FLOAT")

    # Companion metadata file: everything needed to validate the search loop
    meta_path = os.path.join(out_dir, "test_bpsk_fec_interleaved_meta.npz")
    np.savez(
        meta_path,
        info_bits=info_bits,
        coded_bits=coded_bits,
        interleaved_bits=interleaved_bits,
        constraint_length=CONSTRAINT_LENGTH,
        fec_rate_str=FEC_RATE,
        interleaver_depth=INTERLEAVER_DEPTH,
        sample_rate=SAMPLE_RATE,
        carrier_freq=CARRIER_FREQ,
        symbol_rate=SYMBOL_RATE,
        num_info_bits=NUM_INFO_BITS,
        bit_seed=BIT_SEED,
    )

    # ── 6. Print ground-truth summary ────────────────────────────────────
    num_transmitted_symbols = len(interleaved_bits)
    duration = len(signal) / SAMPLE_RATE

    print("=" * 65)
    print("  FEC + Block-Interleaved BPSK Test Signal -- Ground Truth")
    print("=" * 65)
    print(f"  WAV file           : {wav_path}")
    print(f"  Metadata file      : {meta_path}")
    print(f"  Sample rate        : {SAMPLE_RATE} Hz")
    print(f"  Carrier frequency  : {CARRIER_FREQ} Hz")
    print(f"  Symbol rate        : {SYMBOL_RATE} sym/s")
    print(f"  Duration           : {duration:.3f} s")
    print(f"  -- Information ----------------------------")
    print(f"  Info bits          : {NUM_INFO_BITS}")
    print(f"  Bit seed           : {BIT_SEED}")
    print(f"  First 20 info bits : {list(info_bits[:20])}")
    print(f"  -- FEC ------------------------------------")
    print(f"  FEC type           : Convolutional")
    print(f"  Rate               : {FEC_RATE}")
    print(f"  Constraint length  : {CONSTRAINT_LENGTH} (memory={CONSTRAINT_LENGTH-1})")
    print(f"  Generators (octal) : 171, 133")
    print(f"  Coded bits         : {len(coded_bits)}")
    print(f"  -- Interleaver ----------------------------")
    print(f"  Type               : Block")
    print(f"  Depth              : {INTERLEAVER_DEPTH}")
    print(f"  Interleaved bits   : {len(interleaved_bits)}")
    print(f"  -- Transmitted ----------------------------")
    print(f"  Symbols (BPSK)     : {num_transmitted_symbols}")
    print(f"  Samples            : {len(signal)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
