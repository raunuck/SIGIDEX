"""
make_test_bpsk_2400.py
----------------------
Generate a BPSK signal at 2400 sym/s (higher than the 500 sym/s default)
for cross-validation of Week 2 feature extractors.

Ground-truth parameters are printed to stdout.
"""

import numpy as np
import soundfile as sf
import os

# ── Ground-truth parameters ─────────────────────────────────────────────────
SAMPLE_RATE   = 44_100          # Hz
CARRIER_FREQ  = 8_000           # Hz  (different from the 500-sym/s file)
SYMBOL_RATE   = 2_400           # symbols/sec
NUM_SYMBOLS   = 800             # total symbols -> 0.333 s of data
BIT_SEED      = 99              # different seed for variety


def make_bpsk_signal():
    """Return a 1-D real-valued BPSK waveform and the bit sequence used."""

    rng = np.random.default_rng(BIT_SEED)
    bits = rng.integers(0, 2, size=NUM_SYMBOLS)

    # Map bits -> BPSK symbols: 0 -> -1, 1 -> +1
    symbols = 2 * bits - 1

    # Up-sample: each symbol lasts (SAMPLE_RATE / SYMBOL_RATE) samples
    # Use integer division; 44100 / 2400 = 18.375 -> 18 samples/symbol
    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE
    baseband = np.repeat(symbols, samples_per_symbol).astype(np.float64)

    # Time vector
    num_samples = len(baseband)
    t = np.arange(num_samples) / SAMPLE_RATE

    # Modulate onto carrier
    carrier = np.cos(2 * np.pi * CARRIER_FREQ * t)
    signal = baseband * carrier

    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal))

    return signal, bits


def main():
    signal, bits = make_bpsk_signal()

    out_dir  = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "test_bpsk_2400.wav")
    os.makedirs(out_dir, exist_ok=True)

    sf.write(out_path, signal, SAMPLE_RATE, subtype="FLOAT")

    print("=" * 60)
    print("  BPSK Test Signal (2400 sym/s) -- Ground Truth")
    print("=" * 60)
    print(f"  Output file       : {out_path}")
    print(f"  Sample rate       : {SAMPLE_RATE} Hz")
    print(f"  Carrier frequency : {CARRIER_FREQ} Hz")
    print(f"  Symbol rate       : {SYMBOL_RATE} sym/s")
    print(f"  Number of symbols : {NUM_SYMBOLS}")
    print(f"  Duration          : {len(signal) / SAMPLE_RATE:.3f} s")
    print(f"  Bit seed          : {BIT_SEED}")
    print(f"  First 20 bits     : {list(bits[:20])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
