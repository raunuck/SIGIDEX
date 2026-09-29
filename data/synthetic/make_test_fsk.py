"""
make_test_fsk.py
----------------
Generate a synthetic 2-FSK signal (continuous-phase) and write it to
data/synthetic/test_fsk.wav.

Ground-truth parameters are printed to stdout.
"""

import numpy as np
import soundfile as sf
import os

# ── Ground-truth parameters ─────────────────────────────────────────────────
SAMPLE_RATE   = 44_100          # Hz
FREQ_0        = 4_000           # Hz — tone for bit 0
FREQ_1        = 6_000           # Hz — tone for bit 1
SYMBOL_RATE   = 1000            # symbols/sec
NUM_SYMBOLS   = 200             # total symbols
BIT_SEED      = 42              # reproducible random bit sequence


def make_fsk_signal():
    """Return a 1-D real-valued continuous-phase 2-FSK waveform and bits."""

    rng = np.random.default_rng(BIT_SEED)
    bits = rng.integers(0, 2, size=NUM_SYMBOLS)

    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE

    # Build per-sample instantaneous frequency
    freq_per_symbol = np.where(bits == 0, FREQ_0, FREQ_1)
    inst_freq = np.repeat(freq_per_symbol, samples_per_symbol).astype(np.float64)

    # Continuous-phase: integrate frequency to get phase
    phase = np.cumsum(2.0 * np.pi * inst_freq / SAMPLE_RATE)
    signal = np.cos(phase)

    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal))

    return signal, bits


def main():
    signal, bits = make_fsk_signal()

    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "test_fsk.wav")
    os.makedirs(out_dir, exist_ok=True)

    sf.write(out_path, signal, SAMPLE_RATE, subtype="FLOAT")

    center_freq = (FREQ_0 + FREQ_1) / 2
    deviation = (FREQ_1 - FREQ_0) / 2

    print("=" * 60)
    print("  2-FSK Test Signal -- Ground Truth")
    print("=" * 60)
    print(f"  Output file       : {out_path}")
    print(f"  Sample rate       : {SAMPLE_RATE} Hz")
    print(f"  Tone 0 (bit=0)    : {FREQ_0} Hz")
    print(f"  Tone 1 (bit=1)    : {FREQ_1} Hz")
    print(f"  Center frequency  : {center_freq} Hz")
    print(f"  Frequency deviation: {deviation} Hz")
    print(f"  Symbol rate       : {SYMBOL_RATE} sym/s")
    print(f"  Number of symbols : {NUM_SYMBOLS}")
    print(f"  Duration          : {len(signal) / SAMPLE_RATE:.3f} s")
    print(f"  Bit seed          : {BIT_SEED}")
    print(f"  First 20 bits     : {list(bits[:20])}")
    print("=" * 60)


if __name__ == "__main__":
    main()
