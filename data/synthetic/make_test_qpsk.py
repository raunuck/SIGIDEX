"""
make_test_qpsk.py
-----------------
Generate a synthetic QPSK signal modulated onto a carrier and write it to
data/synthetic/test_qpsk.wav.

Ground-truth parameters are printed to stdout.
"""

import numpy as np
import soundfile as sf
import os

# ── Ground-truth parameters ─────────────────────────────────────────────────
SAMPLE_RATE   = 44_100          # Hz
CARRIER_FREQ  = 5_000           # Hz
SYMBOL_RATE   = 500             # symbols/sec  (each symbol = 2 bits)
NUM_SYMBOLS   = 200             # total QPSK symbols (= 400 bits)
BIT_SEED      = 42              # reproducible random bit sequence

# Gray-coded QPSK constellation
#   bit pair  -> carrier phase offset
PHASE_MAP = {
    (0, 0): np.pi / 4,
    (0, 1): 3 * np.pi / 4,
    (1, 1): 5 * np.pi / 4,
    (1, 0): 7 * np.pi / 4,
}


def make_qpsk_signal():
    """Return a 1-D real-valued QPSK waveform and the bit sequence used."""

    rng = np.random.default_rng(BIT_SEED)
    bits = rng.integers(0, 2, size=NUM_SYMBOLS * 2)  # 2 bits per symbol

    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE

    # Map bit pairs to phase offsets
    symbol_phases = []
    for i in range(NUM_SYMBOLS):
        pair = (int(bits[2 * i]), int(bits[2 * i + 1]))
        symbol_phases.append(PHASE_MAP[pair])

    # Upsample phase to per-sample
    phase_per_sample = np.repeat(symbol_phases, samples_per_symbol).astype(np.float64)

    # Generate passband signal: cos(2*pi*fc*t + phi_k)
    t = np.arange(len(phase_per_sample)) / SAMPLE_RATE
    signal = np.cos(2.0 * np.pi * CARRIER_FREQ * t + phase_per_sample)

    # Normalize to [-1, 1]
    signal /= np.max(np.abs(signal))

    return signal, bits


def main():
    signal, bits = make_qpsk_signal()

    out_dir = os.path.dirname(__file__)
    out_path = os.path.join(out_dir, "test_qpsk.wav")
    os.makedirs(out_dir, exist_ok=True)

    sf.write(out_path, signal, SAMPLE_RATE, subtype="FLOAT")

    print("=" * 60)
    print("  QPSK Test Signal -- Ground Truth")
    print("=" * 60)
    print(f"  Output file       : {out_path}")
    print(f"  Sample rate       : {SAMPLE_RATE} Hz")
    print(f"  Carrier frequency : {CARRIER_FREQ} Hz")
    print(f"  Symbol rate       : {SYMBOL_RATE} sym/s (2 bits/symbol)")
    print(f"  Number of symbols : {NUM_SYMBOLS}")
    print(f"  Total bits        : {NUM_SYMBOLS * 2}")
    print(f"  Duration          : {len(signal) / SAMPLE_RATE:.3f} s")
    print(f"  Bit seed          : {BIT_SEED}")
    print(f"  First 20 bits     : {list(bits[:20])}")
    print(f"  Phase map (Gray)  : {PHASE_MAP}")
    print("=" * 60)


if __name__ == "__main__":
    main()
