"""
make_test_bpsk.py
-----------------
Generate a synthetic BPSK signal modulated onto a carrier and write it to
data/synthetic/test_bpsk.wav (clean) and test_bpsk_noisy.wav (with AWGN).

Ground-truth parameters are printed to stdout so they can be verified against
the waterfall plot produced by quick_look.py.
"""

import numpy as np
import soundfile as sf
import os

# ── Ground-truth parameters ─────────────────────────────────────────────────
SAMPLE_RATE   = 44_100          # Hz  (CD-quality, easy to work with)
CARRIER_FREQ  = 5_000           # Hz  (should appear as a bright line at 5 kHz)
SYMBOL_RATE   = 500             # symbols/sec
NUM_SYMBOLS   = 200             # total symbols → 0.4 s of data
BIT_SEED      = 42              # reproducible random bit sequence
NOISE_POWER   = 0.05            # AWGN variance for the noisy variant
NOISE_SEED    = 7               # reproducible noise


def make_bpsk_signal():
    """Return a 1-D real-valued BPSK waveform and the bit sequence used."""

    rng = np.random.default_rng(BIT_SEED)
    bits = rng.integers(0, 2, size=NUM_SYMBOLS)           # 0 or 1

    # Map bits → BPSK symbols: 0 → -1, 1 → +1
    symbols = 2 * bits - 1

    # Up-sample: each symbol lasts (SAMPLE_RATE / SYMBOL_RATE) samples
    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE
    baseband = np.repeat(symbols, samples_per_symbol).astype(np.float64)

    # Time vector
    num_samples = len(baseband)
    t = np.arange(num_samples) / SAMPLE_RATE

    # Modulate onto carrier
    carrier = np.cos(2 * np.pi * CARRIER_FREQ * t)
    signal = baseband * carrier

    # Normalize to [-1, 1] to avoid clipping in WAV
    signal /= np.max(np.abs(signal))

    return signal, bits


def main():
    signal, bits = make_bpsk_signal()

    out_dir  = os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # ── Clean version ────────────────────────────────────────────────────
    clean_path = os.path.join(out_dir, "test_bpsk.wav")
    sf.write(clean_path, signal, SAMPLE_RATE, subtype="FLOAT")

    # ── Ground-truth bits companion file ─────────────────────────────────
    bits_path = os.path.join(out_dir, "test_bpsk_bits.npy")
    np.save(bits_path, bits)
    print(f"Saved ground-truth bits to test_bpsk_bits.npy ({len(bits)} bits)")

    # ── Noisy version (AWGN) ─────────────────────────────────────────────
    noise_rng = np.random.default_rng(NOISE_SEED)
    noise = noise_rng.normal(0, np.sqrt(NOISE_POWER), len(signal))
    noisy_signal = signal + noise
    # Re-normalize so peaks don't clip
    noisy_signal /= np.max(np.abs(noisy_signal))

    noisy_path = os.path.join(out_dir, "test_bpsk_noisy.wav")
    sf.write(noisy_path, noisy_signal, SAMPLE_RATE, subtype="FLOAT")

    # ── Compute true SNR for reference ───────────────────────────────────
    signal_power = np.mean(signal ** 2)
    true_snr_db = 10 * np.log10(signal_power / NOISE_POWER)

    # ── Print ground-truth parameters ────────────────────────────────────
    print("=" * 60)
    print("  BPSK Test Signal -- Ground Truth")
    print("=" * 60)
    print(f"  Clean file        : {clean_path}")
    print(f"  Noisy file        : {noisy_path}")
    print(f"  Sample rate       : {SAMPLE_RATE} Hz")
    print(f"  Carrier frequency : {CARRIER_FREQ} Hz")
    print(f"  Symbol rate       : {SYMBOL_RATE} sym/s")
    print(f"  Number of symbols : {NUM_SYMBOLS}")
    print(f"  Duration          : {len(signal) / SAMPLE_RATE:.3f} s")
    print(f"  Bit seed          : {BIT_SEED}")
    print(f"  First 20 bits     : {list(bits[:20])}")
    print(f"  Noise power       : {NOISE_POWER}")
    print(f"  True SNR          : {true_snr_db:.1f} dB")
    print("=" * 60)


if __name__ == "__main__":
    main()

