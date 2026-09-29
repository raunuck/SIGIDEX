"""
make_test_fsk_narrow.py
-----------------------
Generate seeded synthetic narrow-shift 2-FSK signals (200 Hz shift on 5 kHz carrier)
at clean, 20 dB, 10 dB, and 0 dB target SNR, and save companion ground-truth bits.
"""

import os
import numpy as np
import soundfile as sf

SAMPLE_RATE = 44_100
FREQ_0 = 4_900  # Hz (bit 0)
FREQ_1 = 5_100  # Hz (bit 1) -> 200 Hz shift
SYMBOL_RATE = 100
NUM_SYMBOLS = 500
BIT_SEED = 42
NOISE_SEED = 12345


def make_narrow_fsk_waveform():
    rng = np.random.default_rng(BIT_SEED)
    bits = rng.integers(0, 2, size=NUM_SYMBOLS)

    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE
    freq_per_symbol = np.where(bits == 0, FREQ_0, FREQ_1)
    inst_freq = np.repeat(freq_per_symbol, samples_per_symbol).astype(np.float64)

    phase = np.cumsum(2.0 * np.pi * inst_freq / SAMPLE_RATE)
    signal = np.cos(phase)
    signal = signal / np.max(np.abs(signal))
    return signal, bits


def main():
    signal, bits = make_narrow_fsk_waveform()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # Save ground-truth bits
    bits_path = os.path.join(out_dir, "test_fsk_narrow_bits.npy")
    np.save(bits_path, bits)
    print(f"Saved ground truth bits to {bits_path} ({len(bits)} bits)")

    # Clean file
    clean_path = os.path.join(out_dir, "test_fsk_narrow.wav")
    sf.write(clean_path, signal, SAMPLE_RATE, subtype="FLOAT")
    print(f"Wrote clean {clean_path}")

    signal_power = np.mean(signal ** 2)
    noise_rng = np.random.default_rng(NOISE_SEED)

    for target_snr_db in [20, 10, 0]:
        noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
        noise = noise_rng.normal(0, np.sqrt(noise_power), len(signal))
        noisy = signal + noise
        noisy /= np.max(np.abs(noisy))
        out_path = os.path.join(out_dir, f"test_fsk_narrow_snr{target_snr_db}.wav")
        sf.write(out_path, noisy, SAMPLE_RATE, subtype="FLOAT")
        print(f"Wrote {out_path} (target SNR: {target_snr_db} dB)")


if __name__ == "__main__":
    main()
