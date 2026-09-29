"""
make_qpsk_snr_test_files.py
---------------------------
Generate seeded synthetic QPSK signals at 20, 10, and 0 dB target SNR,
and save ground-truth bits to test_qpsk_bits.npy.
"""

import os
import numpy as np
import soundfile as sf

SAMPLE_RATE = 44_100
CARRIER_FREQ = 5_000
SYMBOL_RATE = 500
NUM_SYMBOLS = 500  # 1000 bits
BIT_SEED = 42
NOISE_SEED = 12345

PHASE_MAP = {
    (0, 0): np.pi / 4,
    (0, 1): 3 * np.pi / 4,
    (1, 1): 5 * np.pi / 4,
    (1, 0): 7 * np.pi / 4,
}


def make_qpsk_waveform():
    rng = np.random.default_rng(BIT_SEED)
    bits = rng.integers(0, 2, size=NUM_SYMBOLS * 2)

    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE
    symbol_phases = []
    for i in range(NUM_SYMBOLS):
        pair = (int(bits[2 * i]), int(bits[2 * i + 1]))
        symbol_phases.append(PHASE_MAP[pair])

    phase_per_sample = np.repeat(symbol_phases, samples_per_symbol).astype(np.float64)
    t = np.arange(len(phase_per_sample)) / SAMPLE_RATE
    signal = np.cos(2.0 * np.pi * CARRIER_FREQ * t + phase_per_sample)
    signal = signal / np.max(np.abs(signal))
    return signal, bits


def main():
    signal, bits = make_qpsk_waveform()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # Save ground-truth bits
    bits_path = os.path.join(out_dir, "test_qpsk_bits.npy")
    np.save(bits_path, bits)
    print(f"Saved ground truth bits to {bits_path} ({len(bits)} bits)")

    signal_power = np.mean(signal ** 2)
    noise_rng = np.random.default_rng(NOISE_SEED)

    for target_snr_db in [20, 10, 0]:
        noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
        noise = noise_rng.normal(0, np.sqrt(noise_power), len(signal))
        noisy = signal + noise
        out_path = os.path.join(out_dir, f"test_qpsk_snr{target_snr_db}.wav")
        sf.write(out_path, noisy, SAMPLE_RATE, subtype="FLOAT")
        print(f"Wrote {out_path} (target SNR: {target_snr_db} dB)")


if __name__ == "__main__":
    main()
