"""
make_test_fsk_baseband.py
-------------------------
Generate seeded synthetic complex baseband 2-FSK signals (carrier = 0 Hz, tones at -1 kHz / +1 kHz)
at clean, 20 dB, 10 dB, and 0 dB target SNR, saving directly as raw binary interleaved .IQ files
(float32, interleaved I0, Q0, I1, Q1) and saving companion ground-truth bits.
"""

import os
import numpy as np

SAMPLE_RATE = 44_100
FREQ_0 = -1_000.0  # Hz (bit 0, baseband tone)
FREQ_1 = +1_000.0  # Hz (bit 1, baseband tone) -> 2000 Hz shift across 0 Hz DC
SYMBOL_RATE = 1_000
NUM_SYMBOLS = 500
BIT_SEED = 42
NOISE_SEED = 12345


def make_baseband_fsk_waveform():
    rng = np.random.default_rng(BIT_SEED)
    bits = rng.integers(0, 2, size=NUM_SYMBOLS)

    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE
    freq_per_symbol = np.where(bits == 0, FREQ_0, FREQ_1)
    inst_freq = np.repeat(freq_per_symbol, samples_per_symbol).astype(np.float64)

    phase = np.cumsum(2.0 * np.pi * inst_freq / SAMPLE_RATE)
    # Complex baseband IQ
    signal = np.exp(1j * phase)
    return signal, bits


def save_iq(filepath: str, iq: np.ndarray):
    """Save complex IQ array as interleaved float32 binary."""
    interleaved = np.empty(2 * len(iq), dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    interleaved.tofile(filepath)


def main():
    signal, bits = make_baseband_fsk_waveform()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # Save ground-truth bits
    bits_path = os.path.join(out_dir, "test_fsk_baseband_bits.npy")
    np.save(bits_path, bits)
    print(f"Saved ground truth bits to {bits_path} ({len(bits)} bits)")

    # Clean IQ file
    clean_path = os.path.join(out_dir, "test_fsk_baseband.iq")
    save_iq(clean_path, signal)
    print(f"Wrote clean {clean_path}")

    signal_power = np.mean(np.abs(signal) ** 2)
    noise_rng = np.random.default_rng(NOISE_SEED)

    for target_snr_db in [20, 10, 0]:
        noise_power = signal_power / (10.0 ** (target_snr_db / 10.0))
        noise_i = noise_rng.normal(0, np.sqrt(noise_power / 2.0), len(signal))
        noise_q = noise_rng.normal(0, np.sqrt(noise_power / 2.0), len(signal))
        noise = noise_i + 1j * noise_q
        noisy = signal + noise
        out_path = os.path.join(out_dir, f"test_fsk_baseband_snr{target_snr_db}.iq")
        save_iq(out_path, noisy)
        print(f"Wrote {out_path} (target SNR: {target_snr_db} dB)")


if __name__ == "__main__":
    main()
