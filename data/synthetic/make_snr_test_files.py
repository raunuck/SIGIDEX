import os
import numpy as np
import soundfile as sf

fs = 48000
symbol_rate = 1200
carrier_freq = 5000

out_dir = os.path.dirname(os.path.abspath(__file__))
np.random.seed(42)  # match make_test_bpsk.py so this is the SAME bit sequence
bits = np.random.randint(0, 2, 2000)
np.save(os.path.join(out_dir, "test_bpsk_snr_bits.npy"), bits)

samples_per_symbol = fs // symbol_rate
bpsk_symbols = 2 * bits - 1
signal = np.repeat(bpsk_symbols, samples_per_symbol).astype(np.float32)

t = np.arange(len(signal)) / fs
waveform = signal * np.cos(2 * np.pi * carrier_freq * t)
signal_power = np.mean(waveform**2)

for target_snr_db in [0, 10, 20]:
    noise_power = signal_power / (10**(target_snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(waveform))
    noisy = waveform + noise
    out_path = os.path.join(out_dir, f"test_bpsk_snr{target_snr_db}.wav")
    sf.write(out_path, noisy, fs)
    print(f"Wrote {out_path} (target SNR: {target_snr_db} dB)")