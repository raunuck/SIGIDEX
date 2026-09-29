from src.ingestion.wav_loader import load_wav
from src.features.modulation import classify_modulation, instantaneous_features
from src.features.spectral import compute_psd, estimate_bandwidth, estimate_snr
import numpy as np

files = ["test_bpsk.wav", "test_bpsk_snr20.wav", "test_bpsk_snr10.wav", "test_bpsk_snr0.wav"]

for fname in files:
    iq, fs = load_wav(f"data/synthetic/{fname}")

    result = classify_modulation(iq, fs)
    feats = instantaneous_features(iq)

    freqs, psd = compute_psd(iq, fs)
    bw = estimate_bandwidth(freqs, psd)
    snr = estimate_snr(freqs, psd)  # default band logic, same as the dashboard

    print(f"\n{fname}")
    print(f"  classified : {result['type']} (conf {result['confidence']})")
    print(f"  reason     : {result.get('reason')}")
    print(f"  runner_up  : {result.get('runner_up')}")
    print(f"  freq_iqr_norm: {feats['freq_iqr_norm']:.4f}")
    print(f"  amp_var      : {feats['amp_var']:.6f}")
    print(f"  bandwidth    : {bw:.1f} Hz")
    print(f"  in-band SNR  : {snr:.2f} dB")