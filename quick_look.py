"""
quick_look.py
-------------
CLI demo and validation tool for BandsmithV2.

Loads a signal recording (.wav or .IQ), computes the full Bucket 1 diagnostic
report (RF parameters + modulation classification), prints it as JSON to stdout,
and optionally generates a waterfall (spectrogram) plot.

Usage:
    python quick_look.py
    python quick_look.py data/synthetic/test_bpsk.wav
    python quick_look.py data/synthetic/test_qpsk.wav
    python quick_look.py data/synthetic/test_fsk.wav
    python quick_look.py data/synthetic/test_bpsk.wav --no-gui
"""

import argparse
import json
import os
import sys
import numpy as np

# Ensure src/ is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingestion.wav_loader import load_wav
from ingestion.iq_loader import load_iq
from preprocessing.normalize import remove_dc_offset
from schema import build_bucket1_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="BandsmithV2 — Bucket 1 Diagnostic Report & Waterfall Viewer"
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=os.path.join("data", "synthetic", "test_bpsk.wav"),
        help="Path to signal file (.wav or .iq). Default: data/synthetic/test_bpsk.wav",
    )
    parser.add_argument(
        "--no-gui",
        "--no-plot",
        dest="no_gui",
        action="store_true",
        help="Skip generating the waterfall plot image; print report only.",
    )
    parser.add_argument(
        "--out-plot",
        default="waterfall_check.png",
        help="Output filename for waterfall plot image. Default: waterfall_check.png",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Sample rate in Hz (required for raw .iq files). Default: 44100",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["int8", "int16", "float32"],
        help="Data type for raw .iq files. Default: float32",
    )
    return parser.parse_args()


def load_signal(filepath: str, sample_rate: int, dtype: str):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".wav":
        return load_wav(filepath)
    elif ext in (".iq", ".bin", ".raw"):
        return load_iq(filepath, sample_rate=sample_rate, dtype=dtype)
    else:
        # Try wav first, fallback to iq
        try:
            return load_wav(filepath)
        except Exception:
            return load_iq(filepath, sample_rate=sample_rate, dtype=dtype)


def main():
    args = parse_args()

    if not os.path.isfile(args.file):
        print(f"[!] File not found: {args.file}", file=sys.stderr)
        print("    If testing synthetic signals, generate them first via:", file=sys.stderr)
        print("      python data/synthetic/make_test_bpsk.py", file=sys.stderr)
        print("      python data/synthetic/make_test_qpsk.py", file=sys.stderr)
        print("      python data/synthetic/make_test_fsk.py", file=sys.stderr)
        sys.exit(1)

    # ── Load signal ──────────────────────────────────────────────────────────
    iq, fs = load_signal(args.file, args.sample_rate, args.dtype)

    # ── Build Bucket 1 Report ────────────────────────────────────────────────
    report = build_bucket1_report(iq, fs)

    # Print JSON-formatted report to stdout
    print(json.dumps(report, indent=2))

    # ── Optional Waterfall Plot ──────────────────────────────────────────────
    if not args.no_gui:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt

        iq_clean = remove_dc_offset(iq)
        # Use real part if signal is passband/mono, else envelope
        if np.max(np.abs(iq_clean.imag)) < 1e-10 * (np.max(np.abs(iq_clean.real)) + 1e-30):
            plot_signal = iq_clean.real
        else:
            plot_signal = np.abs(iq_clean)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.specgram(
            plot_signal,
            NFFT=1024,
            Fs=fs,
            noverlap=512,
            cmap="inferno",
        )
        mod_type = report["modulation"]["type"]
        conf = report["modulation"]["confidence"]
        ax.set_title(f"Waterfall — {os.path.basename(args.file)} ({mod_type}, conf={conf:.2f})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_ylim(0, fs / 2)

        fig.savefig(args.out_plot, dpi=150, bbox_inches="tight")
        plt.close(fig)

        print(f"\n[OK] Waterfall plot saved to {args.out_plot}", file=sys.stderr)


if __name__ == "__main__":
    main()
