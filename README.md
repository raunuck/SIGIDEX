# Sigdex

Automated system for ingesting `.wav` and `.IQ` radio-signal recordings, extracting RF parameters, classifying modulation, and generating diagnostic reports.

---

## Capabilities Overview

- **Ingestion & Preprocessing (Week 1)**: Robust `.wav` and raw `.IQ` loaders with DC offset removal and power normalization.
- **RF Parameter Extraction (Week 2)**: Power spectral density (Welch), bandwidth estimation (X-dB threshold), SNR estimation with mirror-peak filtering, and cyclostationary symbol-rate estimation.
- **Modulation Classification & Bucket 1 Reporting (Week 3)**: Classical (non-ML) decision-tree classifier using instantaneous amplitude and frequency features (FSK, PSK, QAM) with confidence scoring, assembled into the standardized Bucket 1 JSON diagnostic schema.

---

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux / macOS

# Install dependencies (numpy, scipy, matplotlib, soundfile, pytest)
pip install -r requirements.txt
```

### 2. Generate Synthetic Test Signals

```bash
# Generate BPSK (5 kHz carrier, 500 sym/s)
python data/synthetic/make_test_bpsk.py

# Generate QPSK (5 kHz carrier, 500 sym/s, Gray-coded)
python data/synthetic/make_test_qpsk.py

# Generate 2-FSK (tones at 4 kHz & 6 kHz, 500 sym/s)
python data/synthetic/make_test_fsk.py
```

### 3. Run Quick Look CLI Demo

Run on any synthetic or real recording to output the full JSON Bucket 1 diagnostic report and produce a waterfall plot:

```bash
# Default: runs on test_bpsk.wav and saves waterfall_check.png
python quick_look.py

# Run on QPSK signal
python quick_look.py data/synthetic/test_qpsk.wav

# Run on FSK signal without displaying/saving waterfall plot
python quick_look.py data/synthetic/test_fsk.wav --no-gui
```

#### Example Bucket 1 Output

```json
{
  "rf": {
    "sampling_rate_hz": 44100,
    "bandwidth_hz": 13436.72,
    "snr_db": 23.45
  },
  "modulation": {
    "type": "PSK",
    "symbol_rate_sps": 501.14,
    "confidence": 0.7575
  },
  "coding": {
    "interleaver": {
      "type": "unknown",
      "confidence": 0.0
    },
    "fec": {
      "type": "unknown",
      "confidence": 0.0
    }
  }
}
```

---

## Running Tests

Run the comprehensive pytest test suite covering ingestion, spectral features, symbol rate estimation, modulation classification, and schema validation:

```bash
pytest tests/ -v
```

---

## Project Structure

```
BandsmithV2/
├── data/
│   ├── raw/                       # Place real .wav / .IQ recordings here
│   └── synthetic/
│       ├── make_test_bpsk.py      # Generates test_bpsk.wav (BPSK, 500 sym/s)
│       ├── make_test_qpsk.py      # Generates test_qpsk.wav (QPSK, 500 sym/s)
│       └── make_test_fsk.py       # Generates test_fsk.wav (2-FSK, 4/6 kHz)
├── src/
│   ├── ingestion/
│   │   ├── wav_loader.py          # Load .wav → complex IQ array
│   │   ├── iq_loader.py           # Load raw .IQ → complex IQ array
│   │   └── metadata.py            # SignalMetadata dataclass
│   ├── preprocessing/
│   │   └── normalize.py           # DC offset removal, power normalization
│   ├── features/
│   │   ├── spectral.py            # Welch PSD, bandwidth, SNR estimation
│   │   ├── symbol_rate.py         # Cyclostationary symbol-rate estimation
│   │   └── modulation.py          # Instantaneous features & rule-based classifier
│   └── schema.py                  # Bucket 1 diagnostic report builder
├── tests/
│   ├── test_ingestion.py          # Ingestion & normalization tests
│   ├── test_spectral.py           # Spectral analysis & SNR tests
│   ├── test_symbol_rate.py        # Symbol rate estimator tests
│   ├── test_modulation.py         # Modulation classification tests
│   └── test_schema.py             # Bucket 1 schema validation tests
├── quick_look.py                  # CLI demo: JSON report + waterfall plot
├── requirements.txt
└── README.md
```
