"""
backend/main.py
---------------
FastAPI server exposing SIGDEX signal analyzer & demodulator endpoints.
Connects directly to existing src/ modules without modifying any src/ logic.
"""

from __future__ import annotations

import os
import sys
import uuid
import tempfile
from datetime import datetime
from typing import Any

# Ensure src/ is importable
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
import scipy.signal as sig
import soundfile as sf

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Backend DSP imports from src/ (DO NOT MODIFY src/ files)
from ingestion.wav_loader import load_wav
from ingestion.iq_loader import load_iq, guess_iq_format
from preprocessing.normalize import remove_dc_offset, normalize_power
from features.spectral import compute_psd
from schema import build_bucket1_report
from demod.psk import demodulate_bpsk_full
from coding.search import search_and_validate
from correlation.framing import (
    SYNC_WORD_16,
    find_sync_word,
    extract_frame,
    bits_to_ascii,
    decode_header_bytes,
)

# Tunable FEC score gate: result is treated as "FEC detected" only if score >= 0.95
FEC_SCORE_THRESHOLD = 0.95

app = FastAPI(title="SIGDEX DSP API", version="2.4.0")

# Enable CORS for local Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for active signal sessions
# Key: file_id -> dict with processed signal array and metadata
SIGNAL_STORE: dict[str, dict[str, Any]] = {}

# In-memory recent operations list
RECENT_OPERATIONS: list[dict[str, Any]] = []


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / 1024:.1f} KB"


def _format_sample_rate(fs: int) -> str:
    if fs >= 1_000_000:
        return f"{fs / 1_000_000:.3f} MHz"
    return f"{fs / 1_000:.1f} kHz"


def _init_synthetic_recents():
    """Preload available synthetic files into recent operations."""
    data_dir = os.path.join(root_dir, "data", "synthetic")
    default_files = [
        ("test_framed_message.wav", 44_100, "WAV Baseband"),
        ("test_bpsk.wav", 44_100, "WAV Baseband"),
        ("test_fsk.wav", 44_100, "WAV Baseband"),
        ("test_bpsk_fec_interleaved.wav", 44_100, "WAV Baseband"),
    ]
    for fn, fs, dt in default_files:
        p = os.path.join(data_dir, fn)
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            RECENT_OPERATIONS.append({
                "filename": fn,
                "filepath": p,
                "timestamp": "2026-09-28 12:00:00",
                "sample_rate_str": _format_sample_rate(fs),
                "sample_rate": fs,
                "filesize_str": _format_size(sz),
                "data_type": dt,
                "is_synthetic": True,
            })


_init_synthetic_recents()


@app.get("/api/health")
def health_check():
    return {"status": "ok", "engine": "SIGDEX DSP ACTIVE", "version": "2.4-PRO"}


@app.post("/api/ingest")
async def ingest_file(
    file: UploadFile = File(...),
    sample_rate: int | None = Form(None),
    data_type: str | None = Form(None),
):
    """
    Accepts file upload (.wav or .iq), runs DC removal & power normalization,
    stores in memory, and returns session ID + signal metadata.
    """
    filename = file.filename or "uploaded_signal.bin"
    ext = os.path.splitext(filename)[1].lower()

    # Save to temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"sigdex_{uuid.uuid4().hex[:6]}_{filename}")
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    size_bytes = len(content)

    try:
        is_iq = ext not in (".wav", ".wave")
        guessed_format = None
        is_onesided = False

        if not is_iq:
            # Check WAV channels: mono WAV is loaded as an analytic signal via Hilbert transform
            info = sf.info(temp_path)
            is_onesided = (info.channels == 1)
            iq, fs = load_wav(temp_path)
            dt_label = "WAV Audio / IF Baseband"
        else:
            # Raw IQ file: true complex IQ with two-sided spectrum
            is_onesided = False
            guessed = guess_iq_format(temp_path)
            guessed_format = {
                "suggested_dtype": guessed.get("suggested_dtype"),
                "suggested_rate": guessed.get("suggested_rate"),
                "detected_patterns": guessed.get("detected_patterns", []),
            }
            fs = sample_rate or guessed.get("suggested_rate") or 44_100
            dtype = data_type or guessed.get("suggested_dtype") or "float32"
            iq, fs = load_iq(temp_path, sample_rate=fs, data_type=dtype)
            dt_label = f"Raw IQ ({dtype})"

        # Preprocessing: DC offset removal + unit power normalization
        iq_clean = remove_dc_offset(iq)
        norm_iq = normalize_power(iq_clean)

        file_id = uuid.uuid4().hex[:8]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        SIGNAL_STORE[file_id] = {
            "file_id": file_id,
            "filename": filename,
            "filepath": temp_path,
            "iq": norm_iq,
            "raw_iq": iq,
            "fs": int(fs),
            "size_bytes": size_bytes,
            "duration_s": float(len(norm_iq) / fs),
            "sample_count": len(norm_iq),
            "data_type": dt_label,
            "timestamp": now_str,
            "is_iq": is_iq,
            "is_onesided": is_onesided,
            "guessed_format": guessed_format,
            "report": None,
            "bitstream_result": None,
        }

        # Track in recent operations (unique by filename)
        op_entry = {
            "file_id": file_id,
            "filename": filename,
            "filepath": temp_path,
            "timestamp": now_str,
            "sample_rate_str": _format_sample_rate(fs),
            "sample_rate": int(fs),
            "filesize_str": _format_size(size_bytes),
            "data_type": dt_label,
            "is_synthetic": False,
        }
        global RECENT_OPERATIONS
        RECENT_OPERATIONS = [r for r in RECENT_OPERATIONS if r["filename"] != filename]
        RECENT_OPERATIONS.insert(0, op_entry)
        if len(RECENT_OPERATIONS) > 12:
            RECENT_OPERATIONS = RECENT_OPERATIONS[:12]

        return {
            "file_id": file_id,
            "filename": filename,
            "sample_rate": int(fs),
            "sample_count": len(norm_iq),
            "duration_s": round(float(len(norm_iq) / fs), 3),
            "filesize_str": _format_size(size_bytes),
            "data_type": dt_label,
            "is_iq": is_iq,
            "is_onesided": is_onesided,
            "guessed_format": guessed_format,
        }

    except Exception as e:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/ingest/sample/{filename}")
def ingest_sample(filename: str):
    """Convenience endpoint to load one of the pre-generated synthetic files directly."""
    data_dir = os.path.join(root_dir, "data", "synthetic")
    target_path = os.path.join(data_dir, filename)
    if not os.path.isfile(target_path):
        raise HTTPException(status_code=404, detail=f"Sample file {filename} not found")

    is_iq = not (filename.lower().endswith(".wav") or filename.lower().endswith(".wave"))
    is_onesided = False
    if not is_iq:
        info = sf.info(target_path)
        is_onesided = (info.channels == 1)
        iq, fs = load_wav(target_path)
        dt_label = "WAV Baseband"
    else:
        is_onesided = False
        iq, fs = load_iq(target_path, sample_rate=44_100, data_type="float32")
        dt_label = "Raw IQ (float32)"

    iq_clean = remove_dc_offset(iq)
    norm_iq = normalize_power(iq_clean)

    file_id = uuid.uuid4().hex[:8]
    size_bytes = os.path.getsize(target_path)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    SIGNAL_STORE[file_id] = {
        "file_id": file_id,
        "filename": filename,
        "filepath": target_path,
        "iq": norm_iq,
        "raw_iq": iq,
        "fs": int(fs),
        "size_bytes": size_bytes,
        "duration_s": float(len(norm_iq) / fs),
        "sample_count": len(norm_iq),
        "data_type": dt_label,
        "timestamp": now_str,
        "is_iq": is_iq,
        "is_onesided": is_onesided,
        "guessed_format": None,
        "report": None,
        "bitstream_result": None,
    }

    return {
        "file_id": file_id,
        "filename": filename,
        "sample_rate": int(fs),
        "sample_count": len(norm_iq),
        "duration_s": round(float(len(norm_iq) / fs), 3),
        "filesize_str": _format_size(size_bytes),
        "data_type": dt_label,
        "is_iq": is_iq,
        "is_onesided": is_onesided,
        "guessed_format": None,
    }


@app.get("/api/analysis/{file_id}")
def get_analysis(file_id: str):
    """
    Runs RF parameter extraction, modulation classification, and generates
    interactive PSD chart data and spectrogram matrix JSON.
    """
    if file_id not in SIGNAL_STORE:
        raise HTTPException(status_code=404, detail="Signal file not found")

    entry = SIGNAL_STORE[file_id]
    iq = entry["iq"]
    fs = entry["fs"]
    is_onesided = entry.get("is_onesided", True)

    # 1. Bucket 1 Diagnostic Report
    if entry.get("report") is None:
        try:
            entry["report"] = build_bucket1_report(iq, fs)
        except Exception as e:
            print(f"ERR: bucket1_report: {e}")
            entry["report"] = {
                "rf": {"center_frequency_hz": 0.0, "bandwidth_hz": 0.0, "snr_db": 0.0},
                "modulation": {"type": "unknown", "confidence": 0.0, "reason": str(e)},
            }
    report = entry["report"]

    # 2. Power Spectral Density (PSD)
    nperseg = min(len(iq), 2048)
    freqs, psd = compute_psd(iq, fs, nperseg=nperseg)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-12))
    freqs_khz = freqs / 1e3

    # For mono .wav (analytic signal), freqs >= 0 is the physical band (one-sided)
    if is_onesided:
        mask = freqs >= 0
        f_psd = freqs_khz[mask]
        p_psd = psd_db[mask]
        spectrum_mode = "one-sided"
    else:
        f_psd = freqs_khz
        p_psd = psd_db
        spectrum_mode = "two-sided"

    # Sort frequencies monotonically
    sort_idx = np.argsort(f_psd)
    f_psd = f_psd[sort_idx]
    p_psd = p_psd[sort_idx]

    # Downsample PSD to at most 1000 points for smooth frontend plotting
    if len(f_psd) > 1000:
        step = int(np.ceil(len(f_psd) / 1000))
        ds_freqs = f_psd[::step]
        ds_psd = p_psd[::step]
    else:
        ds_freqs = f_psd
        ds_psd = p_psd

    peak_idx = int(np.argmax(p_psd))
    peak_freq_khz = float(f_psd[peak_idx])
    peak_power_db = float(p_psd[peak_idx])

    # Y-axis clipping: ~80 dB below peak so empty floor doesn't dominate
    clip_ymin = peak_power_db - 80.0
    clip_ymax = peak_power_db + 5.0

    # 3. Spectrogram Matrix as JSON (NOT a PNG)
    n_fft = min(len(iq), 256)
    n_overlap = n_fft // 2
    hop_samples = n_fft - n_overlap

    f_spec, t_spec, s_spec = sig.spectrogram(
        iq,
        fs=fs,
        nperseg=n_fft,
        noverlap=n_overlap,
        return_onesided=False,
        mode="psd",
    )
    s_db = 10.0 * np.log10(np.maximum(s_spec, 1e-12))

    if is_onesided:
        mask = (f_spec >= 0) & (f_spec <= fs / 2)
        f_axis = f_spec[mask] / 1e3
        z_mat = s_db[mask, :].T  # shape: (time, freq)
        sort_f = np.argsort(f_axis)
        f_axis = f_axis[sort_f]
        z_mat = z_mat[:, sort_f]
    else:
        f_shifted = np.fft.fftshift(f_spec)
        f_shifted = f_shifted - fs * (f_shifted >= fs / 2)
        s_shifted = np.fft.fftshift(s_db, axes=0)
        sort_f = np.argsort(f_shifted)
        f_axis = f_shifted[sort_f] / 1e3
        z_mat = s_shifted[sort_f, :].T

    # Limit time slices to ~350 for fast JSON transmission and snappy 60 FPS rendering
    if len(t_spec) > 350:
        t_step = int(np.ceil(len(t_spec) / 350))
        t_axis = t_spec[::t_step]
        z_mat = z_mat[::t_step, :]
    else:
        t_axis = t_spec

    # Clip color range to top ~60 dB below the peak
    max_spec_val = float(np.max(z_mat))
    z_min = max_spec_val - 60.0
    z_max = max_spec_val
    z_clipped = np.clip(z_mat, z_min, z_max)

    spectrogram_data = {
        "f_khz": [round(float(f), 3) for f in f_axis],
        "t_sec": [round(float(t), 4) for t in t_axis],
        "z_db": np.round(z_clipped, 1).tolist(),
        "nperseg": n_fft,
        "noverlap": n_overlap,
        "hop_samples": hop_samples,
        "z_min": round(z_min, 1),
        "z_max": round(z_max, 1),
    }

    return {
        "file_id": file_id,
        "filename": entry["filename"],
        "sample_rate": fs,
        "spectrum_mode": spectrum_mode,
        "report": report,
        "psd": {
            "freqs_khz": [round(float(f), 3) for f in ds_freqs],
            "psd_db": [round(float(p), 2) for p in ds_psd],
            "y_min": round(clip_ymin, 1),
            "y_max": round(clip_ymax, 1),
        },
        "peak_freq_khz": round(peak_freq_khz, 3),
        "peak_power_db": round(peak_power_db, 2),
        "spectrogram": spectrogram_data,
        "fec": entry.get("bitstream_result"),
    }


@app.get("/api/constellation/{file_id}")
def get_constellation(file_id: str):
    """
    Runs carrier recovery & symbol timing to extract phase-corrected symbols,
    EVM RMS, and demodulator signal metrics.
    Guard against non-BPSK signals: if classifier detects FSK/QAM, returns unsupported.
    """
    if file_id not in SIGNAL_STORE:
        raise HTTPException(status_code=404, detail="Signal file not found")

    entry = SIGNAL_STORE[file_id]
    iq = entry["iq"]
    fs = entry["fs"]

    # 1. Modulation check: Demodulator is BPSK-only
    if entry.get("report") is None:
        try:
            entry["report"] = build_bucket1_report(iq, fs)
        except Exception as e:
            print(f"ERR: modulation_check: {e}")
            entry["report"] = {"modulation": {"type": "unknown", "confidence": 0.0}}

    mod_type = entry["report"].get("modulation", {}).get("type", "unknown").lower()
    if mod_type in ("fsk", "qam"):
        msg = f"DEMOD: not available for {mod_type.upper()} (BPSK only in this build)"
        print(msg)
        return {
            "file_id": file_id,
            "supported": False,
            "mod_type": mod_type.upper(),
            "message": msg,
            "symbol_rate": None,
            "carrier_freq": None,
            "offset_hz": None,
            "evm_rms": None,
            "symbol_count": 0,
            "carrier_locked": False,
            "points": [],
        }

    # 2. Demodulation stage wrapped in try/except
    try:
        demod = demodulate_bpsk_full(iq, fs)
    except Exception as e:
        err_msg = f"ERR: demodulation: {str(e)}"
        print(err_msg)
        return {
            "file_id": file_id,
            "supported": True,
            "stage": "demodulation",
            "error": str(e),
            "symbol_rate": None,
            "carrier_freq": None,
            "offset_hz": None,
            "evm_rms": None,
            "symbol_count": 0,
            "carrier_locked": False,
            "points": [],
        }

    symbols = demod.get("symbols", np.array([]))
    evm_rms = float(demod.get("evm_rms", 0.0))

    # Real carrier lock metric: symbols recovered and EVM < 45%
    carrier_locked = bool(len(symbols) > 50 and evm_rms < 45.0)

    # Normalize symbols so real centroids are around +/- 1.0
    if len(symbols) > 0:
        ref_scale = float(np.median(np.abs(np.real(symbols))))
        if ref_scale > 1e-12:
            norm_symbols = symbols / ref_scale
        else:
            norm_symbols = symbols

        # Sample up to 800 points for frontend rendering
        max_points = 800
        if len(norm_symbols) > max_points:
            indices = np.linspace(0, len(norm_symbols) - 1, max_points, dtype=int)
            sample_points = norm_symbols[indices]
        else:
            sample_points = norm_symbols

        points = [
            {"i": round(float(np.real(s)), 4), "q": round(float(np.imag(s)), 4)}
            for s in sample_points
        ]
    else:
        points = []

    cf = float(demod.get("carrier_freq", 0.0))
    # Estimated offset: only show if computed against a real reference (e.g. metadata).
    # Since test captures have no external ground-truth carrier reference, report None (N/A)
    # rather than hardcoding against 5000.0 Hz.
    offset_hz = None

    return {
        "file_id": file_id,
        "supported": True,
        "symbol_rate": round(float(demod.get("symbol_rate", 500.0)), 1),
        "carrier_freq": round(cf, 1),
        "offset_hz": offset_hz,
        "evm_rms": round(evm_rms, 2),
        "symbol_count": len(symbols),
        "carrier_locked": carrier_locked,
        "points": points,
    }


@app.get("/api/bitstream/{file_id}")
def get_bitstream(file_id: str):
    """
    Executes the full pipeline:
      demodulate_bpsk -> search_and_validate -> find_sync_word -> extract_frame -> bits_to_ascii
    Returns candidate ranking, frame lock status, recovered payload, hex dump, and real pipeline logs.
    """
    if file_id not in SIGNAL_STORE:
        raise HTTPException(status_code=404, detail="Signal file not found")

    entry = SIGNAL_STORE[file_id]
    iq = entry["iq"]
    fs = entry["fs"]
    fn = entry["filename"]

    logs: list[dict[str, str]] = []

    def log(tag: str, msg: str):
        logs.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "tag": tag,
            "message": msg,
        })

    log("INIT", f"Target baseband capture: {fn}")
    log("INGEST", f"Loaded {len(iq):,} complex samples at {fs:,} Hz")
    log("DSP", "DC bias removed and signal normalized to unit power")

    # 1. Check modulation type: Demodulator is BPSK-only
    if entry.get("report") is None:
        try:
            entry["report"] = build_bucket1_report(iq, fs)
        except Exception as e:
            print(f"ERR: modulation_check: {e}")
            entry["report"] = {"modulation": {"type": "unknown", "confidence": 0.0}}

    mod_type = entry["report"].get("modulation", {}).get("type", "unknown").lower()
    if mod_type in ("fsk", "qam"):
        unsupported_msg = f"DEMOD: not available for {mod_type.upper()} (BPSK only in this build)"
        print(unsupported_msg)
        log("WARN", unsupported_msg)
        res = {
            "file_id": file_id,
            "supported": False,
            "mod_type": mod_type.upper(),
            "message": unsupported_msg,
            "winner": None,
            "fec_detected": False,
            "fec_summary": unsupported_msg,
            "candidate_scores": [],
            "sync_found": False,
            "sync_position": None,
            "payload_ascii": "",
            "payload_hex": [],
            "payload_bytes_count": 0,
            "dump_label": f"DEMODULATION NOT AVAILABLE FOR {mod_type.upper()}",
            "dump_is_raw": False,
            "truncated": False,
            "logs": logs,
        }
        entry["bitstream_result"] = res
        return res

    # 2. Stage: DEMODULATION
    raw_bits = np.array([], dtype=int)
    try:
        demod = demodulate_bpsk_full(iq, fs)
        raw_bits = demod.get("bits", np.array([], dtype=int))
        evm = demod.get("evm_rms", 0.0)
        log("DEMOD", f"Costas loop locked ({len(raw_bits)} bits recovered, EVM: {evm:.2f}% RMS)")
    except Exception as e:
        err_msg = f"ERR: demodulation: {str(e)}"
        print(err_msg)
        log("ERROR", err_msg)
        res = {
            "file_id": file_id,
            "supported": True,
            "stage": "demodulation",
            "error": str(e),
            "winner": None,
            "fec_detected": False,
            "fec_summary": f"Demodulation failed: {str(e)}",
            "candidate_scores": [],
            "sync_found": False,
            "sync_position": None,
            "payload_ascii": "",
            "payload_hex": [],
            "payload_bytes_count": 0,
            "dump_label": "NO DEMODULATED BITS AVAILABLE",
            "dump_is_raw": False,
            "truncated": False,
            "logs": logs,
        }
        entry["bitstream_result"] = res
        return res

    # 3. Stage: FEC SEARCH & VALIDATION
    fec_detected = False
    winner_payload = None
    decoded_bits = np.array([], dtype=int)
    all_candidates: list[dict[str, Any]] = []

    try:
        search_res = search_and_validate(raw_bits)
        winner = search_res.get("winner")
        all_candidates = search_res.get("all_candidates", [])

        # Honest FEC gate: best score must be >= FEC_SCORE_THRESHOLD (0.95)
        if winner and winner["score"] >= FEC_SCORE_THRESHOLD:
            fec_detected = True
            log(
                "FEC_SEARCH",
                f"Viterbi search identified depth {winner['depth']} "
                f"(rate: {winner['fec_rate']}, K: {winner['fec_constraint_length']}, score: {winner['score']:.4f})",
            )
            decoded_bits = winner["decoded_bits"]
            winner_payload = {
                "depth": winner["depth"],
                "fec_rate": winner["fec_rate"],
                "fec_constraint_length": winner["fec_constraint_length"],
                "score": round(float(winner["score"]), 4),
                "polarity": winner["polarity"],
            }
        else:
            best_score_str = f"{winner['score']:.4f}" if winner else "N/A"
            log("FEC_SEARCH", f"FEC: not detected (best score {best_score_str})")
            winner_payload = None
            decoded_bits = np.array([], dtype=int)
    except Exception as e:
        err_msg = f"ERR: fec_search: {str(e)}"
        print(err_msg)
        log("ERROR", err_msg)
        winner_payload = None
        decoded_bits = np.array([], dtype=int)

    # 4. Stage: FRAME SYNC & PAYLOAD EXTRACTION
    sync_found = False
    sync_pos = None
    recovered_text = ""
    payload_bits = np.array([], dtype=int)
    truncated = False

    if fec_detected and len(decoded_bits) > 0:
        try:
            sync_matches = find_sync_word(decoded_bits, SYNC_WORD_16, threshold=0.9)
            if sync_matches:
                sync_found = True
                sync_pos = sync_matches[0]
                log("SYNC_LOCK", f"16-bit telemetry sync word (0xEB90) acquired at offset {sync_pos}")

                frame_data = extract_frame(
                    bits=decoded_bits,
                    sync_position=sync_pos,
                    sync_word_length=len(SYNC_WORD_16),
                    header_length=16,
                    header_decoder=decode_header_bytes,
                )
                payload_bits = frame_data["payload_bits"]
                truncated = frame_data["truncated"]
                recovered_text = bits_to_ascii(payload_bits)
                log("PAYLOAD", f"Extracted {len(payload_bits)} payload bits -> ASCII: '{recovered_text}'")
            else:
                log("WARN", "Sync word correlation failed (no match above threshold 0.9)")
        except Exception as e:
            err_msg = f"ERR: frame_sync: {str(e)}"
            print(err_msg)
            log("ERROR", err_msg)

    log("DONE", "Pipeline execution complete.")

    # 5. Format Hexadecimal Dump:
    # If FEC / sync succeeded, show frame payload bytes.
    # When FEC or sync fails, still show the raw demodulated bitstream as hex/ASCII in the hex dump,
    # labeled "RAW DEMODULATED BITS (FEC not resolved)". Never show an empty dump when demodulation succeeded.
    if sync_found and len(payload_bits) > 0:
        dump_bits = payload_bits
        dump_label = "FRAME PAYLOAD (SYNC ACQUIRED)"
        dump_is_raw = False
    elif len(raw_bits) > 0:
        dump_bits = raw_bits
        dump_label = "RAW DEMODULATED BITS (FEC not resolved)"
        dump_is_raw = True
    else:
        dump_bits = np.array([], dtype=int)
        dump_label = "NO BITS AVAILABLE"
        dump_is_raw = False

    n_bytes = len(dump_bits) // 8
    # Cap displayed rows to at most 512 bytes (32 rows) for responsive rendering
    display_bytes_count = min(n_bytes, 512)
    raw_bytes = bytearray()
    for i in range(display_bytes_count):
        val = 0
        for b in dump_bits[i * 8 : (i + 1) * 8]:
            val = (val << 1) | int(b)
        raw_bytes.append(val)

    hex_dump_lines = []
    for offset in range(0, len(raw_bytes), 16):
        chunk = raw_bytes[offset : offset + 16]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        hex_padded = f"{hex_str:<48}"
        ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        hex_dump_lines.append({
            "address": f"{offset:08x}",
            "hex": hex_padded,
            "ascii": ascii_str,
        })

    best_score = float(winner["score"]) if (winner and "score" in winner) else 0.0
    fec_summary = (
        f"DEPTH {winner['depth']} (score: {best_score:.4f})"
        if fec_detected
        else f"FEC: not detected (best score {best_score:.4f})"
    )

    res = {
        "file_id": file_id,
        "supported": True,
        "winner": winner_payload,
        "fec_detected": fec_detected,
        "fec_summary": fec_summary,
        "candidate_scores": [
            {
                "depth": c["depth"],
                "fec_rate": c["fec_rate"],
                "fec_constraint_length": c["fec_constraint_length"],
                "score": round(float(c["score"]), 4),
                "polarity": c["polarity"],
            }
            for c in all_candidates
        ],
        "sync_found": sync_found,
        "sync_position": sync_pos,
        "payload_ascii": recovered_text,
        "payload_hex": hex_dump_lines,
        "payload_bytes_count": n_bytes,
        "dump_label": dump_label,
        "dump_is_raw": dump_is_raw,
        "truncated": truncated,
        "logs": logs,
    }

    # Store per-file pipeline results server-side so Analysis tab reflects it
    entry["bitstream_result"] = res
    return res


@app.get("/api/ingest/recent")
def get_recent_operations():
    """Returns recently ingested files in this server session."""
    return RECENT_OPERATIONS
