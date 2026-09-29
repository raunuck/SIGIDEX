"""
test_api.py
-----------
Tests for the FastAPI backend (backend/main.py):
  - Ingestion endpoint (file upload and sample loading)
  - Analysis endpoint (PSD, Peak, JSON Spectrogram Matrix, Bucket 1 Report)
  - Constellation endpoint (Symbols, EVM RMS, Carrier Lock, N/A offset)
  - Bitstream endpoint (Score >= 0.95 gate, Sync Word, Recovered Payload ASCII, Hex Dump)
  - Non-BPSK Guard (FSK returns unsupported)
  - Recents endpoint
"""

import os
import sys
import pytest
from starlette.testclient import TestClient

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(root_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.main import app

client = TestClient(app)
DATA_DIR = os.path.join(root_dir, "data", "synthetic")


def test_health_check():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def _ingest_framed_file():
    wav_path = os.path.join(DATA_DIR, "test_framed_message.wav")
    if not os.path.isfile(wav_path):
        pytest.skip("test_framed_message.wav not found")

    with open(wav_path, "rb") as f:
        res = client.post("/api/ingest", files={"file": ("test_framed_message.wav", f, "audio/wav")})

    assert res.status_code == 200
    data = res.json()
    assert "file_id" in data
    return data["file_id"]


def test_ingest_sample_file():
    fid = _ingest_framed_file()
    assert fid is not None


def test_full_api_workflow_framed():
    file_id = _ingest_framed_file()

    # 1. Analysis endpoint (interactive PSD + JSON spectrogram matrix)
    res_analysis = client.get(f"/api/analysis/{file_id}")
    assert res_analysis.status_code == 200
    analysis_data = res_analysis.json()
    assert "report" in analysis_data
    assert analysis_data["report"]["modulation"]["type"].lower() == "psk"
    assert "psd" in analysis_data
    assert len(analysis_data["psd"]["freqs_khz"]) > 0
    assert analysis_data["spectrum_mode"] == "one-sided"
    # Spectrogram matrix returned as JSON (not PNG)
    assert "spectrogram" in analysis_data
    spec = analysis_data["spectrogram"]
    assert "z_db" in spec
    assert "f_khz" in spec
    assert "t_sec" in spec
    assert spec["nperseg"] == 256
    assert spec["hop_samples"] == 128
    assert len(spec["z_db"]) > 0

    # 2. Constellation endpoint (EVM RMS, CARRIER LOCK, N/A offset)
    res_const = client.get(f"/api/constellation/{file_id}")
    assert res_const.status_code == 200
    const_data = res_const.json()
    assert const_data["supported"] is True
    assert len(const_data["points"]) > 0
    assert 0.0 < const_data["evm_rms"] < 10.0
    assert const_data["carrier_locked"] is True
    assert const_data["offset_hz"] is None  # N/A unless reference metadata provided

    # 3. Bitstream endpoint (Score >= 0.95 gate -> depth 8, score ~1.0, sync 0, 'HELLO SIH')
    res_bit = client.get(f"/api/bitstream/{file_id}")
    assert res_bit.status_code == 200
    bit_data = res_bit.json()
    assert bit_data["supported"] is True
    assert bit_data["fec_detected"] is True
    assert bit_data["winner"]["depth"] == 8
    assert bit_data["winner"]["score"] >= 0.95
    assert bit_data["sync_found"] is True
    assert bit_data["sync_position"] == 0
    assert bit_data["payload_ascii"] == "HELLO SIH"
    assert len(bit_data["payload_hex"]) > 0
    assert len(bit_data["logs"]) > 0

    # 4. Recent operations endpoint
    res_recent = client.get("/api/ingest/recent")
    assert res_recent.status_code == 200
    assert len(res_recent.json()) > 0


def test_bpsk_no_fec_workflow():
    """Verify test_bpsk.wav: expect FEC not detected (score < 0.95) and raw demodulated bits dump."""
    res = client.post("/api/ingest/sample/test_bpsk.wav")
    assert res.status_code == 200
    file_id = res.json()["file_id"]

    res_bit = client.get(f"/api/bitstream/{file_id}")
    assert res_bit.status_code == 200
    bit_data = res_bit.json()
    assert bit_data["supported"] is True
    assert bit_data["fec_detected"] is False
    assert "FEC: not detected" in bit_data["fec_summary"]
    assert bit_data["winner"] is None
    assert bit_data["dump_label"] == "RAW DEMODULATED BITS (FEC not resolved)"
    assert bit_data["dump_is_raw"] is True
    assert len(bit_data["payload_hex"]) > 0


def test_fsk_unsupported_guard():
    """Verify test_fsk.wav: FSK classified, demod unsupported result returned with notice."""
    res = client.post("/api/ingest/sample/test_fsk.wav")
    assert res.status_code == 200
    file_id = res.json()["file_id"]

    # Constellation endpoint
    res_const = client.get(f"/api/constellation/{file_id}")
    assert res_const.status_code == 200
    const_data = res_const.json()
    assert const_data["supported"] is False
    assert "FSK" in const_data["message"]
    assert const_data["symbol_rate"] is None
    assert const_data["evm_rms"] is None

    # Bitstream endpoint
    res_bit = client.get(f"/api/bitstream/{file_id}")
    assert res_bit.status_code == 200
    bit_data = res_bit.json()
    assert bit_data["supported"] is False
    assert "FSK" in bit_data["message"]
    assert bit_data["fec_detected"] is False
