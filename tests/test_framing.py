"""
test_framing.py
---------------
Tests for Week 6 bitstream correlation, frame synchronization, and
header/payload extraction (src/correlation/framing.py).

Validation stages:
  1. Unit tests: find_sync_word, extract_frame, bits_to_ascii in isolation.
  2. Isolated framing test on known ground-truth bits (no demod/FEC noise).
  3. Full end-to-end pipeline test:
     load_wav -> demodulate_bpsk -> search_and_validate -> find_sync_word
     -> extract_frame -> bits_to_ascii -> assert exact message match.
  4. Robustness tests: noise/no-sync, offsets, bit errors, truncation, leftover bits.
"""

import os
import sys
import numpy as np
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.wav_loader import load_wav
from demod.psk import demodulate_bpsk
from coding.search import search_and_validate
from correlation.framing import (
    SYNC_WORD_16,
    SYNC_WORD_32,
    find_sync_word,
    extract_frame,
    bits_to_ascii,
    decode_header_bytes,
    decode_header_bits,
    AsciiDecoded,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def framed_signal():
    """Load the synthetic framed BPSK test signal and its companion metadata."""
    wav_path = os.path.join(DATA_DIR, "test_framed_message.wav")
    meta_path = os.path.join(DATA_DIR, "test_framed_message_meta.npz")
    if not os.path.isfile(wav_path):
        pytest.skip("test_framed_message.wav not found")
    if not os.path.isfile(meta_path):
        pytest.skip("test_framed_message_meta.npz not found")

    iq, fs = load_wav(wav_path)
    meta = np.load(meta_path)
    return iq, fs, meta


# ── 1. Isolated Ground-Truth Framing Tests ────────────────────────────────────

def test_framing_on_ground_truth_bits(framed_signal):
    """
    Test find_sync_word, extract_frame, and bits_to_ascii directly on the
    generator's saved ground-truth frame bits (isolation test without demod/FEC).
    """
    _, _, meta = framed_signal
    frame_bits = meta["frame_bits"]
    sync_word = meta["sync_word"]
    header_width = int(meta["header_field_width"])
    expected_message = str(meta["message_str"])

    # 1. Locate sync word
    matches = find_sync_word(frame_bits, sync_word, threshold=0.9)
    assert len(matches) >= 1, "Sync word not found in ground-truth frame bits"
    assert matches[0] == int(meta["sync_position"]), (
        f"Expected sync at {meta['sync_position']}, found at {matches[0]}"
    )

    # 2. Extract frame
    frame_data = extract_frame(
        bits=frame_bits,
        sync_position=matches[0],
        sync_word_length=len(sync_word),
        header_length=header_width,
        header_decoder=decode_header_bytes,
    )

    assert not frame_data["truncated"], "Frame was unexpectedly marked truncated"
    assert frame_data["expected_payload_bits"] == len(meta["payload_bits"])
    assert frame_data["actual_payload_bits"] == len(meta["payload_bits"])
    np.testing.assert_array_equal(frame_data["payload_bits"], meta["payload_bits"])

    # 3. Decode to ASCII
    recovered_text = bits_to_ascii(frame_data["payload_bits"])
    assert recovered_text == expected_message, (
        f"Decoded text '{recovered_text}' does not match expected '{expected_message}'"
    )


# ── 2. Full End-to-End Pipeline Test ─────────────────────────────────────────

def test_full_pipeline_end_to_end(framed_signal):
    """
    Full closed-loop pipeline test:
      1. load_wav: real WAV -> complex analytic IQ (Week 1)
      2. demodulate_bpsk: carrier recovery + timing recovery + bit decisions (Week 4)
      3. search_and_validate: blind de-interleave + Viterbi FEC decode (Week 5)
      4. find_sync_word: sliding correlation frame detection (Week 6)
      5. extract_frame: header decoding and payload separation (Week 6)
      6. bits_to_ascii: payload bits to human-readable string (Week 6)

    Asserts that the final recovered ASCII string matches the original message.
    """
    iq, fs, meta = framed_signal
    sync_word = meta["sync_word"]
    header_width = int(meta["header_field_width"])
    expected_message = str(meta["message_str"])
    expected_depth = int(meta["interleaver_depth"])

    # Step 1 & 2: Demodulate
    demod_bits = demodulate_bpsk(iq, fs)
    assert len(demod_bits) > 0, "Demodulation produced no bits"

    # Step 3: Blind de-interleaving and FEC decoding
    search_result = search_and_validate(demod_bits)
    winner = search_result["winner"]
    assert winner is not None, "search_and_validate produced no winner"
    assert winner["depth"] == expected_depth, (
        f"Blind search identified depth={winner['depth']}, expected {expected_depth}"
    )
    assert winner["score"] > 0.95, f"Winning decode score too low: {winner['score']:.4f}"

    decoded_bits = winner["decoded_bits"]

    # Step 4: Locate sync word
    sync_matches = find_sync_word(decoded_bits, sync_word, threshold=0.9)
    assert len(sync_matches) >= 1, (
        f"Sync word {list(sync_word)} not detected in decoded bitstream "
        f"(stream length {len(decoded_bits)})"
    )

    sync_pos = sync_matches[0]
    assert sync_pos == int(meta["sync_position"]), (
        f"Sync word found at position {sync_pos}, expected {meta['sync_position']}"
    )

    # Step 5: Extract frame
    frame_data = extract_frame(
        bits=decoded_bits,
        sync_position=sync_pos,
        sync_word_length=len(sync_word),
        header_length=header_width,
        header_decoder=decode_header_bytes,
    )

    assert not frame_data["truncated"], "Recovered frame is unexpectedly truncated"
    assert frame_data["actual_payload_bits"] == len(meta["payload_bits"])

    # Step 6: Decode payload to ASCII
    recovered_text = bits_to_ascii(frame_data["payload_bits"])

    print("\n" + "=" * 65)
    print("  Week 6 Full Pipeline Validation")
    print("=" * 65)
    print(f"  Expected message : '{expected_message}'")
    print(f"  Recovered text   : '{recovered_text}'")
    print(f"  Sync position    : {sync_pos}")
    print(f"  Interleaver depth: {winner['depth']} (score={winner['score']:.4f})")
    print(f"  Polarity         : {winner['polarity']}")
    print(f"  Payload bits     : {len(frame_data['payload_bits'])}")
    print("=" * 65)

    assert recovered_text == expected_message, (
        f"Recovered message '{recovered_text}' != expected '{expected_message}'"
    )


# ── 3. Unit Tests: find_sync_word ────────────────────────────────────────────

def test_find_sync_word_no_match_on_random_bits():
    """
    On pure random bits with no sync word present, find_sync_word should
    return an empty list (not a false match or crash).
    """
    rng = np.random.default_rng(12345)
    # Generate 1000 random bits
    random_bits = rng.integers(0, 2, size=1000)

    # Use SYNC_WORD_16 (0xEB90)
    matches = find_sync_word(random_bits, SYNC_WORD_16, threshold=0.9)
    assert matches == [], f"Expected no matches on random bits, got {matches}"


def test_find_sync_word_offset():
    """
    When a sync word is embedded at an arbitrary non-zero offset,
    find_sync_word should locate the exact start position.
    """
    rng = np.random.default_rng(42)
    prefix_len = 37
    prefix = rng.integers(0, 2, size=prefix_len)
    suffix = rng.integers(0, 2, size=100)

    stream = np.concatenate([prefix, SYNC_WORD_16, suffix])

    matches = find_sync_word(stream, SYNC_WORD_16, threshold=1.0)
    assert prefix_len in matches, f"Expected match at {prefix_len}, got {matches}"


def test_find_sync_word_tolerance_threshold():
    """
    Threshold controls error tolerance:
    - 16-bit sync word with 1 bit error = 15/16 = 0.9375 >= 0.90 -> matched
    - 16-bit sync word with 2 bit errors = 14/16 = 0.875 < 0.90 -> not matched
    """
    corrupted_1 = SYNC_WORD_16.copy()
    corrupted_1[0] = 1 - corrupted_1[0]  # 1 flip

    corrupted_2 = SYNC_WORD_16.copy()
    corrupted_2[0] = 1 - corrupted_2[0]
    corrupted_2[1] = 1 - corrupted_2[1]  # 2 flips

    # 1 flip matches at threshold 0.9
    assert find_sync_word(corrupted_1, SYNC_WORD_16, threshold=0.9) == [0]

    # 1 flip does NOT match at threshold 1.0 (exact match only)
    assert find_sync_word(corrupted_1, SYNC_WORD_16, threshold=1.0) == []

    # 2 flips do NOT match at threshold 0.9
    assert find_sync_word(corrupted_2, SYNC_WORD_16, threshold=0.9) == []


def test_find_sync_word_empty_or_short():
    """Short or empty inputs should return empty list without error."""
    assert find_sync_word(np.array([]), SYNC_WORD_16) == []
    assert find_sync_word(np.array([1, 0, 1]), SYNC_WORD_16) == []
    assert find_sync_word(SYNC_WORD_16, np.array([])) == []


# ── 4. Unit Tests: extract_frame & Truncation ─────────────────────────────────

def test_extract_frame_clean():
    """Normal frame extraction extracts the exact payload without truncation."""
    header = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=int)  # 2 bytes
    payload = np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0], dtype=int)  # 'AB' (16 bits)
    stream = np.concatenate([SYNC_WORD_16, header, payload, [1, 1, 1]])

    data = extract_frame(
        stream,
        sync_position=0,
        sync_word_length=16,
        header_length=16,
        header_decoder=decode_header_bytes,
    )

    assert not data["truncated"]
    assert data["header_value"] == 16  # 2 bytes = 16 bits
    assert data["actual_payload_bits"] == 16
    np.testing.assert_array_equal(data["payload_bits"], payload)


def test_extract_frame_truncated_payload():
    """When stream cuts off inside the payload, truncated=True and partial bits are returned."""
    header = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=int)  # 4 bytes = 32 bits
    partial_payload = np.array([1] * 12, dtype=int)  # Only 12 bits available instead of 32
    stream = np.concatenate([SYNC_WORD_16, header, partial_payload])

    data = extract_frame(
        stream,
        sync_position=0,
        sync_word_length=16,
        header_length=16,
        header_decoder=decode_header_bytes,
    )

    assert data["truncated"] is True
    assert data["expected_payload_bits"] == 32
    assert data["actual_payload_bits"] == 12
    assert len(data["payload_bits"]) == 12


def test_extract_frame_truncated_header():
    """When stream cuts off inside the header, truncated=True and empty payload returned."""
    truncated_header = np.array([0, 0, 0, 1], dtype=int)  # only 4 bits of 16-bit header
    stream = np.concatenate([SYNC_WORD_16, truncated_header])

    data = extract_frame(
        stream,
        sync_position=0,
        sync_word_length=16,
        header_length=16,
        header_decoder=decode_header_bytes,
    )

    assert data["truncated"] is True
    assert data["actual_payload_bits"] == 0
    assert len(data["payload_bits"]) == 0


# ── 5. Unit Tests: bits_to_ascii ─────────────────────────────────────────────

def test_bits_to_ascii_round_trip():
    """Standard ASCII conversion round trip."""
    msg = "TESTING 123 !?"
    raw = msg.encode("ascii")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))

    recovered = bits_to_ascii(bits)
    assert recovered == msg
    assert isinstance(recovered, str)
    assert not recovered.has_leftover
    assert recovered.leftover_count == 0
    assert recovered.warning is None


def test_bits_to_ascii_leftover_handling():
    """When bits are not a multiple of 8, leftover bits are dropped and warned."""
    msg = "OK"
    bits = np.unpackbits(np.frombuffer(msg.encode("ascii"), dtype=np.uint8))
    bits_with_leftover = np.append(bits, [1, 1, 0])  # 3 leftover bits

    with pytest.warns(UserWarning, match="dropped 3 leftover bit"):
        recovered = bits_to_ascii(bits_with_leftover)

    assert recovered == "OK"
    assert recovered.has_leftover is True
    assert recovered.leftover_count == 3
    assert "dropped 3 leftover bit" in recovered.warning

    # Also test tuple return mode
    text, warn = bits_to_ascii(bits_with_leftover, return_warning=True)
    assert text == "OK"
    assert "dropped 3 leftover bit" in warn


def test_bits_to_ascii_empty():
    """Empty bits array produces empty string without crashing."""
    recovered = bits_to_ascii(np.array([]))
    assert recovered == ""
    assert isinstance(recovered, str)
    assert not recovered.has_leftover
