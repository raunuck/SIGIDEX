"""
test_coding.py
--------------
Tests for Week 5 de-interleaving and FEC decoding (src/coding/).

End-to-end validation:
  1. Load the FEC+interleaved synthetic test signal.
  2. Demodulate with demodulate_bpsk (reusing Week 4).
  3. Run search_and_validate to blindly identify the correct interleaver
     depth and FEC config.
  4. Assert the winner matches ground-truth metadata and that the decoded
     bits match the original information bits.
"""

import os
import sys
import numpy as np
import pytest

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingestion.wav_loader import load_wav
from demod.psk import demodulate_bpsk
from coding.deinterleave import block_deinterleave
from coding.fec import viterbi_decode, decode_quality_score, _conv_encode
from coding.search import search_and_validate

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic")


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fec_interleaved_signal():
    """Load the FEC+interleaved BPSK test signal and its ground-truth metadata."""
    wav_path = os.path.join(DATA_DIR, "test_bpsk_fec_interleaved.wav")
    meta_path = os.path.join(DATA_DIR, "test_bpsk_fec_interleaved_meta.npz")
    if not os.path.isfile(wav_path):
        pytest.skip("test_bpsk_fec_interleaved.wav not found")
    if not os.path.isfile(meta_path):
        pytest.skip("test_bpsk_fec_interleaved_meta.npz not found")

    iq, fs = load_wav(wav_path)
    meta = np.load(meta_path)
    return iq, fs, meta


@pytest.fixture(scope="module")
def demodulated_bits(fec_interleaved_signal):
    """Demodulate the FEC+interleaved test signal using the Week 4 pipeline."""
    iq, fs, meta = fec_interleaved_signal
    bits = demodulate_bpsk(iq, fs)
    return bits


# ── End-to-End Search-and-Validate Test ──────────────────────────────────────

def test_search_finds_correct_depth(fec_interleaved_signal, demodulated_bits):
    """
    search_and_validate should correctly identify the true interleaver depth
    via scoring, without being told the answer.
    """
    _, _, meta = fec_interleaved_signal
    true_depth = int(meta["interleaver_depth"])
    true_k = int(meta["constraint_length"])
    info_bits = meta["info_bits"]

    result = search_and_validate(demodulated_bits)

    # Print full candidate score table for diagnostics
    print("\n" + "=" * 70)
    print("  search_and_validate -- Candidate Score Table")
    print("=" * 70)
    for i, c in enumerate(result["all_candidates"]):
        marker = " <-- WINNER" if i == 0 else ""
        print(
            f"  depth={c['depth']:3d}  rate={c['fec_rate']}  "
            f"K={c['fec_constraint_length']}  pol={c['polarity']:8s}  "
            f"score={c['score']:.4f}{marker}"
        )
    print(f"  Confidence margin: {result['confidence_margin']:.4f}")
    print("=" * 70)

    winner = result["winner"]
    assert winner is not None, "search_and_validate returned no winner"

    # (a) Correct depth identified
    assert winner["depth"] == true_depth, (
        f"Expected depth={true_depth}, got depth={winner['depth']} "
        f"(score={winner['score']:.4f})"
    )

    # (b) Correct constraint length
    assert winner["fec_constraint_length"] == true_k

    # (c) High score (near 1.0)
    assert winner["score"] > 0.95, f"Winning score too low: {winner['score']:.4f}"


def test_decoded_bits_match_ground_truth(fec_interleaved_signal, demodulated_bits):
    """
    After search_and_validate identifies the correct configuration, the
    decoded bits should match the original uncoded information bits with
    0% or near-0% BER.
    """
    _, _, meta = fec_interleaved_signal
    info_bits = meta["info_bits"]

    result = search_and_validate(demodulated_bits)
    winner = result["winner"]
    decoded = winner["decoded_bits"]

    # Compare decoded to ground truth, accounting for possible length difference
    # (Viterbi may produce a few extra tail bits)
    n = min(len(decoded), len(info_bits))
    assert n > 0, "No decoded bits produced"

    errors = int(np.sum(decoded[:n] != info_bits[:n]))
    ber = errors / n

    print("\n" + "=" * 60)
    print("  Decoded Bits vs Ground Truth")
    print("=" * 60)
    print(f"  Decoded bits count    : {len(decoded)}")
    print(f"  Ground-truth count    : {len(info_bits)}")
    print(f"  Compared bits         : {n}")
    print(f"  Bit errors            : {errors}")
    print(f"  Bit Error Rate (BER)  : {ber * 100:.2f}%")
    print(f"  Polarity              : {winner['polarity']}")
    print("=" * 60)

    assert ber <= 0.02, f"BER too high: {ber * 100:.2f}% ({errors}/{n})"


def test_confidence_margin_is_meaningful(fec_interleaved_signal, demodulated_bits):
    """
    The winning candidate should have a meaningfully higher score than
    wrong-depth candidates (not just barely winning).
    """
    _, _, meta = fec_interleaved_signal
    true_depth = int(meta["interleaver_depth"])

    result = search_and_validate(demodulated_bits)

    # Find the best score among wrong-depth candidates
    wrong_depth_scores = [
        c["score"]
        for c in result["all_candidates"]
        if c["depth"] != true_depth
    ]

    if wrong_depth_scores:
        best_wrong = max(wrong_depth_scores)
        winning_score = result["winner"]["score"]
        margin = winning_score - best_wrong

        print(f"\n  Winning score: {winning_score:.4f}")
        print(f"  Best wrong-depth score: {best_wrong:.4f}")
        print(f"  Margin: {margin:.4f}")

        # The correct depth should score significantly higher than any
        # wrong depth (expect > 0.1 gap on clean signals)
        assert margin > 0.05, (
            f"Margin between correct and wrong depth too small: {margin:.4f}"
        )


# ── Unit Tests: Block De-interleaver ─────────────────────────────────────────

def test_block_deinterleave_round_trip():
    """
    Interleave then de-interleave should recover the original bits
    (when length is an exact multiple of depth).
    """
    rng = np.random.default_rng(42)
    original = rng.integers(0, 2, size=120)  # 120 = 8 * 15, exact multiple

    # Interleave: write row-by-row, read column-by-column
    depth = 8
    cols = len(original) // depth
    matrix = original.reshape(depth, cols)
    interleaved = matrix.T.flatten()

    # De-interleave should recover the original
    recovered = block_deinterleave(interleaved, depth)
    np.testing.assert_array_equal(recovered[:len(original)], original)


def test_block_deinterleave_non_divisible():
    """
    When the bit count is not a multiple of depth, zero-padding should
    still allow correct round-trip for the original bits.
    """
    rng = np.random.default_rng(42)
    original = rng.integers(0, 2, size=103)  # Not divisible by 8
    depth = 8
    cols = int(np.ceil(len(original) / depth))  # 13
    total = depth * cols  # 104

    # Interleave with padding
    padded = np.zeros(total, dtype=original.dtype)
    padded[:len(original)] = original
    matrix = padded.reshape(depth, cols)
    interleaved = matrix.T.flatten()

    # De-interleave
    recovered = block_deinterleave(interleaved, depth)
    np.testing.assert_array_equal(recovered[:len(original)], original)


def test_block_deinterleave_invalid_depth():
    """De-interleaver should reject non-positive depth."""
    with pytest.raises(ValueError):
        block_deinterleave(np.array([1, 0, 1]), depth=0)
    with pytest.raises(ValueError):
        block_deinterleave(np.array([1, 0, 1]), depth=-1)


def test_block_deinterleave_empty():
    """De-interleaver should handle empty input."""
    result = block_deinterleave(np.array([]), depth=8)
    assert len(result) == 0


# ── Unit Tests: Viterbi Decode ───────────────────────────────────────────────

def test_viterbi_decode_clean_round_trip():
    """
    Encode then decode with the correct FEC config should recover the
    original bits with 0% BER.
    """
    rng = np.random.default_rng(42)
    info_bits = rng.integers(0, 2, size=200)

    coded = _conv_encode(info_bits, constraint_length=7, rate="1/2")
    decoded = viterbi_decode(coded, constraint_length=7, rate="1/2")

    n = min(len(decoded), len(info_bits))
    ber = np.mean(decoded[:n] != info_bits[:n])
    assert ber == 0.0, f"Clean encode/decode BER should be 0%, got {ber * 100:.2f}%"


def test_viterbi_decode_empty():
    """Viterbi decoder should handle empty input."""
    result = viterbi_decode(np.array([]), constraint_length=7, rate="1/2")
    assert len(result) == 0


def test_viterbi_decode_invalid_config():
    """Viterbi decoder should raise on unknown FEC config."""
    with pytest.raises(ValueError):
        viterbi_decode(np.array([1, 0, 1, 0]), constraint_length=99, rate="1/2")


# ── Unit Tests: Decode Quality Score ─────────────────────────────────────────

def test_quality_score_perfect():
    """
    When decode + re-encode reproduces the original coded bits, the
    quality score should be 1.0.
    """
    rng = np.random.default_rng(42)
    info_bits = rng.integers(0, 2, size=200)

    coded = _conv_encode(info_bits, constraint_length=7, rate="1/2")
    decoded = viterbi_decode(coded, constraint_length=7, rate="1/2")

    score = decode_quality_score(coded, decoded, constraint_length=7, rate="1/2")
    assert score > 0.99, f"Perfect decode should score ~1.0, got {score:.4f}"


def test_quality_score_random():
    """
    Random bits fed as 'decoded' should produce a score around 0.5.
    """
    rng = np.random.default_rng(42)
    coded = rng.integers(0, 2, size=400)
    random_decoded = rng.integers(0, 2, size=200)

    score = decode_quality_score(coded, random_decoded, constraint_length=7, rate="1/2")
    assert 0.3 < score < 0.7, f"Random decode should score ~0.5, got {score:.4f}"


# ── Unit Tests: Search-and-Validate ──────────────────────────────────────────

def test_search_with_known_bits():
    """
    Run search_and_validate on perfectly known encoded+interleaved bits
    (no demodulation noise), confirm it finds the right depth.
    """
    rng = np.random.default_rng(99)
    info_bits = rng.integers(0, 2, size=200)

    coded = _conv_encode(info_bits, constraint_length=7, rate="1/2")

    # Block interleave with depth=8
    depth = 8
    n = len(coded)
    cols = int(np.ceil(n / depth))
    total = depth * cols
    padded = np.zeros(total, dtype=coded.dtype)
    padded[:n] = coded
    matrix = padded.reshape(depth, cols)
    interleaved = matrix.T.flatten()

    result = search_and_validate(interleaved)
    assert result["winner"]["depth"] == depth
    assert result["winner"]["score"] > 0.95

    decoded = result["winner"]["decoded_bits"]
    m = min(len(decoded), len(info_bits))
    ber = np.mean(decoded[:m] != info_bits[:m])
    assert ber <= 0.02
