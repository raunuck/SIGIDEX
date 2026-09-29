"""
make_test_framed_message.py
---------------------------
Generate a synthetic BPSK signal transmitting a framed ASCII message:

FRAME FORMAT SPECIFICATION
==========================
  +--------------------+--------------------+-------------------------------+
  | Sync Word (16 bit) |   Header (16 bit)  |       Payload (N * 8 bit)     |
  +--------------------+--------------------+-------------------------------+
  |  0xEB90            | Payload byte count | ASCII text, 8-bit MSB-first   |
  |  1110 1011 1001 0000| Unsigned big-endian| e.g. "HELLO SIH" (9 bytes)    |
  +--------------------+--------------------+-------------------------------+

1. Sync Word (16 bits):
   Hex: 0xEB90
   Binary: [1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]
   Autocorrelation: peak = 16, maximum sidelobe = 4 (peak-to-sidelobe ratio = 4.0).
   Widely used in aerospace/telemetry framing.

2. Header (16 bits):
   Unsigned 16-bit big-endian integer indicating the payload length in bytes.
   For "HELLO SIH" (9 bytes): 0x0009 -> [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,1].

3. Payload (N * 8 bits):
   ASCII characters encoded as 8 bits per byte, MSB-first.
   For "HELLO SIH" (9 chars): 72 bits.

Total Frame Length: 16 + 16 + 72 = 104 bits.

TRANSMISSION CHAIN
==================
  Frame Bits (104 bits)
    --> FEC Encoding: Rate 1/2, K=7 convolutional code (NASA standard, octal 171, 133)
        Termination: 'term' -> 2 * (104 + 6) = 220 coded bits
    --> Block Interleaving: Depth = 8, padded to next multiple of 8
        220 bits -> padded to 224 bits -> written row-by-row, read col-by-col
    --> BPSK Modulation:
        Carrier = 5000 Hz, Symbol Rate = 500 sym/s, Sample Rate = 44100 Hz
    --> WAV file output: data/synthetic/test_framed_message.wav
    --> Companion ground-truth metadata: data/synthetic/test_framed_message_meta.npz
"""

import os
import numpy as np
import soundfile as sf
import commpy.channelcoding.convcode as cc


# ── Frame format parameters ──────────────────────────────────────────────────
MESSAGE_TEXT    = "HELLO SIH"
SYNC_WORD_HEX   = 0xEB90
SYNC_WORD_BITS  = np.array([(SYNC_WORD_HEX >> (15 - i)) & 1 for i in range(16)], dtype=int)
HEADER_WIDTH    = 16            # bits
HEADER_ENCODING = "uint16_payload_bytes"

# ── Transmission parameters (reused from Weeks 4 & 5) ────────────────────────
SAMPLE_RATE       = 44_100      # Hz
CARRIER_FREQ      = 5_000       # Hz
SYMBOL_RATE       = 500         # symbols/sec
INTERLEAVER_DEPTH = 8           # block interleaver depth (rows)

# FEC: rate 1/2, K=7 convolutional code (NASA/CCSDS standard)
CONSTRAINT_LENGTH = 7
FEC_RATE          = "1/2"
MEMORY            = np.array([CONSTRAINT_LENGTH - 1])       # [6]
G_MATRIX          = np.array([[0o171, 0o133]])               # octal generators


def ascii_to_bits(text: str) -> np.ndarray:
    """Convert an ASCII string to a 1D numpy array of bits (8 bits/byte, MSB first)."""
    raw_bytes = text.encode("ascii")
    return np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).astype(int)


def encode_header_bytes(byte_count: int, width: int = 16) -> np.ndarray:
    """Encode an integer byte count into an unsigned big-endian bit array."""
    return np.array([(byte_count >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)


def block_interleave(bits: np.ndarray, depth: int) -> np.ndarray:
    """Standard block interleaver: write row-by-row, read column-by-column."""
    n = len(bits)
    cols = int(np.ceil(n / depth))
    total = depth * cols
    padded = np.zeros(total, dtype=bits.dtype)
    padded[:n] = bits
    matrix = padded.reshape(depth, cols)
    return matrix.T.flatten()


def main():
    # ── 1. Construct Frame ───────────────────────────────────────────────────
    payload_bits = ascii_to_bits(MESSAGE_TEXT)
    payload_byte_count = len(payload_bits) // 8
    header_bits = encode_header_bytes(payload_byte_count, HEADER_WIDTH)
    sync_bits = SYNC_WORD_BITS.copy()

    frame_bits = np.concatenate([sync_bits, header_bits, payload_bits])
    frame_len = len(frame_bits)

    # ── 2. FEC Encode (Rate 1/2, K=7 convolutional) ─────────────────────────
    trellis = cc.Trellis(MEMORY, G_MATRIX)
    coded_bits = cc.conv_encode(frame_bits, trellis, termination="term").astype(int)

    # ── 3. Block Interleave ──────────────────────────────────────────────────
    interleaved_bits = block_interleave(coded_bits, INTERLEAVER_DEPTH)

    # ── 4. BPSK Modulate ─────────────────────────────────────────────────────
    symbols = 2 * interleaved_bits - 1  # 0 -> -1,  1 -> +1
    samples_per_symbol = SAMPLE_RATE // SYMBOL_RATE
    baseband = np.repeat(symbols, samples_per_symbol).astype(np.float64)

    t = np.arange(len(baseband)) / SAMPLE_RATE
    carrier = np.cos(2 * np.pi * CARRIER_FREQ * t)
    signal = baseband * carrier
    signal /= np.max(np.abs(signal))

    # ── 5. Save Output Files ─────────────────────────────────────────────────
    out_dir = os.path.dirname(__file__)
    os.makedirs(out_dir, exist_ok=True)

    wav_path = os.path.join(out_dir, "test_framed_message.wav")
    sf.write(wav_path, signal, SAMPLE_RATE, subtype="FLOAT")

    meta_path = os.path.join(out_dir, "test_framed_message_meta.npz")
    np.savez(
        meta_path,
        message_str=MESSAGE_TEXT,
        sync_word=sync_bits,
        sync_word_hex=SYNC_WORD_HEX,
        sync_position=0,
        header_bits=header_bits,
        header_field_width=HEADER_WIDTH,
        header_encoding=HEADER_ENCODING,
        payload_byte_count=payload_byte_count,
        payload_bits=payload_bits,
        frame_bits=frame_bits,
        frame_len=frame_len,
        coded_bits=coded_bits,
        interleaved_bits=interleaved_bits,
        interleaver_depth=INTERLEAVER_DEPTH,
        constraint_length=CONSTRAINT_LENGTH,
        fec_rate_str=FEC_RATE,
        sample_rate=SAMPLE_RATE,
        carrier_freq=CARRIER_FREQ,
        symbol_rate=SYMBOL_RATE,
    )

    # ── 6. Print Ground-Truth Summary ────────────────────────────────────────
    duration = len(signal) / SAMPLE_RATE
    print("=" * 65)
    print("  Framed Message BPSK Test Signal -- Ground Truth")
    print("=" * 65)
    print(f"  WAV file              : {wav_path}")
    print(f"  Metadata file         : {meta_path}")
    print(f"  Original message      : '{MESSAGE_TEXT}'")
    print(f"  Sample rate           : {SAMPLE_RATE} Hz")
    print(f"  Carrier frequency     : {CARRIER_FREQ} Hz")
    print(f"  Symbol rate           : {SYMBOL_RATE} sym/s")
    print(f"  Duration              : {duration:.3f} s")
    print("  -- Frame Structure ------------------------------------")
    print(f"  Sync word (hex)       : 0x{SYNC_WORD_HEX:04X} ({len(sync_bits)} bits)")
    print(f"  Sync word bits        : {list(sync_bits)}")
    print(f"  Header width          : {HEADER_WIDTH} bits ({HEADER_ENCODING})")
    print(f"  Header bits           : {list(header_bits)} (value={payload_byte_count} bytes)")
    print(f"  Payload bits          : {len(payload_bits)} bits ({payload_byte_count} bytes)")
    print(f"  Total frame bits      : {frame_len} bits")
    print("  -- Coding & Modulation --------------------------------")
    print(f"  FEC                   : Rate {FEC_RATE}, K={CONSTRAINT_LENGTH} Convolutional")
    print(f"  Coded bits            : {len(coded_bits)}")
    print(f"  Interleaver depth     : {INTERLEAVER_DEPTH}")
    print(f"  Interleaved bits      : {len(interleaved_bits)}")
    print(f"  BPSK symbols          : {len(interleaved_bits)}")
    print(f"  Total samples         : {len(signal)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
