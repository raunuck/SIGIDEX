import React, { useEffect, useState, useRef } from 'react';
import Card from '../components/Card';
import Pill from '../components/Pill';
import { AlertCircle } from 'lucide-react';

export default function BitstreamTab({ fileId, onBitstreamLoaded }) {
  const [bitstream, setBitstream] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const terminalEndRef = useRef(null);

  useEffect(() => {
    if (fileId) {
      fetchBitstream(fileId);
    }
  }, [fileId]);

  const fetchBitstream = async (id) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/bitstream/${id}`);
      if (!res.ok) throw new Error('Failed to load bitstream data');
      const data = await res.json();
      setBitstream(data);
      if (onBitstreamLoaded) {
        onBitstreamLoaded(data);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const isSupported = bitstream?.supported !== false;
  const logs = bitstream?.logs || [];
  const hexLines = bitstream?.payload_hex || [];
  const syncFound = bitstream?.sync_found || false;
  const fecDetected = bitstream?.fec_detected || false;
  const winner = bitstream?.winner;
  const recoveredText = bitstream?.payload_ascii || '';
  const dumpLabel = bitstream?.dump_label || 'HEX DUMP';
  const dumpIsRaw = bitstream?.dump_is_raw || false;
  const fecSummary = bitstream?.fec_summary || (fecDetected ? 'LOCKED' : 'NOT DETECTED');

  const getTagColor = (tag) => {
    switch (tag) {
      case 'SYNC_LOCK':
      case 'DONE':
        return 'text-[#F5A623]';
      case 'FEC_SEARCH':
        return fecDetected ? 'text-[#F5A623]' : 'text-[#8E8E93]';
      case 'PAYLOAD':
        return 'text-[#50E3C2]';
      case 'ERROR':
        return 'text-[#E85D75] font-extrabold';
      case 'WARN':
        return 'text-[#E85D75]';
      default:
        return 'text-white';
    }
  };

  return (
    <div className="max-w-7xl mx-auto py-6 px-4 flex flex-col space-y-6">
      {/* ── Top Row: Decoded ASCII Terminal (Left) + Hex Dump (Right) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-stretch">
        {/* 1. Decoded ASCII Terminal Card */}
        <Card
          title="Decoded ASCII Terminal"
          glyph=">_"
          statusText={
            !isSupported
              ? 'DEMOD UNSUPPORTED'
              : error
              ? 'DSP ERROR'
              : 'REALTIME DECODE'
          }
          statusVariant={!isSupported ? 'muted' : error ? 'pink' : 'amber'}
          className="min-h-[380px]"
        >
          <div className="bg-[#0A0A0B] border border-[#2A2A2E] rounded-md p-4 font-mono text-xs flex-1 overflow-y-auto max-h-[420px] space-y-2 select-text">
            {loading ? (
              <div className="text-[#8E8E93] animate-pulse">
                [..] EXECUTING DE-INTERLEAVE &amp; VITERBI PIPELINE...
              </div>
            ) : error ? (
              <div className="text-[#E85D75] font-bold">
                [ERR: PIPELINE]: {error}
              </div>
            ) : !fileId ? (
              /* AWAITING placeholder appears ONLY when no file is loaded yet */
              <div className="text-[#555558]">
                [00:00:00] AWAITING BASEBAND SIGNAL INGESTION...
              </div>
            ) : logs.length === 0 ? (
              <div className="text-[#8E8E93]">
                [00:00:00] PIPELINE EXECUTION COMPLETED - NO LOGS RECORDED
              </div>
            ) : (
              logs.map((item, idx) => (
                <div key={idx} className="leading-relaxed">
                  <span className="text-[#F5A623]">[{item.time}]</span>{' '}
                  <span className={`font-bold ${getTagColor(item.tag)}`}>
                    {item.tag}:
                  </span>{' '}
                  <span className={item.tag === 'ERROR' ? 'text-[#E85D75] font-semibold' : 'text-[#D1D1D6]'}>
                    {item.message}
                  </span>
                </div>
              ))
            )}
            <div ref={terminalEndRef} />
          </div>
        </Card>

        {/* 2. Hexadecimal Dump Card */}
        <Card
          title="Hexadecimal Dump"
          glyph="❖"
          statusText={dumpLabel}
          statusVariant={syncFound ? 'amber' : 'muted'}
          className="min-h-[380px]"
        >
          <div className="bg-[#0A0A0B] border border-[#2A2A2E] rounded-md p-4 font-mono text-xs flex-1 overflow-y-auto max-h-[420px] select-text">
            {dumpIsRaw && (
              <div className="mb-3 px-3 py-1.5 rounded bg-[#1C140C] border border-[#F5A623]/30 text-[#F5A623] flex items-center space-x-2 text-[11px]">
                <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                <span>
                  RAW DEMODULATED BITS — FEC not resolved or frame sync word not acquired.
                </span>
              </div>
            )}

            {hexLines.length === 0 ? (
              <div className="text-[#555558] py-8 text-center">
                {!isSupported
                  ? `DEMODULATION NOT AVAILABLE FOR ${bitstream?.mod_type || 'FSK'}`
                  : 'NO DEMODULATED BYTES AVAILABLE'}
              </div>
            ) : (
              <table className="w-full">
                <tbody>
                  {hexLines.map((row, idx) => (
                    <tr key={idx} className="hover:bg-[#141416] py-0.5">
                      {/* Address */}
                      <td className="text-[#E85D75] pr-4 select-none">
                        {row.address}
                      </td>
                      {/* Hex bytes */}
                      <td className="text-white pr-4 font-medium tracking-wide">
                        {row.hex}
                      </td>
                      {/* ASCII column */}
                      <td className="text-[#F5A623] font-bold tracking-widest border-l border-[#2A2A2E] pl-3">
                        {row.ascii}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Card>
      </div>

      {/* ── Bottom Status Bar ── */}
      <div className="min-h-12 bg-[#141416] border border-[#2A2A2E] rounded-lg px-6 py-2 flex flex-wrap items-center justify-between gap-4 text-xs font-mono select-none">
        <div className="flex flex-wrap items-center gap-6">
          {/* Frame Sync Lock */}
          <div className="flex items-center space-x-2">
            <span className="text-[#8E8E93] font-bold">FRAME SYNC:</span>
            <Pill variant={!isSupported ? 'muted' : syncFound ? 'solidAmber' : 'muted'}>
              {!isSupported ? 'N/A' : syncFound ? 'LOCKED' : 'NOT FOUND'}
            </Pill>
          </div>

          {/* Interleaver Depth / FEC Gate (score >= 0.95) */}
          <div className="flex items-center space-x-2">
            <span className="text-[#8E8E93] font-bold">INTERLEAVER:</span>
            <span
              className={`font-bold ${
                !isSupported
                  ? 'text-[#8E8E93]'
                  : fecDetected
                  ? 'text-[#F5A623]'
                  : 'text-[#8E8E93]'
              }`}
            >
              {!isSupported ? 'UNSUPPORTED' : fecSummary}
            </span>
          </div>

          {/* Recovered ASCII Payload summary */}
          {recoveredText ? (
            <div className="flex items-center space-x-2">
              <span className="text-[#8E8E93] font-bold">RECOVERED:</span>
              <span className="text-[#50E3C2] font-extrabold font-mono bg-[#0D231A] px-2 py-0.5 rounded border border-[#50E3C2]/40">
                "{recoveredText}"
              </span>
            </div>
          ) : (
            <div className="flex items-center space-x-2 text-[#8E8E93]">
              <span className="font-bold">PAYLOAD:</span>
              <span>{isSupported ? 'NONE (FRAME NOT RESOLVED)' : 'N/A'}</span>
            </div>
          )}
        </div>

        {/* Timestamp */}
        <div className="flex items-center space-x-2 text-[#8E8E93]">
          <span className="font-bold">STATUS:</span>
          <span className="text-white">
            {!isSupported ? 'UNSUPPORTED' : syncFound ? 'SYNC ACQUIRED' : 'RAW STREAM'}
          </span>
        </div>
      </div>
    </div>
  );
}
