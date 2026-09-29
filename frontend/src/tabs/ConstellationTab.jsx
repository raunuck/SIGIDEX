import React, { useEffect, useState } from 'react';
import Card from '../components/Card';
import Pill from '../components/Pill';
import ConstellationPlot from '../components/ConstellationPlot';
import { AlertTriangle } from 'lucide-react';

export default function ConstellationTab({ fileId, onConstellationLoaded }) {
  const [constData, setConstData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (fileId) {
      fetchConstellation(fileId);
    }
  }, [fileId]);

  const fetchConstellation = async (id) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/constellation/${id}`);
      if (!res.ok) throw new Error('Failed to load constellation data');
      const data = await res.json();
      setConstData(data);
      if (onConstellationLoaded) {
        onConstellationLoaded(data);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const isSupported = constData?.supported !== false;
  const modType = constData?.mod_type || 'FSK';
  const unsupportedMessage =
    constData?.message || `DEMOD: not available for ${modType} (BPSK only in this build)`;

  const points = constData?.points || [];
  const symbolRate = constData?.symbol_rate;
  const offsetHz = constData?.offset_hz; // null unless computed against real reference
  const evmRms = constData?.evm_rms;
  const carrierLocked = constData?.carrier_locked || false;

  return (
    <div className="max-w-7xl mx-auto py-6 px-4 flex flex-col space-y-6">
      {/* Unsupported Banner if classifier detected FSK / QAM */}
      {!isSupported && (
        <div className="bg-[#1A1208] border border-[#F5A623]/40 rounded-lg p-4 flex items-start space-x-3 text-xs font-mono">
          <AlertTriangle className="w-5 h-5 text-[#F5A623] shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="text-[#F5A623] font-bold tracking-wide uppercase">
              MODULATION UNSUPPORTED BY DEMODULATOR:
            </span>
            <p className="text-[#D1D1D6]">
              {unsupportedMessage}
            </p>
          </div>
        </div>
      )}

      {/* ── Top Row: Demodulator Signal Metrics Card ── */}
      <Card title="Demodulator Signal Metrics" glyph="⚡">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* 1. Symbol Rate */}
          <div className="bg-[#0A0A0B] border border-[#2A2A2E] rounded-lg p-4 flex flex-col">
            <span className="text-[11px] font-mono font-bold text-[#8E8E93] uppercase mb-1">
              Symbol Rate
            </span>
            <span className="text-xl sm:text-2xl font-mono font-black text-white">
              {isSupported && symbolRate != null
                ? `${symbolRate.toFixed(1)} Bd`
                : 'N/A'}
            </span>
          </div>

          {/* 2. Estimated Offset: only shown if computed against a real reference; otherwise N/A */}
          <div className="bg-[#0A0A0B] border border-[#2A2A2E] rounded-lg p-4 flex flex-col">
            <span className="text-[11px] font-mono font-bold text-[#8E8E93] uppercase mb-1">
              Estimated Offset
            </span>
            <span
              className={`text-xl sm:text-2xl font-mono font-black ${
                isSupported && offsetHz != null
                  ? 'text-[#F5A623]'
                  : 'text-[#8E8E93]'
              }`}
            >
              {isSupported && offsetHz != null
                ? `${offsetHz >= 0 ? '+' : ''}${offsetHz.toFixed(1)} Hz`
                : 'N/A'}
            </span>
          </div>

          {/* 3. EVM RMS: Show ONE EVM value computed as RMS distance; EVM Delta removed */}
          <div className="bg-[#0A0A0B] border border-[#2A2A2E] rounded-lg p-4 flex flex-col">
            <span className="text-[11px] font-mono font-bold text-[#8E8E93] uppercase mb-1">
              EVM RMS
            </span>
            <span
              className={`text-xl sm:text-2xl font-mono font-black ${
                isSupported && evmRms != null
                  ? 'text-[#F5A623]'
                  : 'text-[#8E8E93]'
              }`}
            >
              {isSupported && evmRms != null ? `${evmRms.toFixed(2)}%` : 'N/A'}
            </span>
          </div>
        </div>
      </Card>

      {/* ── Bottom Section: Description (Left) + Constellation Plot (Right) ── */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Explanatory & Lock Metrics Card */}
        <div className="lg:col-span-4 flex flex-col space-y-4">
          <Card title="I/Q Constellation" glyph="◈">
            <div className="space-y-4 text-xs text-[#8E8E93] leading-relaxed">
              <p>
                Visualizes amplitude and phase of demodulated symbols. Clean
                two-cluster convergence along the I-axis indicates strong carrier phase
                locking and accurate Gardner symbol timing recovery (BPSK).
              </p>

              <div className="pt-3 border-t border-[#2A2A2E]/50 space-y-3 font-mono">
                <div className="flex items-center justify-between">
                  <span>EVM (Normalized RMS):</span>
                  <span className="text-white font-bold">
                    {isSupported && evmRms != null
                      ? `${evmRms.toFixed(2)}% RMS`
                      : 'N/A'}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span>Recovered Symbols:</span>
                  <span className="text-white font-bold">
                    {isSupported ? constData?.symbol_count || 0 : 'N/A'}
                  </span>
                </div>

                {/* Derived from real carrier lock metric */}
                <div className="flex items-center justify-between">
                  <span>Costas Loop State:</span>
                  <span
                    className={`font-bold ${
                      isSupported && carrierLocked
                        ? 'text-[#F5A623]'
                        : isSupported
                        ? 'text-[#8E8E93]'
                        : 'text-[#E85D75]'
                    }`}
                  >
                    {!isSupported
                      ? 'UNSUPPORTED'
                      : carrierLocked
                      ? 'LOCKED'
                      : 'UNLOCKED'}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        </div>

        {/* Right: High-Resolution Constellation Plot Card */}
        <div className="lg:col-span-8">
          <Card
            title="Complex Baseband Constellation"
            glyph="☩"
            /* Renamed pill to reflect CARRIER LOCK derived from real metric */
            statusText={
              !isSupported
                ? 'CARRIER: UNSUPPORTED'
                : carrierLocked
                ? 'CARRIER LOCK: LOCKED'
                : 'CARRIER LOCK: UNLOCKED'
            }
            statusVariant={
              !isSupported ? 'muted' : carrierLocked ? 'amber' : 'muted'
            }
          >
            {loading ? (
              <div className="h-80 flex items-center justify-center font-mono text-xs text-[#8E8E93]">
                RECOVERING CARRIER &amp; SYMBOL TIMING...
              </div>
            ) : !isSupported ? (
              <div className="h-80 flex flex-col items-center justify-center p-6 text-center space-y-3 font-mono">
                <div className="text-[#E85D75] font-bold text-sm">
                  DEMOD: NOT AVAILABLE FOR {modType.toUpperCase()}
                </div>
                <div className="text-[#8E8E93] text-xs max-w-md leading-relaxed">
                  The receiver demodulator is strictly BPSK-only in this build.
                  Signals classified as FSK or QAM cannot be carrier-locked or mapped to BPSK constellation clusters.
                </div>
                <span className="text-[#F5A623] text-[11px] font-bold border border-[#F5A623]/30 bg-[#231B0D] px-3 py-1 rounded">
                  BPSK DEMODULATION PIPELINE SKIPPED
                </span>
              </div>
            ) : (
              <ConstellationPlot points={points} />
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
