import React, { useEffect, useState } from 'react';
import Card from '../components/Card';
import Pill from '../components/Pill';
import PsdPlotlyChart from '../components/PsdPlotlyChart';
import SpectrogramPlotly from '../components/SpectrogramPlotly';
import { HelpCircle } from 'lucide-react';

export default function AnalysisTab({ fileId, fecData, onAnalysisLoaded }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (fileId) {
      fetchAnalysis(fileId);
    }
  }, [fileId]);

  const fetchAnalysis = async (id) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`/api/analysis/${id}`);
      if (!res.ok) throw new Error('Failed to load analysis data');
      const data = await res.json();
      setAnalysis(data);
      if (onAnalysisLoaded) {
        onAnalysisLoaded(data);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const rf = analysis?.report?.rf || {};
  const mod = analysis?.report?.modulation || {};
  const psd = analysis?.psd || { freqs_khz: [], psd_db: [] };
  const spectrogram = analysis?.spectrogram || null;
  const peakKhz = analysis?.peak_freq_khz || 0.0;
  const peakPower = analysis?.peak_power_db || 0.0;
  const confidencePct = Math.round((mod.confidence || 0) * 100);
  const isUnknownMod = !mod.type || mod.type.toLowerCase() === 'unknown';

  // Real STFT parameters from backend spectrogram
  const nperseg = spectrogram?.nperseg || 256;
  const hopSamples = spectrogram?.hop_samples || 128;
  const sampleRate = analysis?.sample_rate || 44100;
  const hopMs = ((hopSamples / sampleRate) * 1000).toFixed(1);

  // Dual-display SNR calculation
  const inBandSnr = rf.snr_db;
  const isNoisySignal = inBandSnr !== undefined && inBandSnr !== null && inBandSnr < 15.0;
  let fullBandDisplay = '-- dB';
  if (inBandSnr !== undefined && inBandSnr !== null) {
    if (isNoisySignal) {
      const fullBandVal =
        rf.snr_full_band_db !== undefined && rf.snr_full_band_db !== null
          ? rf.snr_full_band_db
          : inBandSnr - 10 * Math.log10((rf.sampling_rate_hz || sampleRate || 44100) / Math.max(rf.bandwidth_hz || 1, 1));
      fullBandDisplay = `${fullBandVal.toFixed(1)} dB`;
    } else {
      fullBandDisplay = 'high/clean, exact value not meaningful';
    }
  }

  // FEC information: use passed fecData or server-cached analysis.fec
  const activeFec = fecData || analysis?.fec;
  const fecRan = !!activeFec;
  const fecDetected = activeFec?.fec_detected || false;
  const fecWinner = activeFec?.winner;
  const fecEncoding = fecRan
    ? fecDetected
      ? 'Convolutional Viterbi'
      : 'not detected'
    : 'not run yet';
  const fecRate = fecRan
    ? fecDetected && fecWinner
      ? `r = ${fecWinner.fec_rate} (K=${fecWinner.fec_constraint_length})`
      : 'not detected'
    : 'not run yet';
  const fecBer =
    fecRan && fecDetected && fecWinner?.score !== undefined
      ? `${((1 - fecWinner.score) * 100).toFixed(2)}%`
      : null;

  return (
    <div className="max-w-7xl mx-auto py-6 px-4 grid grid-cols-1 lg:grid-cols-12 gap-5">
      {/* ── Left Column: Interactive PSD & Spectrogram (7 cols) ── */}
      <div className="lg:col-span-7 flex flex-col space-y-5">
        {/* 1. PSD Card */}
        <Card
          title="Power Spectral Density (PSD)"
          glyph="📈"
          statusText={
            analysis
              ? `SPAN: ${(sampleRate / 1000).toFixed(1)} kHz (${analysis.spectrum_mode || 'one-sided'}) | CF: ${peakKhz.toFixed(2)} kHz`
              : 'SPAN: -- | CF: --'
          }
        >
          {loading ? (
            <div className="h-64 flex items-center justify-center font-mono text-xs text-[#8E8E93]">
              COMPUTING WELCH PSD...
            </div>
          ) : error ? (
            <div className="h-64 flex items-center justify-center font-mono text-xs text-[#E85D75]">
              ERROR: {error}
            </div>
          ) : (
            <PsdPlotlyChart
              freqs={psd.freqs_khz}
              psd={psd.psd_db}
              peakFreqKhz={peakKhz}
              peakPowerDb={peakPower}
              spectrumMode={analysis?.spectrum_mode || 'one-sided'}
              yMinClip={psd.y_min}
              yMaxClip={psd.y_max}
            />
          )}
        </Card>

        {/* 2. Spectrogram Card (Renamed from Waterfall, real STFT window & hop) */}
        <Card
          title="SPECTROGRAM"
          glyph="::"
          statusText={
            spectrogram
              ? `NFFT: ${nperseg} | HOP: ${hopSamples} (${hopMs} ms)`
              : 'NFFT: -- | HOP: --'
          }
        >
          {loading ? (
            <div className="h-64 flex items-center justify-center font-mono text-xs text-[#8E8E93]">
              COMPUTING STFT SPECTROGRAM MATRIX...
            </div>
          ) : (
            <SpectrogramPlotly spectrogram={spectrogram} />
          )}
        </Card>
      </div>

      {/* ── Right Column: Channel Diagnostics, Modulation, FEC (5 cols) ── */}
      <div className="lg:col-span-5 flex flex-col space-y-5">
        {/* 1. Channel Diagnostics Card */}
        <Card title="Channel Diagnostics" glyph="📡">
          <div className="space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[#2A2A2E]/40">
              <span className="text-xs font-mono font-bold text-[#8E8E93] uppercase">
                Center Freq
              </span>
              <span className="text-base font-mono font-extrabold text-white">
                {rf.center_frequency_hz !== undefined
                  ? `${(rf.center_frequency_hz / 1e3).toFixed(5)} kHz`
                  : `${peakKhz.toFixed(5)} kHz`}
              </span>
            </div>

            <div className="flex items-center justify-between pb-3 border-b border-[#2A2A2E]/40">
              <span className="text-xs font-mono font-bold text-[#8E8E93] uppercase">
                Bandwidth
              </span>
              <span className="text-base font-mono font-extrabold text-white">
                {rf.bandwidth_hz !== undefined
                  ? `${(rf.bandwidth_hz / 1e3).toFixed(2)} kHz`
                  : '-- kHz'}
              </span>
            </div>

            {/* Primary: SNR (IN-BAND) */}
            <div className="flex items-center justify-between pb-3 border-b border-[#2A2A2E]/40">
              <div className="flex items-center space-x-1.5 group cursor-help">
                <span
                  className="text-xs font-mono font-bold text-[#8E8E93] uppercase underline decoration-dotted decoration-[#8E8E93]/60"
                  title="Signal vs noise floor within occupied bandwidth; peak spectral clearance."
                >
                  SNR (IN-BAND)
                </span>
                <HelpCircle
                  className="w-3.5 h-3.5 text-[#8E8E93]/60 group-hover:text-[#F5A623] transition-colors"
                  title="Signal vs noise floor within occupied bandwidth; peak spectral clearance."
                />
              </div>

              <div className="flex items-center space-x-2">
                <span className="text-base font-mono font-extrabold text-[#F5A623]">
                  {rf.snr_db !== undefined && rf.snr_db !== null ? `${rf.snr_db.toFixed(1)} dB` : '-- dB'}
                </span>
                <span className="w-2 h-2 rounded-full bg-[#F5A623]" />
              </div>
            </div>

            {/* Secondary: SNR (EST. FULL-BAND) */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-1.5 group cursor-help shrink-0">
                <span
                  className="text-xs font-mono font-bold text-[#8E8E93] uppercase underline decoration-dotted decoration-[#8E8E93]/60"
                  title="Estimated total signal-to-noise ratio across entire sampling bandwidth: SNR(in-band) - 10*log10(fs/B). Meaningful only below ~15 dB in-band SNR (i.e. for signals that are actually noisy). Above ~15 dB in-band SNR, unshaped pulse sidelobes mask the noise floor (-9 to -12 dB error at 20 dB SNR, while accurate to ~1 dB at 0 dB SNR), so it reads as 'high/clean, exact value not meaningful' rather than a misleading precise number."
                >
                  SNR (EST. FULL-BAND)
                </span>
                <HelpCircle
                  className="w-3.5 h-3.5 text-[#8E8E93]/60 group-hover:text-[#F5A623] transition-colors"
                  title="Estimated total signal-to-noise ratio across entire sampling bandwidth: SNR(in-band) - 10*log10(fs/B). Meaningful only below ~15 dB in-band SNR (i.e. for signals that are actually noisy). Above ~15 dB in-band SNR, unshaped pulse sidelobes mask the noise floor (-9 to -12 dB error at 20 dB SNR, while accurate to ~1 dB at 0 dB SNR), so it reads as 'high/clean, exact value not meaningful' rather than a misleading precise number."
                />
              </div>

              <div className="flex items-center justify-end text-right ml-2">
                {isNoisySignal ? (
                  <span className="text-base font-mono font-extrabold text-[#8E8E93]">
                    {fullBandDisplay}
                  </span>
                ) : (
                  <span
                    className="text-[11px] font-mono font-medium text-[#8E8E93]/80 text-right leading-tight"
                    title="Exact value not meaningful above ~15 dB in-band SNR due to pulse sidelobes masking noise floor"
                  >
                    {fullBandDisplay}
                  </span>
                )}
              </div>
            </div>
          </div>
        </Card>

        {/* 2. Modulation Classifier Card */}
        <Card title="Modulation Classifier" glyph="◈">
          <div className="space-y-4">
            <div className="flex flex-col space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-[#8E8E93] uppercase">
                  Detected Scheme
                </span>
                <Pill
                  variant={isUnknownMod ? 'muted' : 'amber'}
                  className="text-xs px-3 py-1 font-bold"
                >
                  {mod.type ? mod.type.toUpperCase() : 'UNKNOWN'}
                </Pill>
              </div>

              {/* Show classifier's reason string under badge when type is unknown */}
              {isUnknownMod && mod.reason && (
                <div className="text-right text-[11px] font-mono text-[#E85D75] italic">
                  ↳ {mod.reason}
                </div>
              )}
            </div>

            <div className="space-y-2 pt-2 border-t border-[#2A2A2E]/40">
              <div className="flex items-center justify-between text-xs font-mono font-bold">
                <span className="text-[#8E8E93]">CLASSIFICATION CONFIDENCE</span>
                <span className="text-[#F5A623]">{confidencePct}%</span>
              </div>
              <div className="w-full bg-[#0A0A0B] border border-[#2A2A2E] rounded h-2 overflow-hidden">
                <div
                  className="bg-[#F5A623] h-full rounded transition-all duration-500"
                  style={{ width: `${confidencePct}%` }}
                />
              </div>
            </div>
          </div>
        </Card>

        {/* 3. Forward Error Correction (FEC) Card */}
        {/* Score >= 0.95 gate: hide BER below threshold, show Pre-FEC BER only when detected, remove NOMINAL pill */}
        <Card title="Forward Error Correction (FEC)" glyph="🔒">
          <div className="space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-[#2A2A2E]/40">
              <span className="text-xs font-mono font-bold text-[#8E8E93] uppercase">
                Encoding
              </span>
              <span
                className={`text-sm font-mono font-bold ${
                  fecEncoding === 'Convolutional Viterbi'
                    ? 'text-white'
                    : 'text-[#8E8E93]'
                }`}
              >
                {fecEncoding}
              </span>
            </div>

            <div className="flex items-center justify-between pb-3 border-b border-[#2A2A2E]/40">
              <span className="text-xs font-mono font-bold text-[#8E8E93] uppercase">
                Code Rate
              </span>
              <span
                className={`text-sm font-mono font-bold ${
                  fecRate.startsWith('r =') ? 'text-white' : 'text-[#8E8E93]'
                }`}
              >
                {fecRate}
              </span>
            </div>

            {/* Show Pre-FEC BER ONLY when FEC is detected */}
            {fecBer ? (
              <div className="flex items-center justify-between">
                <span className="text-xs font-mono font-bold text-[#8E8E93] uppercase">
                  Pre-FEC BER (Bit Error Rate)
                </span>
                <span className="text-sm font-mono font-bold text-[#F5A623]">
                  {fecBer}
                </span>
              </div>
            ) : (
              <div className="text-xs font-mono text-[#8E8E93] italic">
                {!fecRan
                  ? 'Decode pipeline not run yet (switch to Payload Decoder tab)'
                  : activeFec?.fec_summary || 'FEC not detected (BER hidden)'}
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
}
