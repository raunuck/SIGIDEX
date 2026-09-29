import React, { useMemo, useState } from 'react';
import Plot from './PlotComponent';
import { RotateCcw } from 'lucide-react';

/**
 * Interactive Spectrogram Heatmap using Plotly:
 * - Frequency on X, Time on Y with time INCREASING DOWNWARD (autorange: 'reversed')
 * - Interactive hover tooltip: Freq / Time / Power
 * - Zoom & pan enabled (drag & scroll)
 * - Color range clipped to top ~60 dB below the peak
 */
export default function SpectrogramPlotly({ spectrogram }) {
  const [resetKey, setResetKey] = useState(0);

  const traces = useMemo(() => {
    if (
      !spectrogram ||
      !spectrogram.f_khz ||
      !spectrogram.t_sec ||
      !spectrogram.z_db ||
      !spectrogram.z_db.length
    ) {
      return [];
    }

    return [
      {
        type: 'heatmap',
        x: spectrogram.f_khz,
        y: spectrogram.t_sec,
        z: spectrogram.z_db,
        colorscale: 'Inferno',
        zmin: spectrogram.z_min,
        zmax: spectrogram.z_max,
        colorbar: {
          title: {
            text: 'dB',
            font: { color: '#8E8E93', family: 'monospace', size: 9 },
          },
          tickfont: { color: '#8E8E93', family: 'monospace', size: 9 },
          len: 0.95,
          thickness: 12,
          outlinecolor: '#2A2A2E',
        },
        hovertemplate:
          '<b>Freq</b>: %{x:.3f} kHz<br><b>Time</b>: %{y:.4f} s<br><b>Power</b>: %{z:.1f} dB<extra></extra>',
      },
    ];
  }, [spectrogram]);

  const layout = useMemo(() => {
    return {
      autosize: true,
      height: 270,
      paper_bgcolor: '#0A0A0B',
      plot_bgcolor: '#0A0A0B',
      margin: { l: 50, r: 65, t: 15, b: 40 },
      font: {
        family: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
        color: '#8E8E93',
        size: 10,
      },
      xaxis: {
        title: {
          text: 'Frequency (kHz)',
          font: { color: '#8E8E93', size: 10 },
        },
        gridcolor: '#1E1E22',
        linecolor: '#2A2A2E',
        tickcolor: '#2A2A2E',
        tickfont: { color: '#8E8E93', size: 9 },
      },
      yaxis: {
        title: {
          text: 'Time (s) [increasing downward]',
          font: { color: '#8E8E93', size: 10 },
        },
        autorange: 'reversed', // Time INCREASING DOWNWARD per requirement
        gridcolor: '#1E1E22',
        linecolor: '#2A2A2E',
        tickcolor: '#2A2A2E',
        tickfont: { color: '#8E8E93', size: 9 },
      },
      dragmode: 'zoom',
    };
  }, []);

  const config = useMemo(
    () => ({
      scrollZoom: true,
      displayModeBar: true,
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: [
        'lasso2d',
        'select2d',
        'hoverClosestCartesian',
        'hoverCompareCartesian',
      ],
    }),
    []
  );

  if (!spectrogram || !spectrogram.z_db || !spectrogram.z_db.length) {
    return (
      <div className="h-64 flex items-center justify-center font-mono text-xs text-[#555558]">
        NO SPECTROGRAM DATA LOADED
      </div>
    );
  }

  return (
    <div className="w-full flex flex-col space-y-2">
      {/* Mini toolbar */}
      <div className="flex items-center justify-between px-2 pt-1 text-[11px] font-mono text-[#8E8E93]">
        <span className="text-[#8E8E93]">
          HEATMAP CLIPPED: [{spectrogram.z_min} dB .. {spectrogram.z_max} dB]
        </span>
        <button
          onClick={() => setResetKey((k) => k + 1)}
          className="inline-flex items-center px-2 py-0.5 rounded bg-[#1E1E22] hover:bg-[#F5A623] hover:text-[#0A0A0B] text-white transition-colors border border-[#2A2A2E]"
          title="Reset spectrogram zoom and view"
        >
          <RotateCcw className="w-3 h-3 mr-1" />
          RESET ZOOM
        </button>
      </div>

      <div className="w-full bg-[#0A0A0B] rounded border border-[#2A2A2E]/60 overflow-hidden">
        <Plot
          key={resetKey}
          data={traces}
          layout={layout}
          config={config}
          useResizeHandler={true}
          className="w-full"
          style={{ width: '100%', height: '270px' }}
        />
      </div>
    </div>
  );
}
