import React, { useState, useMemo, useRef, useCallback } from 'react';
import Plot from './PlotComponent';
import { RotateCcw, XCircle, Crosshair } from 'lucide-react';

/**
 * Interactive Plotly PSD Chart matching SIGDEX dark aesthetic:
 * - Drag-to-zoom (box zoom), scroll-to-pan
 * - Hover crosshair with frequency + power readouts
 * - Click on plot to place/remove interactive markers
 * - Reset-zoom button
 * - One-sided (0..fs/2) vs Two-sided (-fs/2..fs/2) display
 * - Y-axis clipped to ~80 dB below peak
 */
export default function PsdPlotlyChart({
  freqs = [],
  psd = [],
  peakFreqKhz = null,
  peakPowerDb = null,
  spectrumMode = 'one-sided',
  yMinClip = null,
  yMaxClip = null,
}) {
  const [markers, setMarkers] = useState([]);
  const [userLayout, setUserLayout] = useState(null);
  const [resetKey, setResetKey] = useState(0);

  // Compute default axis bounds
  const defaultBounds = useMemo(() => {
    if (!freqs.length || !psd.length) {
      return { xMin: 0, xMax: 10, yMin: -100, yMax: 0 };
    }
    const xMin = freqs[0];
    const xMax = freqs[freqs.length - 1];

    const maxP = peakPowerDb !== null ? peakPowerDb : Math.max(...psd);
    const yMax = yMaxClip !== null ? yMaxClip : maxP + 5;
    const yMin = yMinClip !== null ? yMinClip : maxP - 80;

    return { xMin, xMax, yMin, yMax };
  }, [freqs, psd, peakPowerDb, yMinClip, yMaxClip]);

  // Click on plot to add or remove markers
  const handlePlotClick = useCallback(
    (eventData) => {
      if (!eventData || !eventData.points || !eventData.points.length) return;
      const pt = eventData.points[0];
      const clickX = pt.x;
      const clickY = pt.y;

      // Check if clicking near an existing marker (within 1.5% of frequency span)
      const span = Math.abs(defaultBounds.xMax - defaultBounds.xMin) || 1.0;
      const existingIdx = markers.findIndex(
        (m) => Math.abs(m.x - clickX) / span < 0.02
      );

      if (existingIdx >= 0) {
        // Remove existing marker
        setMarkers((prev) => prev.filter((_, idx) => idx !== existingIdx));
      } else {
        // Add new marker
        setMarkers((prev) => [
          ...prev,
          {
            x: clickX,
            y: clickY,
            id: Date.now(),
          },
        ]);
      }
    },
    [markers, defaultBounds]
  );

  const handleResetZoom = () => {
    setUserLayout({
      'xaxis.range[0]': defaultBounds.xMin,
      'xaxis.range[1]': defaultBounds.xMax,
      'yaxis.range[0]': defaultBounds.yMin,
      'yaxis.range[1]': defaultBounds.yMax,
    });
    setResetKey((k) => k + 1);
  };

  const handleClearMarkers = () => {
    setMarkers([]);
  };

  // Plotly traces
  const traces = useMemo(() => {
    if (!freqs.length || !psd.length) return [];

    const mainTrace = {
      x: freqs,
      y: psd,
      type: 'scatter',
      mode: 'lines',
      name: 'PSD',
      line: {
        color: '#F5A623',
        width: 1.5,
      },
      hoverinfo: 'x+y',
      hovertemplate:
        '<b>Freq</b>: %{x:.3f} kHz<br><b>Power</b>: %{y:.1f} dB/Hz<extra></extra>',
    };

    // User-placed interactive markers
    const markerTrace = {
      x: markers.map((m) => m.x),
      y: markers.map((m) => m.y),
      type: 'scatter',
      mode: 'markers+text',
      name: 'Markers',
      marker: {
        color: '#E85D75',
        size: 9,
        symbol: 'diamond',
        line: { color: '#FFFFFF', width: 1 },
      },
      text: markers.map((m, i) => `M${i + 1}`),
      textposition: 'top center',
      textfont: {
        family: 'monospace',
        size: 10,
        color: '#E85D75',
      },
      hovertemplate:
        '<b>Marker</b>: %{text}<br><b>Freq</b>: %{x:.3f} kHz<br><b>Power</b>: %{y:.1f} dB<extra></extra>',
    };

    return [mainTrace, markerTrace];
  }, [freqs, psd, markers]);

  // Plotly layout
  const layout = useMemo(() => {
    const shapes = [];

    // Dashed vertical line at peak carrier frequency
    if (peakFreqKhz !== null) {
      shapes.push({
        type: 'line',
        x0: peakFreqKhz,
        x1: peakFreqKhz,
        y0: defaultBounds.yMin,
        y1: defaultBounds.yMax,
        line: {
          color: '#E85D75',
          width: 1,
          dash: 'dash',
        },
      });
    }

    const baseLayout = {
      autosize: true,
      height: 270,
      paper_bgcolor: '#0A0A0B',
      plot_bgcolor: '#0A0A0B',
      margin: { l: 50, r: 25, t: 20, b: 40 },
      font: {
        family: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace',
        color: '#8E8E93',
        size: 10,
      },
      xaxis: {
        title: {
          text: `Frequency (kHz) [${spectrumMode}]`,
          font: { color: '#8E8E93', size: 10 },
        },
        range: [defaultBounds.xMin, defaultBounds.xMax],
        gridcolor: '#1E1E22',
        linecolor: '#2A2A2E',
        tickcolor: '#2A2A2E',
        tickfont: { color: '#8E8E93', size: 9 },
        showspikes: true,
        spikemode: 'across',
        spikecolor: '#8E8E93',
        spikethickness: 1,
        spikedash: 'dash',
        zeroline: false,
      },
      yaxis: {
        title: {
          text: 'Power (dB/Hz)',
          font: { color: '#8E8E93', size: 10 },
        },
        range: [defaultBounds.yMin, defaultBounds.yMax],
        gridcolor: '#1E1E22',
        linecolor: '#2A2A2E',
        tickcolor: '#2A2A2E',
        tickfont: { color: '#8E8E93', size: 9 },
        showspikes: true,
        spikemode: 'across',
        spikecolor: '#8E8E93',
        spikethickness: 1,
        spikedash: 'dash',
        zeroline: false,
      },
      shapes,
      showlegend: false,
      dragmode: 'zoom',
      hovermode: 'closest',
    };

    if (userLayout) {
      if (userLayout['xaxis.range[0]'] !== undefined) {
        baseLayout.xaxis.range = [
          userLayout['xaxis.range[0]'],
          userLayout['xaxis.range[1]'],
        ];
      }
      if (userLayout['yaxis.range[0]'] !== undefined) {
        baseLayout.yaxis.range = [
          userLayout['yaxis.range[0]'],
          userLayout['yaxis.range[1]'],
        ];
      }
    }

    return baseLayout;
  }, [defaultBounds, peakFreqKhz, spectrumMode, userLayout]);

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
        'toggleSpikelines',
      ],
    }),
    []
  );

  if (!freqs.length || !psd.length) {
    return (
      <div className="h-64 flex items-center justify-center text-[#555558] font-mono text-xs">
        NO SPECTRAL DATA LOADED
      </div>
    );
  }

  return (
    <div className="w-full flex flex-col space-y-2">
      {/* Interactive Controls Toolbar */}
      <div className="flex items-center justify-between px-2 pt-1 text-[11px] font-mono text-[#8E8E93]">
        <div className="flex items-center space-x-3">
          <span className="inline-flex items-center text-[#F5A623] font-bold">
            <Crosshair className="w-3.5 h-3.5 mr-1" />
            CLICK CHART TO ADD/REMOVE MARKER
          </span>
          {markers.length > 0 && (
            <span className="text-white bg-[#1E1E22] px-2 py-0.5 rounded border border-[#2A2A2E]">
              {markers.length} MARKER{markers.length > 1 ? 'S' : ''} ACTIVE
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {markers.length > 0 && (
            <button
              onClick={handleClearMarkers}
              className="inline-flex items-center px-2 py-0.5 rounded bg-[#1E1E22] hover:bg-[#2A2A2E] text-[#E85D75] transition-colors border border-[#2A2A2E]"
              title="Clear all placed markers"
            >
              <XCircle className="w-3 h-3 mr-1" />
              CLEAR MARKERS
            </button>
          )}

          <button
            onClick={handleResetZoom}
            className="inline-flex items-center px-2 py-0.5 rounded bg-[#1E1E22] hover:bg-[#F5A623] hover:text-[#0A0A0B] text-white transition-colors border border-[#2A2A2E]"
            title="Reset zoom to initial frequency and power limits"
          >
            <RotateCcw className="w-3 h-3 mr-1" />
            RESET ZOOM
          </button>
        </div>
      </div>

      {/* Plotly Canvas Container */}
      <div className="w-full bg-[#0A0A0B] rounded border border-[#2A2A2E]/60 overflow-hidden">
        <Plot
          key={resetKey}
          data={traces}
          layout={layout}
          config={config}
          onClick={handlePlotClick}
          onRelayout={(ev) => {
            if (ev['xaxis.autorange'] || ev['yaxis.autorange']) {
              setUserLayout(null);
            }
          }}
          useResizeHandler={true}
          className="w-full"
          style={{ width: '100%', height: '270px' }}
        />
      </div>
    </div>
  );
}
