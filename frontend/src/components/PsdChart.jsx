import React, { useMemo } from 'react';

/**
 * Lightweight SVG PSD Chart:
 *  - Amber power trace (#F5A623)
 *  - Dark background and faint grid lines (#2A2A2E)
 *  - Dashed pink vertical marker with label at detected peak carrier frequency
 */
export default function PsdChart({ freqs = [], psd = [], peakFreqKhz = null }) {
  const chart = useMemo(() => {
    if (!freqs.length || !psd.length) return null;

    const minFreq = freqs[0];
    const maxFreq = freqs[freqs.length - 1];
    const minPsd = Math.min(...psd);
    const maxPsd = Math.max(...psd);

    // Add padding to range
    const yMin = Math.floor(minPsd / 10) * 10 - 5;
    const yMax = Math.ceil(maxPsd / 10) * 10 + 5;

    const width = 800;
    const height = 260;
    const padding = { top: 20, right: 30, bottom: 35, left: 55 };

    const innerWidth = width - padding.left - padding.right;
    const innerHeight = height - padding.top - padding.bottom;

    const scaleX = (f) => padding.left + ((f - minFreq) / (maxFreq - minFreq || 1)) * innerWidth;
    const scaleY = (p) => padding.top + innerHeight - ((p - yMin) / (yMax - yMin || 1)) * innerHeight;

    // Generate path data
    let pathD = '';
    for (let i = 0; i < freqs.length; i++) {
      const x = scaleX(freqs[i]);
      const y = scaleY(psd[i]);
      pathD += (i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`);
    }

    // Grid ticks
    const yTicks = [];
    const stepY = (yMax - yMin) / 4;
    for (let i = 0; i <= 4; i++) {
      const val = yMin + i * stepY;
      yTicks.push({ val: Math.round(val), y: scaleY(val) });
    }

    const xTicks = [];
    const stepX = (maxFreq - minFreq) / 6;
    for (let i = 0; i <= 6; i++) {
      const val = minFreq + i * stepX;
      xTicks.push({ val: val.toFixed(1), x: scaleX(val) });
    }

    // Peak marker position
    const peakX = peakFreqKhz !== null ? scaleX(peakFreqKhz) : null;

    return {
      width,
      height,
      padding,
      pathD,
      yTicks,
      xTicks,
      peakX,
      yMin,
      yMax,
    };
  }, [freqs, psd, peakFreqKhz]);

  if (!chart) {
    return (
      <div className="h-64 flex items-center justify-center text-[#555558] font-mono text-xs">
        NO SPECTRAL DATA LOADED
      </div>
    );
  }

  return (
    <div className="w-full h-full flex flex-col justify-center">
      <svg
        viewBox={`0 0 ${chart.width} ${chart.height}`}
        className="w-full h-auto max-h-72 select-none"
      >
        {/* Horizontal grid lines */}
        {chart.yTicks.map((t, idx) => (
          <g key={idx}>
            <line
              x1={chart.padding.left}
              y1={t.y}
              x2={chart.width - chart.padding.right}
              y2={t.y}
              stroke="#2A2A2E"
              strokeDasharray="3 3"
              strokeWidth="1"
            />
            <text
              x={chart.padding.left - 8}
              y={t.y + 4}
              fill="#8E8E93"
              fontSize="10"
              fontFamily="monospace"
              textAnchor="end"
            >
              {t.val} dB
            </text>
          </g>
        ))}

        {/* Vertical grid lines */}
        {chart.xTicks.map((t, idx) => (
          <g key={idx}>
            <line
              x1={t.x}
              y1={chart.padding.top}
              x2={t.x}
              y2={chart.height - chart.padding.bottom}
              stroke="#2A2A2E"
              strokeDasharray="3 3"
              strokeWidth="1"
            />
            <text
              x={t.x}
              y={chart.height - chart.padding.bottom + 16}
              fill="#8E8E93"
              fontSize="10"
              fontFamily="monospace"
              textAnchor="middle"
            >
              {t.val}k
            </text>
          </g>
        ))}

        {/* Axis border */}
        <rect
          x={chart.padding.left}
          y={chart.padding.top}
          width={chart.width - chart.padding.left - chart.padding.right}
          height={chart.height - chart.padding.top - chart.padding.bottom}
          fill="none"
          stroke="#2A2A2E"
          strokeWidth="1"
        />

        {/* PSD Curve */}
        <path
          d={chart.pathD}
          fill="none"
          stroke="#F5A623"
          strokeWidth="2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* Peak Marker Line & Label */}
        {chart.peakX !== null && (
          <g>
            <line
              x1={chart.peakX}
              y1={chart.padding.top}
              x2={chart.peakX}
              y2={chart.height - chart.padding.bottom}
              stroke="#E85D75"
              strokeDasharray="4 4"
              strokeWidth="1.5"
            />
            {/* Marker Label Box */}
            <rect
              x={Math.min(chart.peakX + 6, chart.width - chart.padding.right - 140)}
              y={chart.padding.top + 8}
              width="130"
              height="20"
              rx="3"
              fill="#141416"
              stroke="#E85D75"
              strokeWidth="1"
            />
            <text
              x={Math.min(chart.peakX + 12, chart.width - chart.padding.right - 134)}
              y={chart.padding.top + 22}
              fill="#E85D75"
              fontSize="10"
              fontFamily="monospace"
              fontWeight="bold"
            >
              {peakFreqKhz.toFixed(2)} kHz (MARKER 1)
            </text>
          </g>
        )}
      </svg>
    </div>
  );
}
