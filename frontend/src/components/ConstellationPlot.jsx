import React, { useMemo } from 'react';

/**
 * High-resolution SVG I/Q Constellation Scatter Plot:
 *  - Amber points (#F5A623) with subtle glow
 *  - Centered crosshairs and dashed quadrant circles
 *  - Axis labels for In-Phase (I) and Quadrature (Q)
 */
export default function ConstellationPlot({ points = [] }) {
  const size = 520;
  const padding = 45;
  const plotRadius = (size - padding * 2) / 2;
  const centerX = size / 2;
  const centerY = size / 2;

  // Coordinate mapper: maps [-2.0, 2.0] to plot bounds
  const mapCoord = (val) => (val / 2.0) * plotRadius;

  return (
    <div className="w-full h-full flex items-center justify-center select-none py-2">
      <svg
        viewBox={`0 0 ${size} ${size}`}
        className="w-full max-w-[460px] h-auto aspect-square bg-[#0A0A0B] rounded-lg border border-[#2A2A2E]/60 p-2"
      >
        {/* Background Grid Circles */}
        <circle
          cx={centerX}
          cy={centerY}
          r={plotRadius * 0.5}
          fill="none"
          stroke="#2A2A2E"
          strokeDasharray="3 3"
          strokeWidth="1"
        />
        <circle
          cx={centerX}
          cy={centerY}
          r={plotRadius}
          fill="none"
          stroke="#2A2A2E"
          strokeWidth="1"
        />

        {/* Axis Crosshairs */}
        <line
          x1={padding}
          y1={centerY}
          x2={size - padding}
          y2={centerY}
          stroke="#2A2A2E"
          strokeWidth="1.5"
        />
        <line
          x1={centerX}
          y1={padding}
          x2={centerX}
          y2={size - padding}
          stroke="#2A2A2E"
          strokeWidth="1.5"
        />

        {/* Target Decision Centroid Markers at -1 and +1 */}
        <circle
          cx={centerX + mapCoord(1.0)}
          cy={centerY}
          r="6"
          fill="none"
          stroke="#E85D75"
          strokeWidth="1.5"
          strokeDasharray="2 2"
        />
        <circle
          cx={centerX + mapCoord(-1.0)}
          cy={centerY}
          r="6"
          fill="none"
          stroke="#E85D75"
          strokeWidth="1.5"
          strokeDasharray="2 2"
        />

        {/* Axis Labels */}
        <text
          x={size - padding + 5}
          y={centerY + 4}
          fill="#8E8E93"
          fontSize="10"
          fontFamily="monospace"
          fontWeight="bold"
        >
          +I
        </text>
        <text
          x={padding - 18}
          y={centerY + 4}
          fill="#8E8E93"
          fontSize="10"
          fontFamily="monospace"
          fontWeight="bold"
        >
          -I
        </text>
        <text
          x={centerX}
          y={padding - 8}
          fill="#8E8E93"
          fontSize="10"
          fontFamily="monospace"
          fontWeight="bold"
          textAnchor="middle"
        >
          +Q
        </text>
        <text
          x={centerX}
          y={size - padding + 18}
          fill="#8E8E93"
          fontSize="10"
          fontFamily="monospace"
          fontWeight="bold"
          textAnchor="middle"
        >
          -Q
        </text>

        {/* Symbol Scatter Points */}
        {points.map((pt, idx) => {
          const cx = centerX + mapCoord(pt.i);
          const cy = centerY - mapCoord(pt.q); // Invert Q for Cartesian Y-up
          return (
            <circle
              key={idx}
              cx={cx}
              cy={cy}
              r="3.5"
              fill="#F5A623"
              fillOpacity="0.85"
            />
          );
        })}

        {points.length === 0 && (
          <text
            x={centerX}
            y={centerY}
            fill="#555558"
            fontSize="12"
            fontFamily="monospace"
            textAnchor="middle"
          >
            AWAITING DEMODULATED SYMBOLS...
          </text>
        )}
      </svg>
    </div>
  );
}
