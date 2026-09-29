import React from 'react';
import Pill from './Pill';

/**
 * Reusable SIGDEX Card Panel:
 *  - #141416 panel background
 *  - 1px #2A2A2E border
 *  - Caps-lock header row with icon/glyph + title
 *  - Optional top-right status pill or action widget
 */
export default function Card({
  title,
  glyph = '',
  statusText = null,
  statusVariant = 'amber',
  topRight = null,
  children,
  className = '',
}) {
  return (
    <div className={`bg-[#141416] border border-[#2A2A2E] rounded-lg p-4 sm:p-5 flex flex-col ${className}`}>
      {/* Header Row */}
      <div className="flex items-center justify-between pb-3 mb-3 border-b border-[#2A2A2E]/50">
        <div className="flex items-center space-x-2">
          {glyph && <span className="text-[#8E8E93] text-xs font-mono">{glyph}</span>}
          <h3 className="text-white text-xs font-bold uppercase tracking-wider font-sans">
            {title}
          </h3>
        </div>
        <div>
          {topRight ? (
            topRight
          ) : statusText ? (
            <Pill variant={statusVariant}>{statusText}</Pill>
          ) : null}
        </div>
      </div>

      {/* Body */}
      <div className="flex-1 flex flex-col">
        {children}
      </div>
    </div>
  );
}
