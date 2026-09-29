import React from 'react';
import Pill from './Pill';

export default function Breadcrumb({
  section = 'INGEST',
  screen = 'SIGNAL DATA',
  filename = null,
  sampleRate = null,
  statusBadge = null,
  statusVariant = 'amber',
}) {
  return (
    <div className="h-10 bg-[#0A0A0B] border-b border-[#2A2A2E] px-6 flex items-center justify-between text-xs select-none">
      {/* Left: Section / Screen Name */}
      <div className="flex items-center space-x-2">
        <span className="text-[#8E8E93] font-bold font-sans tracking-widest text-[11px]">
          {section.toUpperCase()}
        </span>
        <span className="text-[#2A2A2E]">/</span>
        <span className="text-[#F5A623] font-extrabold font-sans tracking-widest text-[11px]">
          {screen.toUpperCase()}
        </span>
      </div>

      {/* Right: Contextual Badges */}
      <div className="flex items-center space-x-3">
        {filename && (
          <span className="text-[#8E8E93] font-mono text-[11px] truncate max-w-[280px]">
            FILE: <span className="text-white">{filename}</span>
          </span>
        )}
        {sampleRate && (
          <span className="text-[#F5A623] font-mono font-bold text-[11px] hidden sm:inline">
            {sampleRate}
          </span>
        )}
        {statusBadge && (
          <Pill variant={statusVariant}>{statusBadge}</Pill>
        )}
      </div>
    </div>
  );
}
