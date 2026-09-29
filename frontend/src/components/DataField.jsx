import React from 'react';

/**
 * Standard SIGDEX metric display:
 * Small gray caps-lock label above a large bold monospace value.
 */
export default function DataField({
  label,
  value,
  unit = '',
  highlight = false,
  color = 'amber', // 'amber' | 'white' | 'pink' | 'muted'
  className = '',
}) {
  const colorMap = {
    amber: 'text-[#F5A623]',
    white: 'text-white',
    pink: 'text-[#E85D75]',
    muted: 'text-[#8E8E93]',
  };

  const valColor = colorMap[color] || 'text-[#F5A623]';

  return (
    <div className={`flex flex-col ${className}`}>
      <span className="text-[10px] sm:text-[11px] font-mono font-bold text-[#8E8E93] uppercase tracking-wider mb-1">
        {label}
      </span>
      <div className="flex items-baseline space-x-1">
        <span className={`text-base sm:text-lg font-mono font-extrabold ${valColor}`}>
          {value}
        </span>
        {unit && (
          <span className="text-xs font-mono text-[#8E8E93] font-medium">
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}
