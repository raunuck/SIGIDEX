import React from 'react';

/**
 * Reusable SIGDEX status pill / badge
 */
export default function Pill({ 
  children, 
  variant = 'amber', 
  className = '' 
}) {
  const variantStyles = {
    amber: 'bg-[#231B0D] text-[#F5A623] border-[#F5A623]',
    green: 'bg-[#0D231A] text-[#50E3C2] border-[#50E3C2]',
    pink: 'bg-[#261016] text-[#E85D75] border-[#E85D75]',
    muted: 'bg-[#141416] text-[#8E8E93] border-[#2A2A2E]',
    solidAmber: 'bg-[#F5A623] text-[#0A0A0B] border-[#F5A623] font-extrabold',
  };

  const style = variantStyles[variant] || variantStyles.amber;

  return (
    <span 
      className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-bold uppercase tracking-wider border ${style} ${className}`}
    >
      {children}
    </span>
  );
}
