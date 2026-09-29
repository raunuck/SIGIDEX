import React from 'react';

export default function NavBar({
  activeTab,
  onTabChange,
  contextLabel = 'SDR: USRP B210',
  dspActive = true,
}) {
  const tabs = [
    { id: 0, label: '1. INGESTION' },
    { id: 1, label: '2. ANALYSIS' },
    { id: 2, label: '3. CONSTELLATION' },
    { id: 3, label: '4. BITSTREAM' },
  ];

  return (
    <header className="h-16 bg-[#0A0A0B] border-b border-[#2A2A2E] px-6 flex items-center justify-between sticky top-0 z-30 select-none">
      {/* Left: Brand & Version */}
      <div className="flex items-center space-x-3">
        <div className="w-7 h-7 bg-[#F5A623] rounded flex items-center justify-center text-[#0A0A0B] font-mono font-black text-sm">
          ///
        </div>
        <span className="text-white font-extrabold text-base tracking-wider font-sans">
          SIGDEX
        </span>
        <span className="text-[10px] font-mono font-bold text-[#F5A623] bg-[#231B0D] border border-[#F5A623] px-1.5 py-0.5 rounded">
          v2.4-PRO
        </span>
      </div>

      {/* Center: 4 Navigation Tabs */}
      <nav className="flex items-center space-x-1 sm:space-x-2">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`px-3.5 py-1.5 rounded-full text-xs font-sans tracking-wide transition-all duration-150 ${
                isActive
                  ? 'bg-[#F5A623] text-[#0A0A0B] font-extrabold shadow-sm'
                  : 'text-[#8E8E93] hover:text-white hover:bg-white/5 font-semibold'
              }`}
            >
              {tab.label}
            </button>
          );
        })}
      </nav>

      {/* Right: Engine Status & Hardware Context */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-1.5 text-xs font-mono font-bold text-[#F5A623]">
          <span className="inline-block w-2 h-2 rounded-full bg-[#F5A623] animate-pulse" />
          <span>DSP ENGINE ACTIVE</span>
        </div>
        <span className="text-[#2A2A2E]">|</span>
        <span className="text-xs font-mono text-[#8E8E93] tracking-wide hidden md:inline">
          {contextLabel}
        </span>
      </div>
    </header>
  );
}
