import React, { useState } from 'react';
import NavBar from './components/NavBar';
import Breadcrumb from './components/Breadcrumb';
import IngestionTab from './tabs/IngestionTab';
import AnalysisTab from './tabs/AnalysisTab';
import ConstellationTab from './tabs/ConstellationTab';
import BitstreamTab from './tabs/BitstreamTab';

export default function App() {
  const [activeTab, setActiveTab] = useState(0);
  const [fileId, setFileId] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [fecData, setFecData] = useState(null);
  const [dspStatus, setDspStatus] = useState({ badge: 'IDLE', variant: 'muted' });

  // Breadcrumb titles per tab
  const breadcrumbConfig = [
    { section: 'INGEST', screen: 'SIGNAL DATA' },
    { section: 'ANALYSIS', screen: 'RUNTIME' },
    { section: 'ANALYSIS', screen: 'I/Q CONSTELLATION' },
    { section: 'ANALYSIS', screen: 'PAYLOAD DECODER' },
  ];

  const handleSignalLoaded = (data) => {
    setFileId(data.file_id);
    setFileMeta(data);
    setFecData(null);
    setDspStatus({ badge: 'DSP READY', variant: 'amber' });
    // Advance to Tab 1 (Analysis)
    setActiveTab(1);
  };

  const handleAnalysisLoaded = (data) => {
    if (data.fec) {
      setFecData(data.fec);
    }
    const modType = data.report?.modulation?.type?.toLowerCase();
    if (modType === 'fsk' || modType === 'qam') {
      setDspStatus({ badge: 'UNSUPPORTED', variant: 'muted' });
    } else {
      setDspStatus({ badge: 'DSP READY', variant: 'amber' });
    }
  };

  const handleConstellationLoaded = (data) => {
    if (data.supported === false) {
      setDspStatus({ badge: 'UNSUPPORTED', variant: 'muted' });
    } else if (data.error) {
      setDspStatus({ badge: 'DSP ERROR', variant: 'pink' });
    }
  };

  const handleBitstreamLoaded = (data) => {
    setFecData(data);
    if (data.supported === false) {
      setDspStatus({ badge: 'UNSUPPORTED', variant: 'muted' });
    } else if (data.error) {
      setDspStatus({ badge: 'DSP ERROR', variant: 'pink' });
    } else {
      setDspStatus({ badge: 'DSP READY', variant: 'amber' });
    }
  };

  const currentBc = breadcrumbConfig[activeTab] || breadcrumbConfig[0];
  const sampleRateStr = fileMeta
    ? fileMeta.sample_rate >= 1000000
      ? `${(fileMeta.sample_rate / 1000000).toFixed(3)} MHz`
      : `${(fileMeta.sample_rate / 1000).toFixed(1)} kHz`
    : null;

  return (
    <div className="min-h-screen bg-[#0A0A0B] text-white flex flex-col font-sans selection:bg-[#F5A623] selection:text-[#0A0A0B]">
      {/* 1. Top Navigation Bar */}
      <NavBar
        activeTab={activeTab}
        onTabChange={(tabId) => setActiveTab(tabId)}
        contextLabel={fileMeta ? `SRC: ${fileMeta.filename}` : 'SDR: USRP B210'}
        dspActive={fileId !== null}
      />

      {/* 2. Breadcrumb Subheader */}
      <Breadcrumb
        section={currentBc.section}
        screen={currentBc.screen}
        filename={fileMeta?.filename}
        sampleRate={sampleRateStr}
        statusBadge={dspStatus.badge}
        statusVariant={dspStatus.variant}
      />

      {/* 3. Main Workspace Tab Views */}
      <main className="flex-1 w-full overflow-y-auto">
        {activeTab === 0 && (
          <IngestionTab
            onSignalLoaded={handleSignalLoaded}
            activeFile={fileMeta}
          />
        )}

        {activeTab === 1 && (
          <AnalysisTab
            fileId={fileId}
            fecData={fecData}
            onAnalysisLoaded={handleAnalysisLoaded}
          />
        )}

        {activeTab === 2 && (
          <ConstellationTab
            fileId={fileId}
            onConstellationLoaded={handleConstellationLoaded}
          />
        )}

        {activeTab === 3 && (
          <BitstreamTab
            fileId={fileId}
            onBitstreamLoaded={handleBitstreamLoaded}
          />
        )}
      </main>
    </div>
  );
}
