import React, { useState, useRef, useEffect } from 'react';
import Card from '../components/Card';
import { Upload, HardDrive, Play, Clock, Sparkles } from 'lucide-react';

export default function IngestionTab({ onSignalLoaded, activeFile }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [sampleRate, setSampleRate] = useState('44100');
  const [dataType, setDataType] = useState('float32');
  const [autoDetect, setAutoDetect] = useState(true);
  const [recentOps, setRecentOps] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState(null);
  const fileInputRef = useRef(null);

  // Fetch recent operations on mount
  useEffect(() => {
    fetchRecentOps();
  }, []);

  const fetchRecentOps = async () => {
    try {
      const res = await fetch('/api/ingest/recent');
      if (res.ok) {
        const data = await res.json();
        setRecentOps(data);
      }
    } catch (e) {
      console.error('Failed to fetch recents:', e);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  };

  const handleFileSelected = (file) => {
    setSelectedFile(file);
    setErrorMsg(null);
  };

  const handleInitializeDsp = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('sample_rate', sampleRate);
    formData.append('data_type', dataType);

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Ingestion failed');
      }

      const data = await res.json();
      await fetchRecentOps();
      onSignalLoaded(data);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadSample = async (filename) => {
    setLoading(true);
    setErrorMsg(null);
    try {
      const res = await fetch(`/api/ingest/sample/${filename}`, {
        method: 'POST',
      });
      if (!res.ok) {
        throw new Error(`Failed to load sample ${filename}`);
      }
      const data = await res.json();
      await fetchRecentOps();
      onSignalLoaded(data);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  const isIqFile = selectedFile && !selectedFile.name.toLowerCase().endsWith('.wav');

  return (
    <div className="max-w-4xl mx-auto py-8 px-4 flex flex-col space-y-6">
      {/* Title & Subtitle */}
      <div className="text-center space-y-1.5">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight font-sans">
          Ingest Signal Data
        </h1>
        <p className="text-sm text-[#8E8E93]">
          Upload baseband recordings or configure a real-time software defined radio source.
        </p>
      </div>

      {errorMsg && (
        <div className="bg-[#261016] border border-[#E85D75] text-[#E85D75] p-3 rounded-lg text-xs font-mono">
          ERROR: {errorMsg}
        </div>
      )}

      {/* Drag & Drop Upload Zone */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        className={`border-2 border-dashed rounded-xl p-8 flex flex-col items-center justify-center transition-all duration-200 cursor-pointer ${
          dragActive
            ? 'border-[#F5A623] bg-[#F5A623]/10 scale-[1.01]'
            : 'border-[#F5A623] bg-[#141416]/70 hover:bg-[#141416]'
        }`}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          onChange={handleFileChange}
          accept=".wav,.wave,.iq,.bin,.dat,.raw"
        />

        <div className="w-12 h-12 rounded-full bg-[#231B0D] border border-[#F5A623]/40 flex items-center justify-center text-[#F5A623] mb-3">
          <Upload className="w-6 h-6 text-[#F5A623]" />
        </div>

        <h3 className="text-base font-bold text-white mb-1">
          {selectedFile ? selectedFile.name : 'Drag & Drop Baseband Capture File'}
        </h3>
        <p className="text-xs text-[#8E8E93] mb-4">
          Supports .IQ, .wav, or raw binary format (Max 4.0 GB)
        </p>

        <button
          type="button"
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
          className="bg-[#1A1A1E] hover:border-[#F5A623] hover:text-[#F5A623] text-white border border-[#2A2A2E] px-4 py-2 rounded-lg text-xs font-semibold tracking-wide transition-all shadow-sm"
        >
          Browse Local Storage
        </button>
      </div>

      {/* Specifications Card */}
      <Card title="IQ Baseband Specifications" glyph="⚙">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
          {/* Sample Rate */}
          <div className="flex flex-col space-y-1.5">
            <label className="text-[11px] font-mono font-bold text-[#8E8E93] uppercase">
              SAMPLE RATE
            </label>
            <select
              value={sampleRate}
              onChange={(e) => setSampleRate(e.target.value)}
              className="bg-[#0A0A0B] border border-[#2A2A2E] rounded-md px-3 py-2 text-white font-mono text-xs focus:border-[#F5A623] focus:outline-none"
            >
              <option value="44100">44.1 kHz (Audio / IF)</option>
              <option value="500000">500.0 kHz</option>
              <option value="1000000">1.000 MHz</option>
              <option value="2000000">2.000 MHz</option>
              <option value="2400000">2.400 MHz (UHF / Default)</option>
            </select>
          </div>

          {/* Data Type */}
          <div className="flex flex-col space-y-1.5">
            <label className="text-[11px] font-mono font-bold text-[#8E8E93] uppercase">
              DATA TYPE
            </label>
            <select
              value={dataType}
              onChange={(e) => setDataType(e.target.value)}
              className="bg-[#0A0A0B] border border-[#2A2A2E] rounded-md px-3 py-2 text-white font-mono text-xs focus:border-[#F5A623] focus:outline-none"
            >
              <option value="float32">Complex Float32 (I/Q)</option>
              <option value="int16">Complex Int16 (I/Q)</option>
              <option value="int8">Complex Int8 (I/Q)</option>
              <option value="wav">WAV Baseband</option>
            </select>
          </div>
        </div>

        {/* Checkbox and Initialize Button */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between pt-2 border-t border-[#2A2A2E]/40 gap-3">
          <label className="flex items-center space-x-2 text-xs text-[#8E8E93] cursor-pointer select-none">
            <input
              type="checkbox"
              checked={autoDetect}
              onChange={(e) => setAutoDetect(e.target.checked)}
              className="accent-[#F5A623] rounded w-4 h-4 bg-[#0A0A0B] border-[#2A2A2E]"
            />
            <span>Auto-detect Center Frequency from capture metadata</span>
          </label>

          <button
            type="button"
            disabled={!selectedFile || loading}
            onClick={handleInitializeDsp}
            className={`px-5 py-2.5 rounded-lg text-xs font-bold uppercase tracking-wider transition-all duration-150 ${
              selectedFile && !loading
                ? 'bg-[#F5A623] hover:bg-[#FFB84D] text-[#0A0A0B] shadow-md cursor-pointer'
                : 'bg-[#2A2A2E] text-[#555558] cursor-not-allowed'
            }`}
          >
            {loading ? 'PROCESSING BASEBAND...' : 'INITIALIZE DSP DECODER'}
          </button>
        </div>
      </Card>

      {/* Recent Baseband Operations Card */}
      <Card
        title="Recent Baseband Analysis Operations"
        glyph="🕒"
        topRight={
          <button
            onClick={() => setRecentOps([])}
            className="text-[11px] font-mono text-[#F5A623] hover:underline"
          >
            Clear History
          </button>
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-[#2A2A2E] text-[#8E8E93] text-[10px] uppercase">
                <th className="py-2 px-3">Filename</th>
                <th className="py-2 px-3">Timestamp</th>
                <th className="py-2 px-3 text-center">Sample Rate</th>
                <th className="py-2 px-3 text-right">File Size</th>
                <th className="py-2 px-3 text-center">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#2A2A2E]/40">
              {recentOps.length === 0 ? (
                <tr>
                  <td colSpan="5" className="py-6 text-center text-[#555558]">
                    No recent operations found
                  </td>
                </tr>
              ) : (
                recentOps.map((op, idx) => (
                  <tr
                    key={idx}
                    className="hover:bg-[#1A1A1E]/80 transition-colors group cursor-pointer"
                    onClick={() => {
                      if (op.is_synthetic) {
                        handleLoadSample(op.filename);
                      }
                    }}
                  >
                    <td className="py-2.5 px-3 text-white flex items-center space-x-2 font-medium">
                      <span className="text-[#F5A623]">📄</span>
                      <span className="group-hover:text-[#F5A623] transition-colors truncate max-w-[200px]">
                        {op.filename}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-[#8E8E93]">{op.timestamp}</td>
                    <td className="py-2.5 px-3 text-[#F5A623] text-center font-bold">
                      {op.sample_rate_str}
                    </td>
                    <td className="py-2.5 px-3 text-[#8E8E93] text-right">
                      {op.filesize_str}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <button
                        title="Load signal"
                        className="w-6 h-6 rounded-full bg-[#231B0D] border border-[#F5A623]/30 flex items-center justify-center text-[#F5A623] hover:bg-[#F5A623] hover:text-[#0A0A0B] transition-colors mx-auto"
                      >
                        <Play className="w-3 h-3 fill-current ml-0.5" />
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
