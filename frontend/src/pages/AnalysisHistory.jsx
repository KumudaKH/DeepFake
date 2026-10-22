import React, { useState } from 'react';
import {
  History,
  Trash2,
  Search,
  ExternalLink,
  Eye,
  FileCheck,
  ShieldCheck,
  Filter,
} from 'lucide-react';
import { StatusBadge } from '../components/StatusBadge';

export default function AnalysisHistory({
  history = [],
  setHistory,
  setLatestAnalysis,
  setActiveTab,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [verdictFilter, setVerdictFilter] = useState('all');

  const filteredHistory = history.filter((item) => {
    const matchesSearch =
      item.filename?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.evidenceId?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.sha256?.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesVerdict =
      verdictFilter === 'all' ||
      item.verdict?.toLowerCase() === verdictFilter.toLowerCase();

    return matchesSearch && matchesVerdict;
  });

  const handleClearHistory = () => {
    if (window.confirm('Clear all session analysis history?')) {
      setHistory([]);
      try {
        localStorage.removeItem('forensics_history');
      } catch (e) {
        // Ignore localStorage error
      }
    }
  };

  const handleViewItem = (item) => {
    const rawData = item.data || item;
    const isMock = item.isMock !== undefined ? item.isMock : (rawData.is_mock_prediction ?? true);
    setLatestAnalysis({
      ...rawData,
      is_mock_prediction: isMock,
      isMock: isMock,
      verdict: isMock ? 'inconclusive' : (rawData.verdict || item.verdict),
    });
    setActiveTab('report');
  };

  const handleViewExplain = (item) => {
    const rawData = item.data || item;
    const isMock = item.isMock !== undefined ? item.isMock : (rawData.is_mock_prediction ?? true);
    setLatestAnalysis({
      ...rawData,
      is_mock_prediction: isMock,
      isMock: isMock,
      verdict: isMock ? 'inconclusive' : (rawData.verdict || item.verdict),
    });
    setActiveTab('explainability');
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <History className="w-5 h-5 text-cyan-400" />
            Forensic Examination History
          </h2>
          <p className="text-xs text-slate-400">
            Log of evidence media examined during this session.
          </p>
        </div>

        {history.length > 0 && (
          <button
            onClick={handleClearHistory}
            className="py-1.5 px-3 rounded-lg border border-navy-750 text-slate-400 hover:text-rose-400 hover:border-rose-500/40 text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            <span>Clear Session Log</span>
          </button>
        )}
      </div>

      {/* Filter and Search Bar */}
      <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60 flex flex-col sm:flex-row items-center justify-between gap-3">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            placeholder="Search filename, evidence ID, or SHA-256..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full pl-9 pr-3 py-1.5 bg-navy-950 border border-navy-800 rounded-lg text-xs text-white placeholder-slate-400 focus:outline-none focus:border-brand-blue"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Filter className="w-3.5 h-3.5 text-slate-400" />
          <span className="text-xs text-slate-400 font-medium">Verdict:</span>
          <select
            value={verdictFilter}
            onChange={(e) => setVerdictFilter(e.target.value)}
            className="bg-navy-950 border border-navy-800 rounded-lg text-xs text-slate-200 py-1.5 px-2.5 focus:outline-none focus:border-brand-blue"
          >
            <option value="all">All Verdicts</option>
            <option value="inconclusive">Inconclusive (Demo Mode)</option>
            <option value="fake">Manipulated (Fake)</option>
            <option value="real">Authentic (Real)</option>
          </select>
        </div>
      </div>

      {/* History Table */}
      <div className="rounded-xl border border-navy-800 bg-navy-900/60 overflow-hidden">
        {filteredHistory.length === 0 ? (
          <div className="p-12 text-center text-slate-400 space-y-2">
            <History className="w-8 h-8 text-slate-600 mx-auto" />
            <p className="text-sm font-medium text-slate-400">No Forensic Records Found</p>
            <p className="text-xs text-slate-400">
              {history.length === 0
                ? 'Ingest media files in Image Detection or Video Analysis to populate examination records.'
                : 'No records match your active search filter.'}
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-navy-950/80 text-slate-400 font-mono text-[11px] border-b border-navy-800">
                <tr>
                  <th className="py-3 px-4">Evidence ID</th>
                  <th className="py-3 px-4">Filename</th>
                  <th className="py-3 px-4">Media</th>
                  <th className="py-3 px-4">Verdict</th>
                  <th className="py-3 px-4">Confidence / Score</th>
                  <th className="py-3 px-4">SHA-256 (Truncated)</th>
                  <th className="py-3 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-navy-800/80">
                {filteredHistory.map((item, idx) => (
                  <tr key={idx} className="hover:bg-navy-850/40 transition-colors">
                    <td className="py-3 px-4 font-mono font-medium text-cyan-400">
                      {item.evidenceId}
                      {item.isMock && (
                        <span className="ml-1 text-[9px] px-1 py-0.2 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                          DEMO
                        </span>
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-200 max-w-[180px] truncate">
                      {item.filename}
                    </td>
                    <td className="py-3 px-4 uppercase text-[10px] font-mono text-slate-400">
                      {item.mediaType || 'Image'}
                    </td>
                    <td className="py-3 px-4">
                      <StatusBadge label={item.verdict} isMock={item.isMock} />
                    </td>
                    <td className="py-3 px-4 font-mono font-semibold text-slate-200">
                      {item.isMock ? (
                        <span className="text-amber-300 font-medium text-[11px]">Score unavailable (Demo)</span>
                      ) : (
                        `${Math.round((item.probability || 0) * 100)}%`
                      )}
                    </td>
                    <td className="py-3 px-4 font-mono text-slate-400 text-[10px]">
                      {item.sha256 ? `${item.sha256.slice(0, 10)}...${item.sha256.slice(-6)}` : 'N/A'}
                    </td>
                    <td className="py-3 px-4 text-right space-x-2">
                      <button
                        onClick={() => handleViewExplain(item)}
                        title="View Saliency Map"
                        className="p-1 rounded hover:bg-navy-800 text-slate-400 hover:text-cyan-400 transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5 inline" />
                      </button>
                      <button
                        onClick={() => handleViewItem(item)}
                        title="View Forensic Report"
                        className="p-1 rounded hover:bg-navy-800 text-slate-400 hover:text-emerald-400 transition-colors"
                      >
                        <FileCheck className="w-3.5 h-3.5 inline" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
