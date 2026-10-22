import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import ImageDetection from './pages/ImageDetection';
import VideoAnalysis from './pages/VideoAnalysis';
import GradCAMViewer from './pages/GradCAMViewer';
import ForensicReport from './pages/ForensicReport';
import AnalysisHistory from './pages/AnalysisHistory';
import { checkBackendHealth } from './services/api';

// Canonical benchmark samples clearly labeled as demo data for immediate interface evaluation
const BENCHMARK_SEED_RECORDS = [
  {
    evidenceId: 'FFPP-C23-071',
    filename: '071_054_manipulated_sample.mp4',
    mediaType: 'video',
    fileSizeBytes: 4210540,
    sha256: 'ac5a0539ec24b8cfefb19b5a1f6c0aae9b75239a7c0202abe2e8973cb326b9f1',
    verdict: 'inconclusive',
    probability: 0.5,
    timestamp: '2026-10-22T08:00:00Z',
    isMock: true,
  },
  {
    evidenceId: 'FFHQ-REAL-001',
    filename: 'ffhq_authentic_reference.png',
    mediaType: 'image',
    fileSizeBytes: 184512,
    sha256: '9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08',
    verdict: 'inconclusive',
    probability: 0.5,
    timestamp: '2026-10-22T08:15:00Z',
    isMock: true,
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [backendStatus, setBackendStatus] = useState({ connected: false, data: null });
  const [latestAnalysis, setLatestAnalysis] = useState(null);

  // Load history from localStorage or seed with benchmark reference records
  const [history, setHistory] = useState(() => {
    try {
      const saved = localStorage.getItem('forensics_history');
      if (saved) {
        const parsed = JSON.parse(saved);
        // Normalize any saved records: if mock, ensure verdict is strictly inconclusive
        return parsed.map((item) => {
          if (item.isMock || item.data?.is_mock_prediction) {
            return {
              ...item,
              verdict: 'inconclusive',
              isMock: true,
            };
          }
          return item;
        });
      }
    } catch (e) {
      // Fallback
    }
    return BENCHMARK_SEED_RECORDS;
  });

  // Check backend health on mount and ping every 8 seconds
  useEffect(() => {
    let isMounted = true;

    const pingBackend = async () => {
      const status = await checkBackendHealth();
      if (isMounted) {
        setBackendStatus(status);
      }
    };

    pingBackend();
    const interval = setInterval(pingBackend, 8000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleAnalysisComplete = (newRecord) => {
    setHistory((prev) => {
      const updated = [newRecord, ...prev];
      try {
        localStorage.setItem('forensics_history', JSON.stringify(updated));
      } catch (e) {
        // Ignore quota error
      }
      return updated;
    });
  };

  return (
    <div className="min-h-screen bg-navy-950 flex flex-col font-sans">
      <Navbar backendStatus={backendStatus} />

      <div className="flex-1 flex overflow-hidden">
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />

        <main className="flex-1 overflow-y-auto p-6 md:p-8 bg-gradient-to-b from-navy-900/30 to-navy-950">
          <div className="max-w-7xl mx-auto">
            {activeTab === 'dashboard' && (
              <Dashboard
                setActiveTab={setActiveTab}
                backendStatus={backendStatus}
                recentHistory={history}
              />
            )}

            {activeTab === 'image' && (
              <ImageDetection
                backendStatus={backendStatus}
                onAnalysisComplete={handleAnalysisComplete}
                setLatestAnalysis={setLatestAnalysis}
                setActiveTab={setActiveTab}
              />
            )}

            {activeTab === 'video' && (
              <VideoAnalysis
                backendStatus={backendStatus}
                onAnalysisComplete={handleAnalysisComplete}
                setLatestAnalysis={setLatestAnalysis}
                setActiveTab={setActiveTab}
              />
            )}

            {activeTab === 'explainability' && (
              <GradCAMViewer
                latestAnalysis={latestAnalysis}
                setActiveTab={setActiveTab}
              />
            )}

            {activeTab === 'report' && (
              <ForensicReport
                latestAnalysis={latestAnalysis}
                setActiveTab={setActiveTab}
              />
            )}

            {activeTab === 'history' && (
              <AnalysisHistory
                history={history}
                setHistory={setHistory}
                setLatestAnalysis={setLatestAnalysis}
                setActiveTab={setActiveTab}
              />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
