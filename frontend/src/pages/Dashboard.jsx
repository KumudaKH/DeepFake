import React from 'react';
import {
  ShieldAlert,
  Image as ImageIcon,
  Video,
  FileCheck,
  AlertTriangle,
  ArrowRight,
  Cpu,
  Layers,
  Database,
  Info
} from 'lucide-react';
import { StatusBadge } from '../components/StatusBadge';

export default function Dashboard({ setActiveTab, backendStatus, recentHistory = [] }) {
  const isConnected = backendStatus?.connected;
  const isMock = backendStatus?.data?.is_model_trained === false;

  const totalAnalyses = recentHistory.length;
  const imageAnalyses = recentHistory.filter(h => h.mediaType === 'image').length;
  const videoAnalyses = recentHistory.filter(h => h.mediaType === 'video').length;
  const flaggedCount = recentHistory.filter(h => !h.isMock && (h.verdict?.toLowerCase() === 'fake' || h.verdict?.toLowerCase() === 'manipulated')).length;
  const demoCount = recentHistory.filter(h => h.isMock || h.verdict?.toLowerCase() === 'inconclusive').length;

  return (
    <div className="space-y-6">
      {/* Top Banner if Backend Disconnected */}
      {!isConnected && (
        <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/10 text-rose-300 flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-rose-400 mt-0.5 shrink-0" />
          <div className="text-xs space-y-1">
            <p className="font-semibold text-rose-200">FastAPI Forensic Backend Not Connected</p>
            <p className="text-rose-300/90 leading-relaxed">
              The frontend is currently unable to communicate with <code className="font-mono bg-rose-950/60 px-1 py-0.5 rounded">http://localhost:8000</code>.
              Start the backend using <code className="font-mono bg-rose-950/60 px-1 py-0.5 rounded">.\.venv\Scripts\python.exe -m uvicorn src.backend.app.main:app --port 8000</code> to enable live neural inference and Grad-CAM generation.
            </p>
          </div>
        </div>
      )}

      {/* Model State Alert if Running Mock */}
      {isConnected && isMock && (
        <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/10 text-amber-300 flex items-start gap-3">
          <Info className="w-5 h-5 text-amber-400 mt-0.5 shrink-0" />
          <div className="text-xs space-y-1">
            <p className="font-semibold text-amber-200">System Running in Interface Demonstration Mode</p>
            <p className="text-amber-300/90 leading-relaxed">
              No trained weights were found at <code className="font-mono bg-amber-950/60 px-1 py-0.5 rounded">models/image/model_best_fp16.pt</code>.
              The pipeline executes with an uninitialized architecture mock so you can verify uploads, face tracking, and Grad-CAM extraction. Predictions are clearly tagged as <span className="font-mono font-bold text-white">[DEMO]</span> and must not be used for legal interpretation.
            </p>
          </div>
        </div>
      )}

      {/* Model State Alert if Trained Checkpoint Active */}
      {isConnected && !isMock && (
        <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/10 text-emerald-300 flex items-start gap-3">
          <CheckCircle2 className="w-5 h-5 text-emerald-400 mt-0.5 shrink-0" />
          <div className="text-xs space-y-1">
            <p className="font-semibold text-emerald-200">Trained Deepfake Model Active</p>
            <p className="text-emerald-300/90 leading-relaxed">
              Trained EfficientNet-B4 weights loaded from <code className="font-mono bg-emerald-950/60 px-1 py-0.5 rounded">{backendStatus?.data?.checkpoint_path || 'models/image/model_best_fp16.pt'}</code>. System running verified neural inference.
            </p>
          </div>
        </div>
      )}

      {/* Metrics Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Total Forensics Scans</span>
            <FileCheck className="w-4 h-4 text-cyan-400" />
          </div>
          <p className="text-2xl font-bold text-white font-mono mt-2">{totalAnalyses}</p>
          <span className="text-[11px] text-slate-400 mt-1 block">Active workspace sessions</span>
        </div>

        <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Image Forgery Scans</span>
            <ImageIcon className="w-4 h-4 text-brand-lightBlue" />
          </div>
          <p className="text-2xl font-bold text-white font-mono mt-2">{imageAnalyses}</p>
          <span className="text-[11px] text-slate-400 mt-1 block">Single-frame crops</span>
        </div>

        <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">Video Forgery Scans</span>
            <Video className="w-4 h-4 text-purple-400" />
          </div>
          <p className="text-2xl font-bold text-white font-mono mt-2">{videoAnalyses}</p>
          <span className="text-[11px] text-slate-400 mt-1 block">16-frame temporal pooling</span>
        </div>

        <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-medium">{isMock ? 'Demo Inconclusive Scans' : 'Manipulated Verdicts'}</span>
            <ShieldAlert className={`w-4 h-4 ${isMock ? 'text-amber-400' : 'text-rose-400'}`} />
          </div>
          <p className={`text-2xl font-bold font-mono mt-2 ${isMock ? 'text-amber-400' : 'text-rose-400'}`}>
            {isMock ? demoCount : flaggedCount}
          </p>
          <span className="text-[11px] text-slate-400 mt-1 block">
            {isMock ? 'Scores unavailable (No checkpoint)' : 'Flagged above threshold (τ=0.50)'}
          </span>
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (2/3): Action Quick Links & Architecture */}
        <div className="lg:col-span-2 space-y-6">
          <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/40">
            <h2 className="text-sm font-semibold text-white tracking-wide uppercase flex items-center gap-2 mb-3">
              <Layers className="w-4 h-4 text-cyan-400" />
              Forensic Workflow Launchpad
            </h2>
            <p className="text-xs text-slate-400 mb-4 leading-relaxed">
              Select an investigation module below to inspect media files, extract facial crops, compute Grad-CAM visual heatmaps, and produce audit reports.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                onClick={() => setActiveTab('image')}
                className="p-4 rounded-lg border border-navy-750 bg-navy-950/60 hover:border-brand-blue/60 hover:bg-navy-850/80 transition-all text-left group"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="p-2 rounded bg-brand-blue/15 text-brand-lightBlue">
                    <ImageIcon className="w-4 h-4" />
                  </div>
                  <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 transition-colors" />
                </div>
                <h3 className="text-sm font-medium text-white group-hover:text-cyan-300">Image Authentication</h3>
                <p className="text-[11px] text-slate-400 mt-1">Single-crop facial verification with continuous logit calibration.</p>
              </button>

              <button
                onClick={() => setActiveTab('video')}
                className="p-4 rounded-lg border border-navy-750 bg-navy-950/60 hover:border-brand-blue/60 hover:bg-navy-850/80 transition-all text-left group"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="p-2 rounded bg-purple-500/15 text-purple-400">
                    <Video className="w-4 h-4" />
                  </div>
                  <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-purple-400 transition-colors" />
                </div>
                <h3 className="text-sm font-medium text-white group-hover:text-purple-300">Video Forgery Tracking</h3>
                <p className="text-[11px] text-slate-400 mt-1">Uniform 16-frame temporal sampling, YuNet IoU tracking, and keyframe isolation.</p>
              </button>

              <button
                onClick={() => setActiveTab('explainability')}
                className="p-4 rounded-lg border border-navy-750 bg-navy-950/60 hover:border-brand-blue/60 hover:bg-navy-850/80 transition-all text-left group"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="p-2 rounded bg-cyan-500/15 text-cyan-400">
                    <Cpu className="w-4 h-4" />
                  </div>
                  <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-cyan-400 transition-colors" />
                </div>
                <h3 className="text-sm font-medium text-white group-hover:text-cyan-300">Grad-CAM Saliency Maps</h3>
                <p className="text-[11px] text-slate-400 mt-1">Spatial attribution overlays targeting terminal convolutional feature layers.</p>
              </button>

              <button
                onClick={() => setActiveTab('report')}
                className="p-4 rounded-lg border border-navy-750 bg-navy-950/60 hover:border-brand-blue/60 hover:bg-navy-850/80 transition-all text-left group"
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="p-2 rounded bg-emerald-500/15 text-emerald-400">
                    <FileCheck className="w-4 h-4" />
                  </div>
                  <ArrowRight className="w-4 h-4 text-slate-600 group-hover:text-emerald-400 transition-colors" />
                </div>
                <h3 className="text-sm font-medium text-white group-hover:text-emerald-300">Forensic Examination Sheets</h3>
                <p className="text-[11px] text-slate-400 mt-1">Export cryptographic SHA-256 evidence records in JSON and HTML format.</p>
              </button>
            </div>
          </div>

          {/* Forensic System Architecture Specs */}
          <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/40 space-y-3">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Database className="w-3.5 h-3.5 text-cyan-400" />
              Verified Forensic Pipeline Architecture
            </h3>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
              <div className="p-3 bg-navy-950/70 border border-navy-800 rounded-lg">
                <span className="text-[10px] text-slate-400 block font-mono">BACKBONE</span>
                <span className="font-semibold text-white">EfficientNet-B4</span>
              </div>
              <div className="p-3 bg-navy-950/70 border border-navy-800 rounded-lg">
                <span className="text-[10px] text-slate-400 block font-mono">FEATURE LAYER</span>
                <span className="font-semibold text-white">conv_head (1,792d)</span>
              </div>
              <div className="p-3 bg-navy-950/70 border border-navy-800 rounded-lg">
                <span className="text-[10px] text-slate-400 block font-mono">TEMPORAL POOL</span>
                <span className="font-semibold text-white">Mean Logits (k=16)</span>
              </div>
              <div className="p-3 bg-navy-950/70 border border-navy-800 rounded-lg">
                <span className="text-[10px] text-slate-400 block font-mono">FACE DETECTOR</span>
                <span className={`font-semibold ${backendStatus?.data?.yunet_detector_available ? 'text-white' : 'text-amber-400'}`}>
                  {backendStatus?.data?.yunet_detector_available ? 'OpenCV YuNet ONNX' : 'YuNet (Missing)'}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column (1/3): Recent Examination Activity */}
        <div className="space-y-4">
          <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/60 h-full flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                  Recent Examination Items
                </h3>
                <button
                  onClick={() => setActiveTab('history')}
                  className="text-[11px] text-brand-lightBlue hover:underline"
                >
                  View All
                </button>
              </div>

              {recentHistory.length === 0 ? (
                <div className="py-12 text-center text-slate-400 space-y-2">
                  <FileCheck className="w-8 h-8 text-slate-600 mx-auto" />
                  <p className="text-xs">No media files examined in this session.</p>
                  <p className="text-[11px] text-slate-400">Upload an image or video to run inference.</p>
                </div>
              ) : (
                <div className="space-y-2.5">
                  {recentHistory.slice(0, 5).map((item, idx) => (
                    <div
                      key={idx}
                      className="p-3 bg-navy-950/60 border border-navy-800 rounded-lg text-xs space-y-1.5"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono font-medium text-slate-200 truncate max-w-[160px]">
                          {item.filename}
                        </span>
                        <StatusBadge label={item.verdict} isMock={item.isMock} />
                      </div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono">
                        <span>
                          {item.isMock ? 'Score unavailable (Demo)' : `P(Fake): ${Math.round((item.probability || 0) * 100)}%`}
                        </span>
                        <span className="text-[10px] text-slate-400">
                          {item.timestamp ? new Date(item.timestamp).toLocaleTimeString() : 'Recent'}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="pt-4 border-t border-navy-800/80 text-[11px] text-slate-400 leading-relaxed mt-4">
              <strong className="text-slate-300">Evidentiary Traceability:</strong> All outputs record SHA-256 digests and frame timestamps to ensure forensic repeatability.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
