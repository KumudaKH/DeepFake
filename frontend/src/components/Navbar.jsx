import React from 'react';
import { Shield, Activity, AlertTriangle, CheckCircle2, Server } from 'lucide-react';

export default function Navbar({ backendStatus }) {
  const isConnected = backendStatus?.connected;
  const isMock = backendStatus?.data?.is_model_trained === false;

  return (
    <header className="h-16 border-b border-navy-800 bg-navy-900/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-brand-blue/10 border border-brand-blue/30 rounded-lg text-brand-lightBlue">
          <Shield className="w-5 h-5" />
        </div>
        <div>
          <h1 className="font-semibold text-sm tracking-wide text-white uppercase flex items-center gap-2">
            Multimodal Deepfake Forensics System
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-brand-blue/20 text-cyan-400 border border-brand-blue/30 font-mono">
              EFFICIENTNET-B4
            </span>
          </h1>
          <p className="text-xs text-slate-400">Explainable Digital Media Authentication & Forensics</p>
        </div>
      </div>

      <div className="flex items-center gap-4">
        {/* Backend Connectivity Heartbeat */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full border bg-navy-950/60 text-xs">
          {isConnected ? (
            <>
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-emerald-400 font-medium">Backend Connected</span>
              {isMock ? (
                <span className="ml-1 px-2 py-0.5 text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/30 rounded font-mono font-medium">
                  Demo Mode (Mock Weights)
                </span>
              ) : (
                <span className="ml-1 px-2 py-0.5 text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 rounded font-mono font-medium flex items-center gap-1">
                  <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                  Trained Model
                </span>
              )}
            </>
          ) : (
            <>
              <span className="w-2 h-2 rounded-full bg-rose-500"></span>
              <span className="text-rose-400 font-medium flex items-center gap-1">
                <AlertTriangle className="w-3 h-3" /> Backend Disconnected (localhost:8000)
              </span>
            </>
          )}
        </div>

        <div className="text-xs text-slate-400 hidden sm:flex items-center gap-1 font-mono">
          <Server className="w-3.5 h-3.5 text-slate-500" />
          <span>Device: {backendStatus?.data?.device?.toUpperCase() || 'CPU'}</span>
        </div>
      </div>
    </header>
  );
}
