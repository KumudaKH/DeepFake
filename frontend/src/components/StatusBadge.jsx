import React from 'react';
import { AlertCircle, CheckCircle, HelpCircle } from 'lucide-react';

export function StatusBadge({ label, isMock = false }) {
  const norm = (label || '').toLowerCase();

  // If in demo mode or inconclusive, NEVER present authoritative REAL or FAKE classifications
  if (isMock || norm === 'inconclusive' || norm.includes('demo') || norm === 'inconclusive_demo') {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30">
        <HelpCircle className="w-3.5 h-3.5 text-amber-400" />
        <span>INCONCLUSIVE — DEMO MODE</span>
      </span>
    );
  }

  if (norm === 'fake' || norm === 'manipulated') {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-rose-500/15 text-rose-400 border border-rose-500/30">
        <AlertCircle className="w-3.5 h-3.5" />
        <span>MANIPULATED / FAKE</span>
      </span>
    );
  }

  if (norm === 'real' || norm === 'authentic') {
    return (
      <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
        <CheckCircle className="w-3.5 h-3.5" />
        <span>AUTHENTIC / REAL</span>
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-slate-500/15 text-slate-300 border border-slate-500/30">
      <HelpCircle className="w-3.5 h-3.5" />
      <span>{label ? label.toUpperCase() : 'INCONCLUSIVE'}</span>
    </span>
  );
}

export function ConfidenceMeter({ probability, threshold = 0.5, isMock = false }) {
  if (isMock || probability === null || probability === undefined) {
    return (
      <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-xs text-amber-300">
        No trained checkpoint is loaded. Classification score unavailable.
      </div>
    );
  }

  const pct = Math.round(probability * 100);
  const isFake = probability >= threshold;

  return (
    <div className="space-y-2">
      <div className="flex justify-between items-baseline text-xs">
        <span className="text-slate-400 font-medium">
          Probabilistic Manipulation Score P(Fake):
        </span>
        <span className="font-mono font-semibold text-white text-sm">
          {`${pct}%`}
        </span>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-navy-950 h-3 rounded-full overflow-hidden p-0.5 border border-navy-800">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            isFake
              ? 'bg-gradient-to-r from-amber-500 to-rose-500'
              : 'bg-gradient-to-r from-teal-500 to-emerald-500'
          }`}
          style={{ width: `${Math.max(4, Math.min(100, pct))}%` }}
        />
      </div>

      <div className="flex justify-between text-[10px] text-slate-400 font-mono">
        <span>0.0 (Authentic Human)</span>
        <span className="text-cyan-400">Decision Threshold: {threshold}</span>
        <span>1.0 (Manipulated Synthesis)</span>
      </div>
    </div>
  );
}
