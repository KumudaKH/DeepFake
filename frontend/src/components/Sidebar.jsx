import React from 'react';
import {
  LayoutDashboard,
  Image as ImageIcon,
  Video,
  Eye,
  FileText,
  History,
  ShieldAlert,
} from 'lucide-react';

export default function Sidebar({ activeTab, setActiveTab }) {
  const navItems = [
    { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { id: 'image', label: 'Image Detection', icon: ImageIcon },
    { id: 'video', label: 'Video Analysis', icon: Video },
    { id: 'explainability', label: 'Grad-CAM Saliency', icon: Eye },
    { id: 'report', label: 'Forensic Reports', icon: FileText },
    { id: 'history', label: 'Analysis History', icon: History },
  ];

  return (
    <aside className="w-64 border-r border-navy-800 bg-navy-900/60 p-4 flex flex-col justify-between">
      <div>
        <div className="px-3 py-2 mb-4">
          <p className="text-[10px] font-mono uppercase tracking-wider text-slate-500 font-semibold">
            Digital Forensics Workstation
          </p>
        </div>

        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActiveTab(item.id)}
                className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all ${
                  isActive
                    ? 'bg-brand-blue/15 text-brand-lightBlue border border-brand-blue/40 shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-navy-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="p-3 rounded-lg bg-navy-950/70 border border-navy-800 text-[11px] text-slate-400 space-y-1.5">
        <div className="flex items-center gap-1.5 text-cyan-400 font-medium">
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>Forensic Standard</span>
        </div>
        <p className="text-[10px] text-slate-400 leading-relaxed">
          Predictions are probabilistic neural inferences. Saliency maps do NOT establish certified tampering boundaries.
        </p>
      </div>
    </aside>
  );
}
