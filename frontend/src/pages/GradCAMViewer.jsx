import React, { useState } from 'react';
import {
  Eye,
  Layers,
  Sliders,
  AlertTriangle,
  Info,
  CheckCircle2,
  Sparkles,
  Cpu,
  ArrowRight,
  ShieldAlert,
} from 'lucide-react';

export default function GradCAMViewer({ latestAnalysis, setActiveTab }) {
  const [alphaSlider, setAlphaSlider] = useState(40);

  // Extract explainability assets from latest analysis (supports both direct API response and wrapped history records)
  const analysisObj = latestAnalysis?.data || latestAnalysis;
  const isImage = analysisObj?.mediaType === 'image' || analysisObj?.prediction || !analysisObj?.video_prediction;
  const isVideo = Boolean(analysisObj?.video_prediction || analysisObj?.mediaType === 'video');

  const originalUrl = isImage
    ? (analysisObj?.explainability?.original_data_url || analysisObj?.original_data_url)
    : (analysisObj?.keyframe_explainability?.crop_data_url || analysisObj?.crop_data_url);

  const heatmapUrl = isImage
    ? (analysisObj?.explainability?.heatmap_data_url || analysisObj?.heatmap_data_url)
    : (analysisObj?.keyframe_explainability?.heatmap_data_url || analysisObj?.heatmap_data_url);

  const overlayUrl = isImage
    ? (analysisObj?.explainability?.overlay_data_url || analysisObj?.overlay_data_url)
    : (analysisObj?.keyframe_explainability?.overlay_data_url || analysisObj?.overlay_data_url);

  const targetLayer = isImage
    ? (analysisObj?.explainability?.target_layer_name || 'conv_head')
    : (analysisObj?.keyframe_explainability?.target_layer_name || 'conv_head');

  const evidenceId = analysisObj?.evidence_id || analysisObj?.evidenceId || latestAnalysis?.evidence_id || latestAnalysis?.evidenceId;
  const probability = isImage
    ? (analysisObj?.prediction?.probability ?? analysisObj?.probability)
    : (analysisObj?.video_prediction?.probability ?? analysisObj?.probability);
  const isMock = analysisObj?.is_mock_prediction !== undefined
    ? analysisObj.is_mock_prediction
    : (latestAnalysis?.isMock !== undefined ? latestAnalysis.isMock : true);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Eye className="w-5 h-5 text-cyan-400" />
            Grad-CAM Spatial Explainability Inspector
          </h2>
          <p className="text-xs text-slate-400">
            Gradient-weighted Class Activation Mapping (Selvaraju et al.) visualizes feature maps influencing EfficientNet-B4 predictions.
          </p>
        </div>

        {evidenceId && (
          <span className="font-mono text-xs px-2.5 py-1 rounded bg-navy-900 border border-navy-800 text-cyan-400">
            Target Evidence: {evidenceId}
          </span>
        )}
      </div>

      {!overlayUrl ? (
        /* Empty State */
        <div className="p-12 rounded-xl border border-navy-800 bg-navy-900/40 text-center space-y-4 max-w-xl mx-auto my-8">
          <div className="p-4 rounded-full bg-brand-blue/10 text-cyan-400 w-16 h-16 mx-auto flex items-center justify-center">
            <Eye className="w-8 h-8" />
          </div>
          <h3 className="text-base font-semibold text-white">No Explainability Maps Generated Yet</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Grad-CAM heatmaps are computed dynamically during image or video inference. Upload a facial image in the Image Detection tab or analyze a video to inspect spatial saliency maps.
          </p>
          <div className="flex justify-center gap-3 pt-2">
            <button
              onClick={() => setActiveTab('image')}
              className="py-2 px-4 rounded-lg bg-brand-blue hover:bg-brand-accent text-white text-xs font-medium flex items-center gap-2 transition-colors"
            >
              <span>Go to Image Detection</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTab('video')}
              className="py-2 px-4 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-medium flex items-center gap-2 transition-colors"
            >
              <span>Go to Video Analysis</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      ) : (
        /* Full 3-View Explainability Panel */
        <div className="space-y-6">
          {/* Prominent Demo Notice if Mock Model */}
          {isMock && (
            <div className="p-3.5 rounded-xl border border-amber-500/40 bg-amber-500/10 text-xs text-amber-300 flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-0.5">
                <p className="font-semibold text-amber-200">Demonstration Mode (Mock Weights)</p>
                <p className="text-amber-300/90 text-[11px] leading-relaxed">
                  Grad-CAM activations are computed using an uninitialized architecture mock to demonstrate gradient backpropagation and spatial projection. They do not represent authentic forgery artifacts.
                </p>
              </div>
            </div>
          )}

          {/* Controls Bar */}
          <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="text-xs text-slate-300 font-medium flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-cyan-400" />
                Target Conv Layer:
              </span>
              <span className="font-mono text-xs px-2 py-0.5 rounded bg-brand-blue/20 text-cyan-300 border border-brand-blue/30 font-semibold">
                {targetLayer} (1,792 channels, 7x7 grid)
              </span>
            </div>

            <div className="flex items-center gap-3 text-xs w-full sm:w-auto">
              <span className="text-slate-400 font-medium flex items-center gap-1">
                <Sliders className="w-3.5 h-3.5 text-slate-400" />
                Display Blend Opacity:
              </span>
              <input
                type="range"
                min="0"
                max="100"
                value={alphaSlider}
                onChange={(e) => setAlphaSlider(parseInt(e.target.value))}
                className="w-32 accent-cyan-400 cursor-pointer"
              />
              <span className="font-mono text-cyan-400 w-8">{alphaSlider}%</span>
            </div>
          </div>

          {/* 3-View Triplet Comparison */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Panel 1: Original Facial Crop */}
            <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60 space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300 uppercase tracking-wide">1. Original Crop</span>
                <span className="text-[10px] text-slate-400 font-mono">RGB Input (380x380 model input)</span>
              </div>
              <div className="aspect-square bg-navy-950 rounded-lg overflow-hidden border border-navy-800 flex items-center justify-center p-1">
                <img
                  src={originalUrl}
                  alt="Original Facial Crop"
                  className="w-full h-full object-contain rounded"
                />
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Preprocessed facial crop before neural feature extraction. Preserved defensively without modification.
              </p>
            </div>

            {/* Panel 2: Raw Heatmap */}
            <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60 space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300 uppercase tracking-wide">2. Saliency Heatmap</span>
                <span className="text-[10px] text-cyan-400 font-mono">Normalized [0.0, 1.0]</span>
              </div>
              <div className="aspect-square bg-navy-950 rounded-lg overflow-hidden border border-navy-800 flex items-center justify-center p-1">
                <img
                  src={heatmapUrl}
                  alt="Grad-CAM Activation Heatmap"
                  className="w-full h-full object-contain rounded"
                />
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                Gradient-weighted combination of 1,792 feature activations passed through a positive-only ReLU filter.
              </p>
            </div>

            {/* Panel 3: Blended Overlay */}
            <div className="p-4 rounded-xl border border-navy-800 bg-navy-900/60 space-y-3">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300 uppercase tracking-wide">3. Forensic Overlay</span>
                <span className="text-[10px] text-brand-lightBlue font-mono">OpenCV JET Colormap</span>
              </div>
              <div className="aspect-square bg-navy-950 rounded-lg overflow-hidden border border-navy-800 flex items-center justify-center p-1 relative">
                <img
                  src={overlayUrl}
                  alt="Blended Visual Overlay"
                  className="w-full h-full object-contain rounded transition-opacity"
                  style={{ opacity: 0.2 + (alphaSlider / 100) * 0.8 }}
                />
              </div>
              <p className="text-[11px] text-slate-400 leading-relaxed">
                OpenCV Jet overlay blended with the original crop. High-activation warm regions drove the forgery score.
              </p>
            </div>
          </div>

          {/* Educational Interpretation Box */}
          <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/40 space-y-3">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <Info className="w-4 h-4 text-cyan-400" />
              How to Interpret Grad-CAM Forensic Saliency Maps
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs text-slate-300">
              <div className="space-y-1.5 p-3 rounded-lg bg-navy-950/60 border border-navy-800">
                <p className="font-semibold text-cyan-300 flex items-center gap-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  What the Heatmap Shows
                </p>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Warm regions (red, orange, yellow) indicate spatial receptive fields that positively increased the model's forgery logit score $s$. Cool regions (blue) had negligible or negative influence on the manipulation decision.
                </p>
              </div>

              <div className="space-y-1.5 p-3 rounded-lg bg-navy-950/60 border border-navy-800">
                <p className="font-semibold text-rose-300 flex items-center gap-1.5">
                  <ShieldAlert className="w-3.5 h-3.5" />
                  Why Saliency is NOT Proof of Tampering
                </p>
                <p className="text-slate-400 text-[11px] leading-relaxed">
                  Grad-CAM reflects internal neural attention, NOT physical tampering boundaries. Compression noise, unusual lighting, or asymmetric facial features can attract high gradient weight without representing synthetic generation.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
