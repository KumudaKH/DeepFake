import React, { useState } from 'react';
import {
  Upload,
  Image as ImageIcon,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Layers,
  ArrowRight,
  RefreshCw,
  Eye,
  FileCheck,
  ShieldCheck,
} from 'lucide-react';
import { analyzeImage, computeClientSha256 } from '../services/api';
import { StatusBadge, ConfidenceMeter } from '../components/StatusBadge';

export default function ImageDetection({
  backendStatus,
  onAnalysisComplete,
  setLatestAnalysis,
  setActiveTab,
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [clientSha256, setClientSha256] = useState('');
  const [fileDimensions, setFileDimensions] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [threshold, setThreshold] = useState(0.5);

  const isConnected = backendStatus?.connected;
  const isMock = backendStatus?.data?.is_model_trained === false;

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Validate type
    const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
    if (!validTypes.includes(file.type)) {
      setError(`Unsupported file type (${file.type}). Please select a JPEG, PNG, WebP, or BMP image.`);
      return;
    }

    // Validate size (max 20 MB)
    if (file.size > 20 * 1024 * 1024) {
      setError('File size exceeds 20 MB limit.');
      return;
    }

    setError(null);
    setResult(null);
    setSelectedFile(file);

    // Create object URL for instant preview
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Read image dimensions
    const img = new Image();
    img.onload = () => {
      setFileDimensions({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.src = url;

    // Compute real cryptographic SHA-256 hash client-side
    const hash = await computeClientSha256(file);
    setClientSha256(hash);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    if (!isConnected) {
      setError('Cannot execute inference: FastAPI backend is not connected (http://localhost:8000). Please start the backend service.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const data = await analyzeImage(selectedFile, threshold);
      setResult(data);

      const historyItem = {
        evidenceId: data.evidence_id,
        filename: data.filename,
        mediaType: 'image',
        fileSizeBytes: data.file_size_bytes,
        sha256: data.sha256,
        verdict: data.prediction.predicted_label,
        probability: data.prediction.probability,
        timestamp: data.examination_timestamp,
        isMock: data.is_mock_prediction,
        data: data,
      };

      setLatestAnalysis(data);
      onAnalysisComplete(historyItem);
    } catch (err) {
      setError(err.message || 'An error occurred during image inference.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setClientSha256('');
    setFileDimensions(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <ImageIcon className="w-5 h-5 text-brand-lightBlue" />
            Single-Image Deepfake Detection
          </h2>
          <p className="text-xs text-slate-400">
            Upload facial photographs or facial crops to compute forgery logits and spatial Grad-CAM saliency.
          </p>
        </div>

        {/* Threshold Adjustment */}
        <div className="flex items-center gap-3 bg-navy-900 border border-navy-800 px-3 py-1.5 rounded-lg text-xs">
          <span className="text-slate-400 font-medium">Decision Threshold:</span>
          <span className="font-mono text-cyan-400 font-semibold">{threshold}</span>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-24 accent-brand-blue cursor-pointer"
          />
        </div>
      </div>

      {/* Main Grid: Upload & Inspection */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Dropzone & File Preview */}
        <div className="space-y-4">
          <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/60 space-y-4">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Upload className="w-4 h-4 text-cyan-400" />
              Evidence Media Ingestion
            </h3>

            {!previewUrl ? (
              <label className="border-2 border-dashed border-navy-750 hover:border-brand-blue/60 bg-navy-950/60 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all group">
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp,image/bmp"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="p-3 rounded-full bg-brand-blue/10 text-brand-lightBlue group-hover:scale-110 transition-transform mb-3">
                  <ImageIcon className="w-6 h-6" />
                </div>
                <p className="text-sm font-medium text-slate-200">Click to upload or drag and drop</p>
                <p className="text-xs text-slate-400 mt-1">JPEG, PNG, WebP, or BMP (max 20 MB)</p>
                <span className="text-[10px] text-cyan-400 font-mono mt-3 px-2 py-0.5 rounded bg-brand-blue/10 border border-brand-blue/20">
                  Images are resized to 380x380 with ImageNet normalization
                </span>
              </label>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden border border-navy-800 bg-navy-950 max-h-72 flex items-center justify-center p-2">
                  <img
                    src={previewUrl}
                    alt="Uploaded Evidence"
                    className="max-h-64 object-contain rounded-lg"
                  />
                </div>

                {/* File Metadata Card */}
                <div className="p-3.5 rounded-lg bg-navy-950/80 border border-navy-800 text-xs space-y-2">
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">File Name:</span>
                    <span className="font-mono font-medium truncate max-w-[240px]">{selectedFile?.name}</span>
                  </div>
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">Payload Size:</span>
                    <span className="font-mono">{(selectedFile?.size / 1024).toFixed(1)} KB</span>
                  </div>
                  {fileDimensions && (
                    <div className="flex justify-between items-center text-slate-300">
                      <span className="text-slate-400">Native Resolution:</span>
                      <span className="font-mono">{fileDimensions.width} &times; {fileDimensions.height} px</span>
                    </div>
                  )}
                  <div className="flex justify-between items-start text-slate-300 pt-1 border-t border-navy-850">
                    <span className="text-slate-400">SHA-256 Hash:</span>
                    <span className="font-mono text-[10px] text-cyan-300 break-all max-w-[240px] text-right">
                      {clientSha256 || 'Computing...'}
                    </span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <button
                    onClick={handleAnalyze}
                    disabled={isLoading || !isConnected}
                    className={`flex-1 py-2.5 px-4 rounded-lg font-medium text-xs flex items-center justify-center gap-2 transition-all ${
                      isLoading || !isConnected
                        ? 'bg-navy-800 text-slate-500 cursor-not-allowed'
                        : 'bg-brand-blue hover:bg-brand-accent text-white shadow-lg shadow-brand-blue/20'
                    }`}
                  >
                    {isLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-white" />
                        <span>Running EfficientNet-B4 Forward Pass...</span>
                      </>
                    ) : (
                      <>
                        <Cpu className="w-4 h-4" />
                        <span>Run Deepfake Inference</span>
                      </>
                    )}
                  </button>

                  <button
                    onClick={handleReset}
                    disabled={isLoading}
                    className="px-4 py-2.5 rounded-lg border border-navy-750 text-slate-300 hover:bg-navy-800 text-xs font-medium"
                  >
                    Clear
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Backend Status Warning */}
          {!isConnected && (
            <div className="p-3.5 rounded-lg border border-rose-500/30 bg-rose-500/10 text-xs text-rose-300 flex items-center gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>Inference unavailable: Start the FastAPI backend server on port 8000.</span>
            </div>
          )}

          {error && (
            <div className="p-3.5 rounded-lg border border-rose-500/40 bg-rose-500/10 text-xs text-rose-300 flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
        </div>

        {/* Right Column: Prediction Results & Explainability Quick View */}
        <div className="space-y-4">
          <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/60 h-full flex flex-col justify-between">
            <div>
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2 mb-4">
                <ShieldCheck className="w-4 h-4 text-emerald-400" />
                Forensic Classification Results
              </h3>

              {!result ? (
                <div className="py-20 text-center text-slate-400 space-y-3">
                  <Layers className="w-10 h-10 text-slate-700 mx-auto" />
                  <p className="text-sm font-medium text-slate-400">Awaiting Evidence Ingestion</p>
                  <p className="text-xs text-slate-400 max-w-xs mx-auto leading-relaxed">
                    Upload an image to run model inference and inspect Grad-CAM when a trained checkpoint is available.
                  </p>
                </div>
              ) : (
                <div className="space-y-5">
                  {/* Prominent Demo Notice if Mock Model */}
                  {result.is_mock_prediction && (
                    <div className="p-3.5 rounded-xl border border-amber-500/40 bg-amber-500/10 text-xs text-amber-300 flex items-start gap-2.5">
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                      <div className="space-y-0.5">
                        <p className="font-semibold text-amber-200">Demonstration Mode (Mock Weights)</p>
                        <p className="text-amber-300/90 text-[11px] leading-relaxed">
                          No trained checkpoint is loaded. This output demonstrates the image-processing pipeline only and cannot establish whether the image is authentic or manipulated.
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Verdict & Probability Badge */}
                  <div className="p-4 rounded-xl bg-navy-950/80 border border-navy-800 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-400">Model Classification:</span>
                      <StatusBadge
                        label={result.prediction.predicted_label}
                        isMock={result.is_mock_prediction}
                      />
                    </div>

                    <ConfidenceMeter
                      probability={result.prediction.probability}
                      threshold={result.prediction.decision_threshold}
                      isMock={result.is_mock_prediction}
                    />

                    {result.is_mock_prediction && (
                      <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300 leading-relaxed">
                        No trained checkpoint is loaded. This output demonstrates the image-processing pipeline only and cannot establish whether the image is authentic or manipulated.
                      </div>
                    )}

                    {!result.is_mock_prediction && (
                      <div className="grid grid-cols-2 gap-2 pt-2 border-t border-navy-850 text-xs font-mono">
                        <div>
                          <span className="text-slate-400 text-[10px] block">RAW LOGIT (s):</span>
                          <span className="text-white font-medium">{result.prediction.raw_logit}</span>
                        </div>
                        <div>
                          <span className="text-slate-400 text-[10px] block">SIGMOID SCORE:</span>
                          <span className="text-cyan-400 font-medium">
                            {(result.prediction.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Grad-CAM Quick Preview */}
                  {result.explainability?.overlay_data_url && (
                    <div className="p-4 rounded-xl bg-navy-950/80 border border-navy-800 space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                          <Eye className="w-3.5 h-3.5 text-cyan-400" />
                          Grad-CAM Saliency Overlay
                        </span>
                        <span className="text-[10px] font-mono text-cyan-400 px-2 py-0.5 rounded bg-brand-blue/15 border border-brand-blue/30">
                          {result.explainability.target_layer_name}
                        </span>
                      </div>

                      <div className="grid grid-cols-3 gap-2">
                        <div className="space-y-1 text-center">
                          <span className="text-[10px] text-slate-400">Original Crop</span>
                          <img
                            src={result.explainability.original_data_url}
                            alt="Original Crop"
                            className="w-full h-20 object-cover rounded-md border border-navy-800"
                          />
                        </div>
                        <div className="space-y-1 text-center">
                          <span className="text-[10px] text-slate-400">Heatmap (Jet)</span>
                          <img
                            src={result.explainability.heatmap_data_url}
                            alt="Raw Heatmap"
                            className="w-full h-20 object-cover rounded-md border border-navy-800"
                          />
                        </div>
                        <div className="space-y-1 text-center">
                          <span className="text-[10px] text-slate-400">Blended Overlay</span>
                          <img
                            src={result.explainability.overlay_data_url}
                            alt="Grad-CAM Overlay"
                            className="w-full h-20 object-cover rounded-md border border-navy-800"
                          />
                        </div>
                      </div>

                      <button
                        onClick={() => setActiveTab('explainability')}
                        className="w-full py-2 px-3 rounded-lg bg-navy-900 hover:bg-navy-850 text-cyan-300 text-xs font-medium border border-navy-750 flex items-center justify-center gap-1.5 transition-colors"
                      >
                        <span>Open Interactive Grad-CAM Inspector</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  )}

                  {/* Forensic Caution Note */}
                  <div className="p-3 bg-navy-950/60 border border-navy-800/80 rounded-lg text-[11px] text-slate-400 leading-relaxed">
                    <strong className="text-slate-300">Cautionary Forensic Notice:</strong> {result.prediction.forensic_disclaimer}
                  </div>
                </div>
              )}
            </div>

            {/* Jump to Report button */}
            {result && (
              <div className="pt-4 mt-4 border-t border-navy-800 flex justify-between items-center">
                <span className="text-[11px] font-mono text-slate-400">Evidence ID: {result.evidence_id}</span>
                <button
                  onClick={() => setActiveTab('report')}
                  className="text-xs text-brand-lightBlue hover:underline flex items-center gap-1"
                >
                  <FileCheck className="w-3.5 h-3.5" />
                  <span>View Full Forensic Examination Sheet</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
