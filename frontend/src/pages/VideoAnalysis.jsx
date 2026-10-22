import React, { useState } from 'react';
import {
  Upload,
  Video,
  AlertTriangle,
  Play,
  Layers,
  RefreshCw,
  Clock,
  Crosshair,
  ShieldAlert,
  ArrowRight,
  Eye,
  FileCheck,
} from 'lucide-react';
import { analyzeVideo, computeClientSha256 } from '../services/api';
import { StatusBadge, ConfidenceMeter } from '../components/StatusBadge';

export default function VideoAnalysis({
  backendStatus,
  onAnalysisComplete,
  setLatestAnalysis,
  setActiveTab,
}) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [clientSha256, setClientSha256] = useState('');
  const [maxFrames, setMaxFrames] = useState(16);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const isConnected = backendStatus?.connected;

  const handleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const validExtensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];
    const hasValidExt = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    if (!hasValidExt) {
      setError(`Unsupported video container. Please select an MP4, AVI, MOV, MKV, or WebM video.`);
      return;
    }

    if (file.size > 100 * 1024 * 1024) {
      setError('Video file exceeds 100 MB limit for prototype processing.');
      return;
    }

    setError(null);
    setResult(null);
    setSelectedFile(file);

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    const hash = await computeClientSha256(file);
    setClientSha256(hash);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    if (!isConnected) {
      setError('Cannot execute inference: FastAPI backend is not connected (http://localhost:8000). Please start the backend service.');
      return;
    }

    if (backendStatus?.data?.yunet_detector_available === false) {
      setError("Face detection model weights are missing. Place 'face_detection_yunet_2023mar.onnx' in 'models/yunet/' to enable video facial analysis.");
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const data = await analyzeVideo(selectedFile, maxFrames);
      setResult(data);

      const historyItem = {
        evidenceId: data.evidence_id,
        filename: data.filename,
        mediaType: 'video',
        fileSizeBytes: data.file_size_bytes,
        sha256: data.sha256,
        verdict: data.video_prediction.predicted_label,
        probability: data.video_prediction.probability,
        timestamp: data.examination_timestamp,
        isMock: data.is_mock_prediction,
        data: data,
      };

      setLatestAnalysis(data);
      onAnalysisComplete(historyItem);
    } catch (err) {
      setError(err.message || 'An error occurred during video analysis.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setClientSha256('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <Video className="w-5 h-5 text-purple-400" />
            Video Deepfake Analysis & Temporal Tracking
          </h2>
          <p className="text-xs text-slate-400">
            Extract temporal facial crops across video recordings, track subjects via IoU, and aggregate frame logits.
          </p>
        </div>

        {/* Frames to Sample */}
        <div className="flex items-center gap-3 bg-navy-900 border border-navy-800 px-3 py-1.5 rounded-lg text-xs">
          <span className="text-slate-400 font-medium">Uniform Frame Budget (k):</span>
          <span className="font-mono text-purple-400 font-semibold">{maxFrames} frames</span>
          <input
            type="range"
            min="8"
            max="32"
            step="4"
            value={maxFrames}
            onChange={(e) => setMaxFrames(parseInt(e.target.value))}
            className="w-24 accent-purple-500 cursor-pointer"
          />
        </div>
      </div>

      {/* Main Grid: Upload & Inspection */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Video Ingestion & Player */}
        <div className="space-y-4">
          <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/60 space-y-4">
            <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2">
              <Upload className="w-4 h-4 text-purple-400" />
              Video Evidence Ingestion
            </h3>

            {!previewUrl ? (
              <label className="border-2 border-dashed border-navy-750 hover:border-purple-500/60 bg-navy-950/60 rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all group">
                <input
                  type="file"
                  accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="p-3 rounded-full bg-purple-500/10 text-purple-400 group-hover:scale-110 transition-transform mb-3">
                  <Video className="w-6 h-6" />
                </div>
                <p className="text-sm font-medium text-slate-200">Click to upload video or drag and drop</p>
                <p className="text-xs text-slate-400 mt-1">MP4, AVI, MOV, MKV, or WebM (max 100 MB)</p>
                <span className="text-[10px] text-purple-400 font-mono mt-3 px-2 py-0.5 rounded bg-purple-500/10 border border-purple-500/20">
                  Samples {maxFrames} frames uniformly using OpenCV VideoCapture
                </span>
              </label>
            ) : (
              <div className="space-y-4">
                <div className="relative rounded-xl overflow-hidden border border-navy-800 bg-navy-950">
                  <video
                    src={previewUrl}
                    controls
                    className="w-full max-h-64 object-contain rounded-lg"
                  />
                </div>

                {/* File Metadata Card */}
                <div className="p-3.5 rounded-lg bg-navy-950/80 border border-navy-800 text-xs space-y-2">
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">File Name:</span>
                    <span className="font-mono font-medium truncate max-w-[240px]">{selectedFile?.name}</span>
                  </div>
                  <div className="flex justify-between items-center text-slate-300">
                    <span className="text-slate-400">Container Size:</span>
                    <span className="font-mono">{(selectedFile?.size / (1024 * 1024)).toFixed(2)} MB</span>
                  </div>
                  <div className="flex justify-between items-start text-slate-300 pt-1 border-t border-navy-850">
                    <span className="text-slate-400">SHA-256 Digest:</span>
                    <span className="font-mono text-[10px] text-purple-300 break-all max-w-[240px] text-right">
                      {clientSha256 || 'Computing...'}
                    </span>
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-3">
                  <button
                    onClick={handleAnalyze}
                    disabled={isLoading || !isConnected || backendStatus?.data?.yunet_detector_available === false}
                    className={`flex-1 py-2.5 px-4 rounded-lg font-medium text-xs flex items-center justify-center gap-2 transition-all ${
                      isLoading || !isConnected || backendStatus?.data?.yunet_detector_available === false
                        ? 'bg-navy-800 text-slate-500 cursor-not-allowed'
                        : 'bg-purple-600 hover:bg-purple-500 text-white shadow-lg shadow-purple-600/20'
                    }`}
                  >
                    {isLoading ? (
                      <>
                        <RefreshCw className="w-4 h-4 animate-spin text-white" />
                        <span>Decoding & Scoring {maxFrames} Frames...</span>
                      </>
                    ) : backendStatus?.data?.yunet_detector_available === false ? (
                      <>
                        <AlertTriangle className="w-4 h-4 text-amber-400" />
                        <span>YuNet Model Weights Required</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        <span>Execute Video Analysis</span>
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

          {/* Missing YuNet Weights Actionable Alert */}
          {isConnected && backendStatus?.data?.yunet_detector_available === false && (
            <div className="p-4 rounded-xl border border-amber-500/40 bg-amber-500/10 text-xs text-amber-300 flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              <div className="space-y-1">
                <p className="font-semibold text-amber-200">YuNet Face Detection Weights Missing</p>
                <p className="text-amber-300/90 leading-relaxed text-[11px]">
                  Video face extraction and tracking requires OpenCV YuNet ONNX weights.
                  Please place <code className="font-mono bg-amber-950/60 px-1 py-0.5 rounded text-white">face_detection_yunet_2023mar.onnx</code> in the <code className="font-mono bg-amber-950/60 px-1 py-0.5 rounded text-white">models/yunet/</code> folder to activate video processing.
                </p>
              </div>
            </div>
          )}

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

        {/* Right Column: Video Aggregate Results & Peak Keyframe */}
        <div className="space-y-4">
          <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/60 h-full flex flex-col justify-between">
            <div>
              <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-2 mb-4">
                <ShieldAlert className="w-4 h-4 text-purple-400" />
                Aggregated Video Verdict
              </h3>

              {!result ? (
                <div className="py-20 text-center text-slate-400 space-y-3">
                  <Layers className="w-10 h-10 text-slate-700 mx-auto" />
                  <p className="text-sm font-medium text-slate-400">Awaiting Video Ingestion</p>
                  <p className="text-xs text-slate-400 max-w-xs mx-auto leading-relaxed">
                    Upload a video to extract temporal facial crops, compute frame-by-frame forgery probabilities, and identify the peak anomaly keyframe.
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
                          No trained checkpoint is loaded. This output demonstrates the video-processing pipeline only and cannot establish whether the video is authentic or manipulated.
                        </p>
                      </div>
                    </div>
                  )}

                  {/* Video-Level Verdict Card */}
                  <div className="p-4 rounded-xl bg-navy-950/80 border border-navy-800 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-400">Aggregated Video Decision:</span>
                      <StatusBadge
                        label={result.video_prediction.predicted_label}
                        isMock={result.is_mock_prediction}
                      />
                    </div>

                    <ConfidenceMeter
                      probability={result.video_prediction.probability}
                      isMock={result.is_mock_prediction}
                    />

                    {result.is_mock_prediction && (
                      <div className="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300 leading-relaxed">
                        No trained checkpoint is loaded. This output demonstrates the video-processing pipeline only and cannot establish whether the video is authentic or manipulated.
                      </div>
                    )}

                    <div className="grid grid-cols-3 gap-2 pt-2 border-t border-navy-850 text-xs font-mono">
                      {!result.is_mock_prediction && (
                        <div>
                          <span className="text-slate-400 text-[10px] block">MEAN LOGIT (s):</span>
                          <span className="text-white font-medium">{result.video_prediction.mean_logit}</span>
                        </div>
                      )}
                      <div>
                        <span className="text-slate-400 text-[10px] block">VALID CROPS:</span>
                        <span className="text-purple-400 font-medium">
                          {result.video_prediction.valid_frames_count} / {result.frames.length}
                        </span>
                      </div>
                      <div>
                        <span className="text-slate-400 text-[10px] block">METHOD:</span>
                        <span className="text-slate-300 font-medium">Arithmetic Mean</span>
                      </div>
                    </div>
                  </div>

                  {/* Peak Anomaly Keyframe Highlight */}
                  {result.keyframe_explainability?.overlay_data_url && (
                    <div className="p-4 rounded-xl bg-navy-950/80 border border-purple-500/30 space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-xs font-semibold text-purple-300 uppercase tracking-wider flex items-center gap-1.5">
                          <Crosshair className="w-3.5 h-3.5 text-purple-400" />
                          Peak Anomaly Keyframe (Frame {result.keyframe_explainability.frame_index})
                        </span>
                        <span className="text-[10px] font-mono text-purple-400 flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {result.keyframe_explainability.timestamp_sec}s
                        </span>
                      </div>

                      <div className="grid grid-cols-2 gap-3">
                        <div className="space-y-1 text-center">
                          <span className="text-[10px] text-slate-400">Extracted Face Crop</span>
                          <img
                            src={result.keyframe_explainability.crop_data_url}
                            alt="Keyframe Face"
                            className="w-full h-28 object-cover rounded-md border border-navy-800"
                          />
                        </div>
                        <div className="space-y-1 text-center">
                          <span className="text-[10px] text-slate-400">Grad-CAM Saliency Overlay</span>
                          <img
                            src={result.keyframe_explainability.overlay_data_url}
                            alt="Keyframe CAM"
                            className="w-full h-28 object-cover rounded-md border border-purple-500/40"
                          />
                        </div>
                      </div>

                      <button
                        onClick={() => setActiveTab('explainability')}
                        className="w-full py-2 px-3 rounded-lg bg-navy-900 hover:bg-navy-850 text-purple-300 text-xs font-medium border border-navy-750 flex items-center justify-center gap-1.5 transition-colors"
                      >
                        <Eye className="w-3.5 h-3.5" />
                        <span>Inspect Full Saliency Attribution</span>
                      </button>
                    </div>
                  )}

                  <div className="p-3 bg-navy-950/60 border border-navy-800/80 rounded-lg text-[11px] text-slate-400 leading-relaxed">
                    <strong className="text-slate-300">Temporal Dependency Notice:</strong> {result.video_prediction.forensic_disclaimer}
                  </div>
                </div>
              )}
            </div>

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

      {/* Bottom Row: Full Sampled Frames Timeline */}
      {result?.frames?.length > 0 && (
        <div className="p-5 rounded-xl border border-navy-800 bg-navy-900/60 space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
              <Clock className="w-4 h-4 text-purple-400" />
              Sampled Frame Temporal Timeline ({result.frames.length} frames)
            </h3>
            <span className="text-[11px] text-slate-400 font-mono">
              Scanned: {result.video_metadata.num_frames_scanned} frames @ {result.video_metadata.fps} fps
            </span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-3">
            {result.frames.map((frame, idx) => {
              const isPeak = frame.frame_index === result.keyframe_explainability?.frame_index;
              const isFake = !result.is_mock_prediction && frame.predicted_label?.toLowerCase() === 'fake';

              return (
                <div
                  key={idx}
                  className={`p-2 rounded-lg border text-xs space-y-1.5 transition-all ${
                    isPeak
                      ? 'border-purple-500 bg-purple-500/15 shadow-md shadow-purple-500/10'
                      : 'border-navy-800 bg-navy-950/80'
                  }`}
                >
                  <div className="flex justify-between items-center text-[10px] text-slate-400 font-mono">
                    <span>#{frame.frame_index}</span>
                    <span>{frame.timestamp_sec}s</span>
                  </div>

                  <div className="aspect-square bg-navy-900 rounded overflow-hidden flex items-center justify-center border border-navy-850">
                    {frame.crop_data_url ? (
                      <img
                        src={frame.crop_data_url}
                        alt={`Frame ${frame.frame_index}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <span className="text-[10px] text-slate-400 text-center px-1">No Face</span>
                    )}
                  </div>

                  <div className="text-center font-mono text-[10px]">
                    {frame.probability !== null ? (
                      <span className={result.is_mock_prediction ? 'text-amber-400 font-medium' : (isFake ? 'text-rose-400 font-bold' : 'text-emerald-400')}>
                        {result.is_mock_prediction ? 'Demo' : `${Math.round(frame.probability * 100)}%`}
                      </span>
                    ) : (
                      <span className="text-slate-400">N/A</span>
                    )}
                  </div>

                  {isPeak && (
                    <div className="text-[9px] font-bold text-purple-300 text-center uppercase tracking-wide bg-purple-900/60 rounded py-0.5">
                      Peak Keyframe
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
