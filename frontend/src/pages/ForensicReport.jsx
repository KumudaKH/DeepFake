import React, { useState } from 'react';
import {
  FileText,
  Download,
  ExternalLink,
  ShieldAlert,
  Hash,
  Calendar,
  Layers,
  ArrowRight,
  HardDrive,
  Copy,
  Check,
  Eye,
  Film,
  Activity,
  Cpu,
  Table,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { getHtmlReportUrl, getJsonReportUrl } from '../services/api';
import { StatusBadge } from '../components/StatusBadge';

export default function ForensicReport({ latestAnalysis, setActiveTab }) {
  const [copiedHash, setCopiedHash] = useState(false);

  if (!latestAnalysis) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-emerald-400" />
            Digital Forensic Examination Sheet
          </h2>
          <p className="text-xs text-slate-400">
            Generate formal digital evidence sheets containing cryptographic hashes, neural predictions, and limitations.
          </p>
        </div>

        <div className="p-12 rounded-xl border border-navy-800 bg-navy-900/40 text-center space-y-4 max-w-xl mx-auto my-8">
          <div className="p-4 rounded-full bg-emerald-500/10 text-emerald-400 w-16 h-16 mx-auto flex items-center justify-center">
            <FileText className="w-8 h-8" />
          </div>
          <h3 className="text-base font-semibold text-white">No Forensic Report Generated Yet</h3>
          <p className="text-xs text-slate-400 leading-relaxed">
            Examination records are created when an image or video is ingested and classified. Run an analysis in the Image or Video tabs to generate a downloadable evidence report.
          </p>
          <div className="flex justify-center gap-3 pt-2">
            <button
              onClick={() => setActiveTab('image')}
              className="py-2 px-4 rounded-lg bg-brand-blue hover:bg-brand-accent text-white text-xs font-medium flex items-center gap-2 transition-colors"
            >
              <span>Examine Image</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setActiveTab('video')}
              className="py-2 px-4 rounded-lg bg-purple-600 hover:bg-purple-500 text-white text-xs font-medium flex items-center gap-2 transition-colors"
            >
              <span>Examine Video</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    );
  }

  const analysisObj = latestAnalysis?.data || latestAnalysis;
  const isImage = analysisObj?.mediaType === 'image' || Boolean(analysisObj?.prediction) || !Boolean(analysisObj?.video_prediction);
  const filename = analysisObj?.filename || latestAnalysis?.filename || 'evidence.bin';
  const evidenceId = analysisObj?.evidence_id || analysisObj?.evidenceId || latestAnalysis?.evidence_id || latestAnalysis?.evidenceId || 'EV-UNKNOWN';
  const sha256 = analysisObj?.sha256 || latestAnalysis?.sha256 || '';
  const fileSizeBytes = analysisObj?.file_size_bytes || analysisObj?.fileSizeBytes || latestAnalysis?.file_size_bytes || latestAnalysis?.fileSizeBytes || 0;
  const timestamp = analysisObj?.examination_timestamp || analysisObj?.timestamp || latestAnalysis?.examination_timestamp || latestAnalysis?.timestamp || new Date().toISOString();

  const isMock = analysisObj?.is_mock_prediction !== undefined
    ? analysisObj.is_mock_prediction
    : (latestAnalysis?.isMock !== undefined ? latestAnalysis.isMock : true);

  const rawVerdict = isImage
    ? (analysisObj?.prediction?.predicted_label || analysisObj?.verdict || latestAnalysis?.verdict || 'inconclusive')
    : (analysisObj?.video_prediction?.predicted_label || analysisObj?.verdict || latestAnalysis?.verdict || 'inconclusive');

  const verdict = isMock ? 'inconclusive' : rawVerdict;

  const probability = isMock
    ? null
    : (isImage
      ? (analysisObj?.prediction?.probability ?? latestAnalysis?.probability ?? null)
      : (analysisObj?.video_prediction?.probability ?? latestAnalysis?.probability ?? null));

  const rawLogit = isMock
    ? null
    : (isImage
      ? (analysisObj?.prediction?.raw_logit ?? null)
      : (analysisObj?.video_prediction?.mean_logit ?? null));

  const targetLayerName = isImage
    ? (analysisObj?.explainability?.target_layer_name || 'conv_head')
    : (analysisObj?.keyframe_explainability?.target_layer_name || 'conv_head');
  const modelArchitecture = analysisObj?.prediction?.model_architecture
    || analysisObj?.model_architecture
    || 'efficientnet_b4';
  const checkpointType = analysisObj?.checkpoint_type || (isMock ? 'uninitialized_mock' : 'deepfake_trained');
  const checkpointPath = isMock ? null : (
    analysisObj?.prediction?.checkpoint_path || analysisObj?.checkpoint_path || null
  );
  const decisionThreshold = isMock
    ? null
    : (isImage
      ? (analysisObj?.prediction?.decision_threshold ?? 0.5)
      : (analysisObj?.video_prediction?.decision_threshold ?? 0.5));

  const visualEvidence = isImage ? {
    originalUrl: analysisObj?.explainability?.original_data_url || null,
    heatmapUrl: analysisObj?.explainability?.heatmap_data_url || null,
    overlayUrl: analysisObj?.explainability?.overlay_data_url || null,
    title: 'Input Evidence & Face Crop',
    label: 'Single-Image Facial Crop & Spatial Saliency',
    layer: targetLayerName,
  } : {
    originalUrl: analysisObj?.keyframe_explainability?.crop_data_url || null,
    heatmapUrl: analysisObj?.keyframe_explainability?.heatmap_data_url || null,
    overlayUrl: analysisObj?.keyframe_explainability?.overlay_data_url || null,
    keyframeIdx: analysisObj?.keyframe_explainability?.frame_index,
    keyframeTime: analysisObj?.keyframe_explainability?.timestamp_sec,
    title: `Peak Anomaly Keyframe #${analysisObj?.keyframe_explainability?.frame_index ?? 0}`,
    label: `Peak Anomaly Keyframe #${analysisObj?.keyframe_explainability?.frame_index ?? 0} (t = ${(analysisObj?.keyframe_explainability?.timestamp_sec ?? 0).toFixed(2)}s)`,
    layer: targetLayerName,
  };

  const videoMeta = !isImage ? (analysisObj?.video_metadata || {}) : null;
  const frames = !isImage ? (analysisObj?.frames || []) : [];

  const generateNarrative = () => {
    const probPct = probability === null ? null : Math.round(probability * 100);
    const logitStr = rawLogit === null ? null : Number(rawLogit).toFixed(4);

    if (isMock) {
      if (!isImage) {
        return `TEMPORAL PIPELINE DEMONSTRATION NOTICE: Video evidence '${filename}' was processed across ${videoMeta?.num_frames_scanned || frames.length || 0} sampled temporal frames using OpenCV YuNet face detection and tracking. No trained checkpoint is loaded, so no classification score or authenticity finding is available.`;
      }
      return `DEMONSTRATION MODE NOTICE: Image evidence '${filename}' was received by the image-analysis pipeline. No trained checkpoint is loaded, so no classification score, authenticity finding, or Grad-CAM attribution is available.`;
    }

    if (probability === null || rawLogit === null) {
      return `FORENSIC ASSESSMENT: No measured model score is available for '${filename}', so no probability-based classification narrative can be generated.`;
    }

    if (verdict === 'fake') {
      if (!isImage) {
        const kfInfo = visualEvidence.keyframeIdx !== undefined ? `Peak anomaly was localized at Keyframe #${visualEvidence.keyframeIdx} (timestamp: ${Number(visualEvidence.keyframeTime).toFixed(2)}s). ` : '';
        return `TEMPORAL FORENSIC ANALYSIS: Video evidence '${filename}' was evaluated across ${videoMeta?.num_frames_scanned || frames.length || 16} frames. Temporal facial crops were tracked via OpenCV YuNet and scored. Arithmetic mean logit pooling produced an aggregate continuous logit of ${logitStr}, corresponding to a model probability P(Fake) = ${probPct}%, exceeding the configured decision threshold. ${kfInfo}Grad-CAM highlights regions that influenced the model score; it does not prove which pixels were manipulated.`;
      }
      return `SINGLE-IMAGE FORENSIC ANALYSIS: Examination of '${filename}' yielded a model probability P(Fake) = ${probPct}%, exceeding the configured decision threshold. Grad-CAM indicates regions that influenced the model score; it does not prove which pixels were manipulated.`;
    }

    if (verdict === 'real') {
      if (!isImage) {
        return `TEMPORAL FORENSIC ANALYSIS: Video evidence '${filename}' was evaluated across ${videoMeta?.num_frames_scanned || frames.length || 0} temporal frames. Arithmetic mean logit pooling yielded an aggregate continuous logit of ${logitStr}, corresponding to a model probability P(Fake) = ${probPct}%, below the configured decision threshold. This is a model output, not proof of authenticity.`;
      }
      return `SINGLE-IMAGE FORENSIC ANALYSIS: Examination of '${filename}' yielded a model probability P(Fake) = ${probPct}%, below the configured decision threshold. This is a model output, not proof that the image is authentic.`;
    }

    return `FORENSIC ASSESSMENT: Examination of '${filename}' was inconclusive. No valid trained-model classification score is available.`;
  };

  const narrativeText = generateNarrative();

  const handleDownloadSummaryJson = () => {
    let reportData;
    const examinationTimestamp = timestamp;

    if (isImage) {
      reportData = {
        report_title: "Digital Forensic Examination Sheet - Image Evidence",
        report_version: "1.0.0",
        academic_project: "Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos",
        evidence_metadata: {
          evidence_id: evidenceId,
          filename: filename,
          file_size_bytes: fileSizeBytes,
          sha256_hash: sha256,
          examination_timestamp_utc: examinationTimestamp,
          media_type: "image",
          dimensions: analysisObj?.dimensions || null,
        },
        model_provenance: {
          model_architecture: modelArchitecture,
          checkpoint_status: checkpointType,
          checkpoint_path: checkpointPath,
          target_layer_name: targetLayerName,
          device: "cpu",
        },
        classification: {
          verdict: verdict,
          verdict_display: isMock ? "INCONCLUSIVE — DEMO MODE" : verdict.toUpperCase(),
          probability,
          raw_logit: rawLogit,
          decision_threshold: decisionThreshold,
          is_mock_prediction: isMock,
        },
        explainable_forensic_narrative: narrativeText,
        explainability_metadata: {
          target_layer_name: targetLayerName,
          alpha: 0.4,
          has_heatmap: Boolean(visualEvidence.heatmapUrl),
          has_overlay: Boolean(visualEvidence.overlayUrl),
        },
        visual_evidence: {
          original_data_url: visualEvidence.originalUrl,
          heatmap_data_url: visualEvidence.heatmapUrl,
          overlay_data_url: visualEvidence.overlayUrl,
        },
        forensic_limitations_and_legal_notice: isMock
          ? "No trained checkpoint is loaded. This output demonstrates the image-processing pipeline only and cannot establish whether the image is authentic or manipulated."
          : "Model classification probabilities represent probabilistic neural inferences and do NOT constitute definitive legal or judicial proof of manipulation. Grad-CAM highlights regions influencing the model score; it does not prove the exact pixels that were manipulated.",
      };
    } else {
      reportData = {
        report_title: "Digital Forensic Examination Sheet - Video Evidence Summary",
        report_version: "1.0.0",
        academic_project: "Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos",
        evidence_metadata: {
          evidence_id: evidenceId,
          filename: filename,
          file_size_bytes: fileSizeBytes,
          sha256_hash: sha256,
          media_type: "video",
          examination_timestamp_utc: examinationTimestamp,
        },
        video_container_metadata: {
          num_frames_scanned: videoMeta?.num_frames_scanned || frames.length,
          num_faces_extracted: videoMeta?.num_faces_extracted || 0,
          num_unscorable_frames: videoMeta?.num_unscorable_frames || 0,
          fps: videoMeta?.fps || 0,
          video_dimensions: videoMeta?.video_dimensions || null,
        },
        model_provenance: {
          model_architecture: modelArchitecture,
          checkpoint_status: checkpointType,
          checkpoint_path: checkpointPath,
          face_detector: "OpenCV YuNet ONNX (face_detection_yunet_2023mar.onnx)",
          target_layer_name: targetLayerName,
        },
        aggregated_video_verdict: {
          verdict: verdict,
          verdict_display: isMock ? "INCONCLUSIVE — DEMO MODE" : verdict.toUpperCase(),
          mean_logit: rawLogit,
          probability,
          aggregation_method: "arithmetic_mean_logits",
          valid_frames_count: videoMeta?.num_faces_extracted || frames.filter(f => f.has_face).length,
          decision_threshold: decisionThreshold,
          is_mock_prediction: isMock,
        },
        peak_anomaly_keyframe: {
          frame_index: visualEvidence.keyframeIdx ?? null,
          timestamp_sec: visualEvidence.keyframeTime ?? null,
          target_layer_name: targetLayerName,
        },
        visual_evidence: {
          crop_data_url: visualEvidence.originalUrl,
          heatmap_data_url: visualEvidence.heatmapUrl,
          overlay_data_url: visualEvidence.overlayUrl,
        },
        video_frame_results: frames.map((frame) => ({
          frame_index: frame.frame_index,
          timestamp_sec: frame.timestamp_sec,
          tracking_status: frame.tracking_status,
          has_face: frame.has_face,
          box: frame.box,
          logit: isMock ? null : frame.logit,
          probability: isMock ? null : frame.probability,
          predicted_label: isMock ? 'inconclusive' : frame.predicted_label,
        })),
        explainable_forensic_narrative: narrativeText,
        forensic_limitations_and_legal_notice: isMock
          ? "No trained checkpoint is loaded. This output demonstrates the video-processing pipeline only and cannot establish whether the video is authentic or manipulated."
          : "Video aggregation averages facial frame logits. Consecutive frames are temporally correlated and must not be treated as independent observations. Grad-CAM indicates regions influencing the model score; it does not prove the exact pixels that were manipulated.",
      };
    }

    const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = isImage ? `forensic_report_${evidenceId}.json` : `video_forensic_report_summary_${evidenceId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadVideoMetadata = () => {
    if (isImage || !frames.length) return;
    const metadata = {
      evidence_id: evidenceId,
      filename: filename,
      sha256_hash: sha256,
      examination_timestamp_utc: new Date(timestamp).toISOString(),
      total_frames_sampled: frames.length,
      frames: frames.map(f => ({
        frame_index: f.frame_index,
        timestamp_sec: f.timestamp_sec,
        tracking_status: f.tracking_status,
        has_face: f.has_face,
        box: f.box,
        logit: f.logit,
        probability: f.probability,
        predicted_label: f.predicted_label,
      })),
    };
    const blob = new Blob([JSON.stringify(metadata, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `video_evidence_metadata_${evidenceId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrintableHtml = () => {
    const verdictColor = isMock ? '#f59e0b' : (verdict === 'fake' ? '#ef4444' : (verdict === 'real' ? '#10b981' : '#64748b'));
    const verdictLabel = isMock ? 'INCONCLUSIVE — DEMO MODE' : verdict.toUpperCase();
    const nowStr = new Date(timestamp).toUTCString();

    const framesTableHtml = !isImage && frames.length ? `
      <h3 style="font-size: 14px; font-weight: 700; text-transform: uppercase; color: #475569; margin: 28px 0 10px 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">
        Sampled Frame Temporal Breakdown (${frames.length} frames)
      </h3>
      <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px;">
        <thead>
          <tr style="background: #f1f5f9; color: #475569; text-align: left;">
            <th style="padding: 8px;">Frame #</th>
            <th style="padding: 8px;">Time (s)</th>
            <th style="padding: 8px;">Tracking Status</th>
            <th style="padding: 8px;">Bounding Box</th>
            <th style="padding: 8px;">Logit (s_i)</th>
            <th style="padding: 8px;">Score P_i</th>
            <th style="padding: 8px;">Frame Status</th>
          </tr>
        </thead>
        <tbody>
          ${frames.map(f => `
            <tr style="border-bottom: 1px solid #e2e8f0;">
              <td style="padding: 8px; font-family: monospace;">#${f.frame_index}</td>
              <td style="padding: 8px; font-family: monospace;">${Number(f.timestamp_sec).toFixed(3)}s</td>
              <td style="padding: 8px;"><span style="display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; background: ${f.tracking_status === 'detected' ? '#dcfce7; color: #166534;' : '#f1f5f9; color: #475569;'}">${f.tracking_status}</span></td>
              <td style="padding: 8px; font-family: monospace; font-size: 11px;">${f.box ? `[${f.box.map(v => Math.round(v)).join(', ')}]` : 'N/A'}</td>
              <td style="padding: 8px; font-family: monospace;">${f.logit !== null && f.logit !== undefined ? f.logit.toFixed(4) : '—'}</td>
              <td style="padding: 8px; font-family: monospace;">${f.probability !== null && f.probability !== undefined ? `${(f.probability * 100).toFixed(1)}%` : '—'}</td>
              <td style="padding: 8px; font-weight: 600; color: ${isMock ? '#d97706' : (f.predicted_label === 'fake' ? '#dc2626' : (f.predicted_label === 'real' ? '#16a34a' : '#64748b'))}">${isMock ? 'INCONCLUSIVE' : (f.predicted_label || '—').toUpperCase()}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    ` : '';

    const visualEvidenceHtml = (visualEvidence.originalUrl || visualEvidence.heatmapUrl || visualEvidence.overlayUrl) ? `
      <h3 style="font-size: 14px; font-weight: 700; text-transform: uppercase; color: #475569; margin: 28px 0 10px 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">
        Visual Forensic Evidence & Spatial Interpretability (Tri-Pane)
      </h3>
      <p style="font-size: 12px; color: #64748b; margin-bottom: 12px;">${visualEvidence.label} &bull; Target Layer: <code>model.${targetLayerName}</code></p>
      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 24px;">
        <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; background: #fafafa; text-align: center;">
          <strong style="display: block; font-size: 12px; margin-bottom: 6px; color: #334155;">Original Evidence Crop</strong>
          ${visualEvidence.originalUrl ? `<img src="${visualEvidence.originalUrl}" style="max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto;" />` : '<div style="padding: 40px; font-size: 11px; color: #94a3b8;">Not available</div>'}
          <span style="font-size: 10px; color: #64748b; display: block; margin-top: 6px;">Input RGB Face Crop</span>
        </div>
        <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; background: #fafafa; text-align: center;">
          <strong style="display: block; font-size: 12px; margin-bottom: 6px; color: #334155;">Grad-CAM Saliency Map</strong>
          ${visualEvidence.heatmapUrl ? `<img src="${visualEvidence.heatmapUrl}" style="max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto;" />` : '<div style="padding: 40px; font-size: 11px; color: #94a3b8;">Not available</div>'}
          <span style="font-size: 10px; color: #64748b; display: block; margin-top: 6px;">Jet Colormap Attribution</span>
        </div>
        <div style="border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px; background: #fafafa; text-align: center;">
          <strong style="display: block; font-size: 12px; margin-bottom: 6px; color: #334155;">Saliency Overlay</strong>
          ${visualEvidence.overlayUrl ? `<img src="${visualEvidence.overlayUrl}" style="max-width: 100%; height: auto; border-radius: 4px; display: block; margin: 0 auto;" />` : '<div style="padding: 40px; font-size: 11px; color: #94a3b8;">Not available</div>'}
          <span style="font-size: 10px; color: #64748b; display: block; margin-top: 6px;">Blended &alpha; = 0.4 Overlay</span>
        </div>
      </div>
    ` : '';

    const htmlDoc = `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Digital Forensics Examination Report - ${evidenceId}</title>
  <style>
    body { font-family: 'Segoe UI', -apple-system, Roboto, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 32px 16px; }
    .report-card { max-width: 900px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 40px; }
    .badge { display: inline-block; padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 13px; text-transform: uppercase; color: #fff; background-color: ${verdictColor}; }
    .table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
    .table th, .table td { text-align: left; padding: 10px 12px; border-bottom: 1px solid #e2e8f0; }
    .table th { background-color: #f8fafc; color: #475569; width: 34%; font-weight: 600; }
    .narrative-box { background-color: #f1f5f9; border-left: 4px solid #0284c7; padding: 16px 20px; border-radius: 6px; font-size: 13px; line-height: 1.6; color: #0f172a; margin-top: 8px; }
    .disclaimer-box { background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 16px 20px; margin-top: 24px; font-size: 12px; color: #991b1b; line-height: 1.6; border-radius: 6px; }
    @media print { body { padding: 0; background: #fff; } .report-card { border: none; padding: 0; } }
  </style>
</head>
<body>
  <div class="report-card">
    <div style="display: flex; justify-content: space-between; align-items: flex-start; border-bottom: 2px solid #0f172a; padding-bottom: 16px; margin-bottom: 20px;">
      <div>
        <h1 style="font-size: 22px; margin: 0 0 6px 0; color: #0f172a;">Digital Forensics Examination Sheet</h1>
        <p style="margin: 0; color: #64748b; font-size: 13px;">Multimodal Deepfake Detection and Explainable Digital Forensics System &bull; Academic Forensic Prototype</p>
      </div>
      <div>
        <span class="badge">${verdictLabel}</span>
      </div>
    </div>

    ${isMock ? `
      <div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; margin-bottom: 20px; font-size: 13px; color: #92400e; border-radius: 6px;">
        <strong>DEMO MODE NOTICE:</strong> No trained checkpoint is loaded. This output demonstrates the media-processing pipeline only and cannot establish whether the evidence is authentic or manipulated.
      </div>
    ` : ''}

    <div style="background: #f8fafc; padding: 12px 16px; border-radius: 8px; border: 1px solid #e2e8f0; margin-bottom: 20px; display: flex; justify-content: space-between;">
      <div>
        <span style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600;">Evidence Identifier:</span>
        <strong style="font-family: monospace; font-size: 16px; margin-left: 8px; color: #0f172a;">${evidenceId}</strong>
      </div>
      <div>
        <span style="font-size: 11px; color: #64748b;">Examination Date:</span>
        <span style="font-family: monospace; font-size: 12px; margin-left: 6px;">${nowStr}</span>
      </div>
    </div>

    <h3 style="font-size: 14px; font-weight: 700; text-transform: uppercase; color: #475569; margin: 20px 0 10px 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">
      Evidence & Model Provenance
    </h3>
    <table class="table">
      <tr><th>Original File Name</th><td style="font-family: monospace;">${filename}</td></tr>
      <tr><th>Cryptographic Hash (SHA-256)</th><td style="font-family: monospace; font-size: 11px; word-break: break-all;">${sha256}</td></tr>
      <tr><th>Media Type</th><td>${isImage ? 'Single-Image Face Crop' : 'Temporal Video Sequence'}</td></tr>
      <tr><th>File Payload Size</th><td>${(fileSizeBytes / 1024).toFixed(1)} KB (${fileSizeBytes.toLocaleString()} bytes)</td></tr>
      ${!isImage && videoMeta ? `<tr><th>Container Specifications</th><td>${videoMeta.fps || 0} FPS &bull; ${videoMeta.num_frames_scanned || frames.length} frames scanned &bull; ${videoMeta.num_faces_extracted || frames.filter(f => f.has_face).length} faces scored</td></tr>` : ''}
      <tr><th>Classification Verdict</th><td><strong>${verdictLabel}</strong></td></tr>
      ${rawLogit !== null ? `<tr><th>Continuous Logit Projection (s)</th><td><code>${Number(rawLogit).toFixed(4)}</code></td></tr>` : ''}
      ${probability !== null ? `<tr><th>Probability P(Fake)</th><td><strong>${Number(probability).toFixed(4)}</strong> (${(probability * 100).toFixed(1)}%)</td></tr>` : ''}
      ${decisionThreshold !== null ? `<tr><th>Decision Threshold</th><td><code>&tau; = ${Number(decisionThreshold).toFixed(4)}</code></td></tr>` : ''}
      <tr><th>Model Architecture</th><td>${modelArchitecture} (${isMock ? 'Demo Mode' : 'Trained Checkpoint'})</td></tr>
      <tr><th>Explainability Method</th><td>${isMock ? 'Unavailable without trained weights' : `Grad-CAM targeting model.${targetLayerName}; highlights influential regions, not exact manipulated pixels`}</td></tr>
    </table>

    ${visualEvidenceHtml}

    <h3 style="font-size: 14px; font-weight: 700; text-transform: uppercase; color: #475569; margin: 24px 0 10px 0; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px;">
      Explainable Forensic Narrative
    </h3>
    <div class="narrative-box">
      ${narrativeText}
    </div>

    ${framesTableHtml}

    <div class="disclaimer-box">
      <strong>Evidentiary Limitations & Forensic Legal Notice:</strong><br>
      Neural network classification scores represent probabilistic pattern matching and do NOT constitute self-authenticating judicial proof under Federal Rule of Evidence 901/902 or equivalent legal standards. Grad-CAM saliency heatmaps indicate internal receptive field activations, not certified physical tampering masks or compression boundary delineations. Formal legal custody must be established through independent evidentiary documentation.
    </div>

    <div style="margin-top: 32px; font-size: 11px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 16px;">
      Final-Year Computer Science and Engineering Capstone Project &bull; Department of Computer Science and Engineering &bull; October 2026
    </div>
  </div>
</body>
</html>`;

    const blob = new Blob([htmlDoc], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
  };

  const handleCopyHash = () => {
    if (sha256) {
      navigator.clipboard.writeText(sha256);
      setCopiedHash(true);
      setTimeout(() => setCopiedHash(false), 2000);
    }
  };

  const backendHtmlUrl = getHtmlReportUrl(
    evidenceId,
    filename,
    sha256,
    verdict,
    probability,
    isImage ? 'image' : 'video',
    {
      isMock,
      rawLogit,
      fileSizeBytes,
      framesCount: frames.length || videoMeta?.num_frames_scanned,
      examinationTimestampUtc: timestamp,
      decisionThreshold,
    }
  );

  return (
    <div className="space-y-6">
      {/* Header and Export Toolbar */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div>
          <h2 className="text-lg font-bold text-white flex items-center gap-2">
            <FileText className="w-5 h-5 text-emerald-400" />
            Digital Forensic Examination Sheet
          </h2>
          <p className="text-xs text-slate-400">
            Formal digital evidence report conforming to Colab multimodal forensic analysis standards.
          </p>
        </div>

        {/* Export Buttons */}
        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={handleDownloadSummaryJson}
            className="py-1.5 px-3 rounded-lg bg-navy-800 hover:bg-navy-750 text-slate-200 border border-navy-700 text-xs font-medium flex items-center gap-1.5 transition-colors shadow-sm"
            title="Download full forensic summary in JSON format"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Download Summary JSON</span>
          </button>

          {!isImage && frames.length > 0 && (
            <button
              onClick={handleDownloadVideoMetadata}
              className="py-1.5 px-3 rounded-lg bg-navy-800 hover:bg-navy-750 text-slate-200 border border-navy-700 text-xs font-medium flex items-center gap-1.5 transition-colors shadow-sm"
              title="Download frame-level tracking and detection metadata (video_evidence_metadata.json)"
            >
              <Table className="w-3.5 h-3.5 text-purple-400" />
              <span>Download Frame Metadata (JSON)</span>
            </button>
          )}

          <button
            onClick={handlePrintableHtml}
            className="py-1.5 px-3.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium flex items-center gap-1.5 shadow-md shadow-emerald-600/20 transition-colors"
            title="Open complete self-contained printable report with embedded images"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span>Printable HTML Report</span>
          </button>

          <a
            href={backendHtmlUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="py-1.5 px-2.5 rounded-lg bg-navy-850 hover:bg-navy-800 text-slate-400 hover:text-slate-200 border border-navy-750 text-xs flex items-center gap-1 transition-colors"
            title="Open raw backend HTML export endpoint"
          >
            <span className="text-[11px]">API HTML</span>
          </a>
        </div>
      </div>

      {/* Main Report Card Container */}
      <div className="p-6 rounded-xl border border-navy-800 bg-navy-900/60 space-y-6">
        {/* Prominent Demo Notice if Mock Model */}
        {isMock && (
          <div className="p-4 rounded-lg border border-amber-500/40 bg-amber-500/10 text-xs text-amber-300 flex items-start gap-3">
            <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <span className="font-semibold text-amber-200 text-xs uppercase tracking-wide">Demonstration Mode Examination Sheet</span>
              <p className="text-amber-300/90 leading-relaxed text-[11px]">
                No trained checkpoint is loaded. This output demonstrates the video-processing pipeline only and cannot establish whether the video is authentic or manipulated.
              </p>
            </div>
          </div>
        )}

        {/* Evidence Identifier & Verdict Ribbon */}
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pb-4 border-b border-navy-800">
          <div>
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-widest block">
              EVIDENCE ITEM IDENTIFIER
            </span>
            <span className="text-2xl font-bold font-mono text-white tracking-tight">{evidenceId}</span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <StatusBadge label={verdict} isMock={isMock} />
            <span className="text-xs font-mono text-slate-300 px-2.5 py-1 rounded-md bg-navy-950 border border-navy-800">
              {isMock || probability === null ? 'Score unavailable' : `P(Fake): ${Math.round(probability * 100)}%`}
            </span>
            {!isMock && rawLogit !== null && (
              <span className="text-xs font-mono text-cyan-400 px-2 py-1 rounded-md bg-navy-950 border border-navy-800">
                s = {Number(rawLogit).toFixed(4)}
              </span>
            )}
          </div>
        </div>

        {/* Evidence Specifications Multi-Column Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
          {/* Col 1: File Artifact Specifications */}
          <div className="p-4 rounded-lg bg-navy-950/80 border border-navy-800 space-y-2">
            <h4 className="text-[10px] font-mono uppercase text-slate-400 font-semibold flex items-center gap-1.5">
              <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
              File Artifact Specifications
            </h4>
            <div className="flex justify-between items-center py-1 border-b border-navy-850">
              <span className="text-slate-400">File Name:</span>
              <span className="font-mono text-slate-200 truncate max-w-[150px]" title={filename}>{filename}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-navy-850">
              <span className="text-slate-400">Payload Size:</span>
              <span className="font-mono text-slate-200">{(fileSizeBytes / 1024).toFixed(1)} KB</span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-slate-400">Media Modality:</span>
              <span className="font-mono text-slate-200">{isImage ? 'Single-Image Face Crop' : 'Temporal Video Sequence'}</span>
            </div>
          </div>

          {/* Col 2: Model & Provenance */}
          <div className="p-4 rounded-lg bg-navy-950/80 border border-navy-800 space-y-2">
            <h4 className="text-[10px] font-mono uppercase text-slate-400 font-semibold flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-brand-lightBlue" />
              Model & Checkpoint Provenance
            </h4>
            <div className="flex justify-between items-center py-1 border-b border-navy-850">
              <span className="text-slate-400">Architecture:</span>
              <span className="font-mono text-slate-200">{modelArchitecture}</span>
            </div>
            <div className="flex justify-between items-center py-1 border-b border-navy-850">
              <span className="text-slate-400">Feature Layer:</span>
              <span className="font-mono text-cyan-300"><code>conv_head</code> (1,792 ch)</span>
            </div>
            <div className="flex justify-between items-center py-1">
              <span className="text-slate-400">Checkpoint:</span>
              <span className="font-mono text-amber-300">
                {isMock ? `Not loaded (${checkpointType})` : checkpointType}
              </span>
            </div>
          </div>

          {/* Col 3: Examination & Video Tracking */}
          <div className="p-4 rounded-lg bg-navy-950/80 border border-navy-800 space-y-2">
            <h4 className="text-[10px] font-mono uppercase text-slate-400 font-semibold flex items-center gap-1.5">
              <Calendar className="w-3.5 h-3.5 text-purple-400" />
              Examination Provenance
            </h4>
            <div className="flex justify-between items-center py-1 border-b border-navy-850">
              <span className="text-slate-400">Exam Timestamp:</span>
              <span className="font-mono text-slate-200 text-[11px]">{new Date(timestamp).toUTCString()}</span>
            </div>
            {!isImage && videoMeta ? (
              <>
                <div className="flex justify-between items-center py-1 border-b border-navy-850">
                  <span className="text-slate-400">Sampled Frames (k):</span>
                  <span className="font-mono text-slate-200">{videoMeta.num_frames_scanned || frames.length} frames</span>
                </div>
                <div className="flex justify-between items-center py-1">
                  <span className="text-slate-400">Face Detector:</span>
                  <span className="font-mono text-slate-200">OpenCV YuNet ONNX</span>
                </div>
              </>
            ) : (
              <div className="flex justify-between items-center py-1">
                <span className="text-slate-400">Decision Threshold:</span>
                <span className="font-mono text-slate-200">{isMock ? 'Unavailable in demo mode' : `&tau; = ${analysisObj?.prediction?.decision_threshold ?? 0.5}`}</span>
              </div>
            )}
          </div>
        </div>

        {/* Cryptographic SHA-256 Digest Box */}
        <div className="p-4 rounded-lg bg-navy-950 border border-navy-800 space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <Hash className="w-3.5 h-3.5 text-cyan-400" />
              Cryptographic Integrity Hash (SHA-256)
            </span>
            <button
              onClick={handleCopyHash}
              className="text-[11px] text-cyan-400 hover:text-cyan-300 flex items-center gap-1 bg-navy-900 px-2 py-0.5 rounded border border-navy-800 transition-colors"
            >
              {copiedHash ? (
                <>
                  <Check className="w-3 h-3 text-emerald-400" />
                  <span className="text-emerald-400 font-medium">Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3" />
                  <span>Copy Hash</span>
                </>
              )}
            </button>
          </div>
          <p className="font-mono text-xs text-cyan-300 break-all bg-navy-900/60 p-2.5 rounded border border-navy-850 select-all">
            {sha256}
          </p>
          <span className="text-[10px] text-slate-500 block">
            Cryptographic hash computed directly from the uploaded evidence payload to guarantee immutable forensic chain-of-custody.
          </span>
        </div>

        {/* Visual Forensic Evidence (Tri-Pane Display) */}
        {(visualEvidence.originalUrl || visualEvidence.heatmapUrl || visualEvidence.overlayUrl) && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <Eye className="w-4 h-4 text-brand-lightBlue" />
                Visual Forensic Evidence & Spatial Interpretability
              </h3>
              <span className="text-[11px] font-mono text-slate-400">
                Layer: <code>{targetLayerName}</code> &bull; Jet Colormap (&alpha; = 0.4)
              </span>
            </div>

            <p className="text-[11px] text-slate-400">
              {visualEvidence.label}. Grad-CAM backpropagates continuous logit gradients into terminal convolutional feature activations.
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-1">
              {/* Panel 1: Original Face Crop */}
              <div className="p-3 rounded-lg bg-navy-950/80 border border-navy-800 text-center space-y-2">
                <span className="text-[10px] font-mono uppercase text-slate-400 font-medium block">
                  1. Evidence Face Crop
                </span>
                <div className="aspect-square bg-navy-900 rounded-md overflow-hidden flex items-center justify-center border border-navy-850">
                  {visualEvidence.originalUrl ? (
                    <img
                      src={visualEvidence.originalUrl}
                      alt="Original Evidence Crop"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-slate-500">Not Available</span>
                  )}
                </div>
                <span className="text-[10px] text-slate-400 block font-mono">
                  Input RGB Image / Crop (380&times;380 model input)
                </span>
              </div>

              {/* Panel 2: Grad-CAM Heatmap */}
              <div className="p-3 rounded-lg bg-navy-950/80 border border-navy-800 text-center space-y-2">
                <span className="text-[10px] font-mono uppercase text-slate-400 font-medium block">
                  2. Grad-CAM Saliency Map
                </span>
                <div className="aspect-square bg-navy-900 rounded-md overflow-hidden flex items-center justify-center border border-navy-850">
                  {visualEvidence.heatmapUrl ? (
                    <img
                      src={visualEvidence.heatmapUrl}
                      alt="Grad-CAM Saliency Map"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-slate-500">Not Available</span>
                  )}
                </div>
                <span className="text-[10px] text-slate-400 block font-mono">
                  Normalized Activation [0.0, 1.0]
                </span>
              </div>

              {/* Panel 3: Saliency Overlay */}
              <div className="p-3 rounded-lg bg-navy-950/80 border border-navy-800 text-center space-y-2">
                <span className="text-[10px] font-mono uppercase text-slate-400 font-medium block">
                  3. Forensic Saliency Overlay
                </span>
                <div className="aspect-square bg-navy-900 rounded-md overflow-hidden flex items-center justify-center border border-navy-850">
                  {visualEvidence.overlayUrl ? (
                    <img
                      src={visualEvidence.overlayUrl}
                      alt="Forensic Saliency Overlay"
                      className="w-full h-full object-contain"
                    />
                  ) : (
                    <span className="text-xs text-slate-500">Not Available</span>
                  )}
                </div>
                <span className="text-[10px] text-slate-400 block font-mono">
                  Blended Spatial Overlay (&alpha;=0.4)
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Explainable Forensic Narrative */}
        <div className="p-4 rounded-lg bg-navy-950/90 border border-brand-blue/30 space-y-2">
          <h3 className="text-xs font-semibold text-brand-lightBlue uppercase tracking-wider flex items-center gap-2">
            <Info className="w-4 h-4 text-brand-lightBlue" />
            Explainable Forensic Narrative
          </h3>
          <p className="text-xs text-slate-300 leading-relaxed font-sans">
            {narrativeText}
          </p>
        </div>

        {/* Sampled Frame Temporal Breakdown (Video Evidence) */}
        {!isImage && frames.length > 0 && (
          <div className="space-y-3 pt-2">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
                <Film className="w-4 h-4 text-purple-400" />
                Sampled Frame Temporal Breakdown ({frames.length} frames)
              </h3>
              <span className="text-[11px] font-mono text-slate-400">
                Uniform Bin-Centered Sampling &bull; IoU Continuity Tracking
              </span>
            </div>

            <div className="overflow-x-auto rounded-lg border border-navy-800">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-navy-950/90 text-[11px] text-slate-400 font-mono uppercase tracking-wider border-b border-navy-800">
                  <tr>
                    <th className="py-2.5 px-3">Frame #</th>
                    <th className="py-2.5 px-3">Timestamp</th>
                    <th className="py-2.5 px-3">Crop Preview</th>
                    <th className="py-2.5 px-3">Tracking</th>
                    <th className="py-2.5 px-3">Bounding Box</th>
                    <th className="py-2.5 px-3">Logit (s)</th>
                    <th className="py-2.5 px-3">Score P_i</th>
                    <th className="py-2.5 px-3">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-navy-850 font-mono text-[11px]">
                  {frames.map((frame, idx) => (
                    <tr key={idx} className="hover:bg-navy-800/40 transition-colors">
                      <td className="py-2 px-3 text-white font-bold">#{frame.frame_index}</td>
                      <td className="py-2 px-3 text-slate-300">{Number(frame.timestamp_sec).toFixed(3)}s</td>
                      <td className="py-2 px-3">
                        {frame.crop_data_url ? (
                          <img
                            src={frame.crop_data_url}
                            alt={`Frame #${frame.frame_index}`}
                            className="w-8 h-8 rounded object-cover border border-navy-750"
                          />
                        ) : (
                          <span className="text-slate-600">No Crop</span>
                        )}
                      </td>
                      <td className="py-2 px-3">
                        <span className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-sans font-medium ${
                          frame.tracking_status === 'detected'
                            ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                            : frame.tracking_status === 'neighbour'
                            ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20'
                            : 'bg-slate-700/30 text-slate-400'
                        }`}>
                          {frame.tracking_status}
                        </span>
                      </td>
                      <td className="py-2 px-3 text-slate-400 text-[10px]">
                        {frame.box ? `[${frame.box.map(v => Math.round(v)).join(', ')}]` : 'N/A'}
                      </td>
                      <td className="py-2 px-3 text-cyan-300">
                        {frame.logit !== null && frame.logit !== undefined ? frame.logit.toFixed(4) : '—'}
                      </td>
                      <td className="py-2 px-3 text-slate-200">
                        {frame.probability !== null && frame.probability !== undefined
                          ? `${(frame.probability * 100).toFixed(1)}%`
                          : '—'}
                      </td>
                      <td className="py-2 px-3">
                        <span className={`font-sans font-semibold text-[10px] ${
                          isMock
                            ? 'text-amber-400'
                            : frame.predicted_label === 'fake'
                            ? 'text-rose-400'
                            : frame.predicted_label === 'real'
                            ? 'text-emerald-400'
                            : 'text-slate-500'
                        }`}>
                          {isMock ? 'INCONCLUSIVE' : (frame.predicted_label || '—').toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Legal Limitations & Chain of Custody Box */}
        <div className="p-4 rounded-lg bg-rose-500/10 border border-rose-500/20 text-xs text-rose-300 space-y-1.5 leading-relaxed">
          <div className="flex items-center gap-2 font-semibold text-rose-200">
            <ShieldAlert className="w-4 h-4 text-rose-400" />
            <span>Evidentiary Limitations & Forensic Legal Notice</span>
          </div>
          <p className="text-[11px] text-rose-300/90">
            This examination sheet reflects automated deep learning feature extraction. Neural probabilities are statistical inferences and do NOT constitute self-authenticating judicial proof under Federal Rule of Evidence 901/902 or equivalent legal standards. Saliency maps highlight receptive fields influencing internal activations, not certified physical tampering masks. Chain of custody must be established through independent evidentiary documentation.
          </p>
        </div>

        {/* Report Provenance & Academic Footer */}
        <div className="pt-2 text-center text-[10px] text-slate-500 font-mono border-t border-navy-800">
          Multimodal Deepfake Detection and Explainable Digital Forensics System &bull; Final-Year Capstone Project &bull; October 2026
        </div>
      </div>
    </div>
  );
}
