"""FastAPI REST API server for Deepfake Detection & Explainable Digital Forensics.

Wires the existing DeepfakeDetector, FaceTrackerPipeline, and GradCAMExplainer modules
into clean, secure REST endpoints for the React frontend dashboard.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

from src.data.prepare_data import compute_file_sha256
from src.explainability.gradcam import GradCAMComputationError, GradCAMExplainer
from src.image_detection.detector import DeepfakeDetector, FramePrediction, ImagePrediction
from src.video_detection.face_tracker import FaceTrackerPipeline

app = FastAPI(
    title="Deepfake Detection & Explainable Digital Forensics API",
    description="Multimodal image and video forgery analysis with Grad-CAM spatial interpretability.",
    version="1.0.0",
)

# Enable CORS for local Vite development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------------------
# SINGLETON MODEL & SERVICE INITIALIZATION
# ------------------------------------------------------------------------------

CANDIDATE_CHECKPOINT_PATHS = [
    Path("models/image/model_best_fp16.pt"),
    Path("models/image/model_best.pt"),
]
MODEL_ARCHITECTURE = "efficientnet_b4"
MODEL_INPUT_SIZE = 380
YUNET_PATH = Path("models/yunet/face_detection_yunet_2023mar.onnx")

detector_instance: Optional[DeepfakeDetector] = None
explainer_instance: Optional[GradCAMExplainer] = None
tracker_instance: Optional[FaceTrackerPipeline] = None


def find_checkpoint_path() -> Optional[Path]:
    """Find an explicitly named best-inference checkpoint, never a training-resume file."""
    for p in CANDIDATE_CHECKPOINT_PATHS:
        if p.is_file():
            return p
    return None


def get_detector() -> DeepfakeDetector:
    """Retrieve or initialize the singleton DeepfakeDetector instance.
    
    A new checkpoint is only activated after strict architecture and training
    metadata validation.
    """
    global detector_instance, explainer_instance
    ckpt = find_checkpoint_path()

    if detector_instance is None:
        if ckpt is not None:
            detector_instance = DeepfakeDetector(
                model_name=MODEL_ARCHITECTURE,
                checkpoint_path=ckpt,
                checkpoint_type="deepfake_trained",
                device="cpu",
                default_threshold=0.5,
                input_size=MODEL_INPUT_SIZE,
            )
        else:
            # Running with uninitialized mock model for interface verification
            detector_instance = DeepfakeDetector(
                model_name=MODEL_ARCHITECTURE,
                checkpoint_path=None,
                checkpoint_type="uninitialized_mock",
                device="cpu",
                default_threshold=0.5,
                allow_uninitialized=True,
                input_size=MODEL_INPUT_SIZE,
            )
        explainer_instance = GradCAMExplainer(detector_instance)
    elif detector_instance.checkpoint_type == "uninitialized_mock" and ckpt is not None:
        # Seamlessly load newly available trained checkpoint
        detector_instance.load_checkpoint(ckpt, checkpoint_type="deepfake_trained")
        explainer_instance = GradCAMExplainer(detector_instance)

    return detector_instance


def get_explainer() -> GradCAMExplainer:
    """Retrieve or initialize the Grad-CAM explainer."""
    global explainer_instance
    if explainer_instance is None:
        get_detector()
    return explainer_instance


def get_tracker() -> FaceTrackerPipeline:
    """Retrieve or initialize the FaceTrackerPipeline."""
    global tracker_instance
    yunet_file = YUNET_PATH if YUNET_PATH.is_file() else None
    if tracker_instance is None or (yunet_file and tracker_instance.detector is None):
        tracker_instance = FaceTrackerPipeline(
            yunet_model_path=yunet_file,
            crop_size=get_detector().input_size,
        )
    return tracker_instance


def encode_image_base64(img_rgb: np.ndarray, format: str = "PNG") -> str:
    """Encode an RGB uint8 numpy image array into a base64 Data URL."""
    pil_img = Image.fromarray(img_rgb.astype(np.uint8))
    buffered = io.BytesIO()
    pil_img.save(buffered, format=format)
    b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{b64_str}"


def encode_heatmap_base64(heatmap: np.ndarray) -> str:
    """Encode normalized float32 heatmap [0, 1] as an 8-bit grayscale base64 Data URL."""
    u8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
    return encode_image_base64(u8, format="PNG")


# ------------------------------------------------------------------------------
# REST ROUTES
# ------------------------------------------------------------------------------


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Report backend system status, model checkpoint readiness, and hardware device."""
    det = get_detector()
    is_trained = det.checkpoint_type == "deepfake_trained"
    yunet_available = YUNET_PATH.is_file()

    return {
        "status": "healthy",
        "service": "Multimodal Deepfake Detection & Digital Forensics API",
        "model_architecture": det.model_name,
        "checkpoint_type": det.checkpoint_type,
        "is_model_trained": is_trained,
        "model_readiness": "trained" if is_trained else "demo",
        "checkpoint_path": det.checkpoint_path,
        "input_size": det.input_size,
        "labels": {"real": 0, "fake": 1},
        "device": str(det.device),
        "yunet_detector_available": yunet_available,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "disclaimer": (
            "System operational. When running with uninitialized mock weights, "
            "predictions demonstrate pipeline mechanics and do NOT reflect real forensic findings."
        ),
    }


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    alpha: float = Query(0.4, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """Perform deepfake forgery inference and Grad-CAM interpretability on an uploaded image."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Calculate real SHA-256 hash of the uploaded payload
    file_sha256 = hashlib.sha256(contents).hexdigest()

    # Decode image using OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image. Supported formats: JPEG, PNG, WebP, BMP.",
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    det = get_detector()
    explainer = get_explainer()
    # Model inference
    try:
        prediction: ImagePrediction = det.predict_image(img_rgb, threshold=threshold)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Grad-CAM explainability
    if det.checkpoint_type == "deepfake_trained":
        try:
            cam_result = explainer.generate_cam(img_rgb, alpha=alpha)
            heatmap_b64 = encode_heatmap_base64(cam_result.heatmap)
            overlay_b64 = encode_image_base64(cam_result.overlay_rgb)
            orig_b64 = encode_image_base64(cam_result.original_rgb)
        except Exception as e:
            heatmap_b64 = None
            overlay_b64 = None
            orig_b64 = encode_image_base64(img_rgb)
            cam_error = f"Grad-CAM computation error: {str(e)}"
        else:
            cam_error = None
    else:
        heatmap_b64 = None
        overlay_b64 = None
        orig_b64 = encode_image_base64(img_rgb)
        cam_error = None

    evidence_id = f"IMG-{file_sha256[:10].upper()}"
    exam_time = datetime.now(timezone.utc).isoformat()

    pred_dict = prediction.to_dict()
    is_mock = det.checkpoint_type != "deepfake_trained"
    if is_mock:
        pred_dict["predicted_label"] = "inconclusive"
        pred_dict["raw_logit"] = None
        pred_dict["probability"] = None

    return {
        "evidence_id": evidence_id,
        "filename": file.filename or "unknown.jpg",
        "file_size_bytes": len(contents),
        "sha256": file_sha256,
        "dimensions": {"width": w, "height": h},
        "inference_input": "rgb_image_resize",
        "examination_timestamp": exam_time,
        "model_status": "trained" if not is_mock else "demo",
        "model_architecture": det.model_name,
        "checkpoint_type": det.checkpoint_type,
        "checkpoint_path": det.checkpoint_path,
        "input_size": det.input_size,
        "prediction": pred_dict,
        "explainability": {
            "target_layer_name": explainer.target_layer_name,
            "heatmap_data_url": heatmap_b64,
            "overlay_data_url": overlay_b64,
            "original_data_url": orig_b64,
            "alpha": alpha,
            "error": cam_error,
            "forensic_disclaimer": (
                "Grad-CAM is unavailable because no trained checkpoint is loaded. This output demonstrates the image-processing pipeline only and cannot establish whether the image is authentic or manipulated."
                if is_mock
                else (
                    "Grad-CAM heatmaps highlight spatial receptive fields that influenced the model's verdict. "
                    "They do NOT constitute ground-truth manipulation masks or certified legal proof."
                )
            ),
        },
        "is_mock_prediction": is_mock,
    }


@app.post("/api/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    max_frames: int = Query(16, ge=4, le=64),
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
) -> Dict[str, Any]:
    """Sample temporal frames, track faces across frames, score face crops, and aggregate video verdict."""
    # Verify YuNet face detection weights exist locally before attempting video extraction
    if not YUNET_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "YuNet face detection model weights are missing. "
                f"Video analysis requires '{YUNET_PATH.name}' in '{YUNET_PATH.parent}/' "
                "to extract facial crops and track faces across video frames."
            ),
        )

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded video file is empty.")

    file_sha256 = hashlib.sha256(contents).hexdigest()

    # Write video to temporary file on disk for OpenCV VideoCapture decoding
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(contents)
        tmp_video_path = tmp_file.name

    try:
        tracker = get_tracker()
        det = get_detector()
        explainer = get_explainer()

        # Extract frames and track faces
        try:
            track_result = tracker.extract_from_video(tmp_video_path, max_frames=max_frames)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process video: {str(e)}")

        if track_result.status == "unreadable_video":
            raise HTTPException(
                status_code=400,
                detail=f"Could not decode video file: {track_result.error_message or 'Invalid or corrupted video container.'}",
            )

        is_mock = det.checkpoint_type != "deepfake_trained"
        effective_threshold = float(threshold if threshold is not None else det.default_threshold)
        if is_mock:
            frame_predictions = [
                FramePrediction(
                    video_path=face.video_path,
                    frame_index=face.frame_index,
                    timestamp_sec=face.timestamp_sec,
                    box=face.box,
                    tracking_status=face.tracking_status,
                    has_face=face.crop_rgb is not None,
                    prediction=None,
                )
                for face in track_result.extracted_faces
            ]
        else:
            frame_predictions = det.predict_video_faces(track_result.extracted_faces, threshold=threshold)

        # Video-level aggregation: Arithmetic mean pooling over valid face crop logits
        valid_logits = [
            fp.prediction.raw_logit
            for fp in frame_predictions
            if fp.has_face and fp.prediction is not None
        ]

        if is_mock and track_result.valid_face_count:
            mean_logit = None
            video_probability = None
            video_verdict = "inconclusive"
            keyframe_fp = None
            keyframe_overlay_b64 = None
            keyframe_heatmap_b64 = None
            keyframe_crop_b64 = None
        elif valid_logits:
            mean_logit = float(np.mean(valid_logits))
            video_probability = float(1.0 / (1.0 + np.exp(-mean_logit)))
            video_verdict = "fake" if video_probability >= effective_threshold else "real"

            # Identify Peak Anomaly Keyframe (argmax logit)
            keyframe_idx = int(np.argmax(valid_logits))
            keyframe_fp = [fp for fp in frame_predictions if fp.has_face and fp.prediction is not None][keyframe_idx]
            
            # Find the ExtractedFace matching the keyframe
            keyframe_face = next(
                (f for f in track_result.extracted_faces if f.frame_index == keyframe_fp.frame_index and f.crop_rgb is not None),
                None
            )

            # Generate Grad-CAM for the peak keyframe
            if keyframe_face and keyframe_face.crop_rgb is not None:
                try:
                    cam_result = explainer.generate_cam(keyframe_face.crop_rgb)
                    keyframe_overlay_b64 = encode_image_base64(cam_result.overlay_rgb)
                    keyframe_heatmap_b64 = encode_heatmap_base64(cam_result.heatmap)
                    keyframe_crop_b64 = encode_image_base64(keyframe_face.crop_rgb)
                except Exception:
                    keyframe_overlay_b64 = None
                    keyframe_heatmap_b64 = None
                    keyframe_crop_b64 = encode_image_base64(keyframe_face.crop_rgb)
            else:
                keyframe_overlay_b64 = None
                keyframe_heatmap_b64 = None
                keyframe_crop_b64 = None
        else:
            mean_logit = None
            video_probability = None
            video_verdict = "unscorable"
            keyframe_fp = None
            keyframe_overlay_b64 = None
            keyframe_heatmap_b64 = None
            keyframe_crop_b64 = None

        # Build serializable frame cards
        serialized_frames = []
        for fp, face in zip(frame_predictions, track_result.extracted_faces):
            crop_b64 = encode_image_base64(face.crop_rgb) if face.crop_rgb is not None else None
            if face.crop_rgb is None:
                pred_label = "no_face"
            elif is_mock:
                pred_label = "inconclusive"
            elif not fp.prediction:
                pred_label = "no_face"
            else:
                pred_label = fp.prediction.predicted_label

            serialized_frames.append({
                "frame_index": fp.frame_index,
                "timestamp_sec": round(fp.timestamp_sec, 3),
                "has_face": fp.has_face,
                "tracking_status": fp.tracking_status,
                "box": list(fp.box) if fp.box else None,
                "crop_data_url": crop_b64,
                "logit": round(fp.prediction.raw_logit, 4) if fp.prediction else None,
                "probability": round(fp.prediction.probability, 4) if fp.prediction else None,
                "predicted_label": pred_label,
            })

        evidence_id = f"VID-{file_sha256[:10].upper()}"

        return {
            "evidence_id": evidence_id,
            "filename": file.filename or "unknown.mp4",
            "file_size_bytes": len(contents),
            "sha256": file_sha256,
            "model_status": "demo" if is_mock else "trained",
            "model_architecture": det.model_name,
            "checkpoint_type": det.checkpoint_type,
            "checkpoint_path": det.checkpoint_path,
            "input_size": det.input_size,
            "video_metadata": {
                "num_frames_scanned": track_result.num_frames_scanned,
                "num_faces_extracted": track_result.num_faces_extracted,
                "num_unscorable_frames": track_result.num_unscorable_frames,
                "fps": track_result.fps,
                "video_dimensions": track_result.original_dimensions,
            },
            "video_prediction": {
                "mean_logit": round(mean_logit, 4) if mean_logit is not None else None,
                "probability": round(video_probability, 4) if video_probability is not None else None,
                "predicted_label": video_verdict,
                "valid_frames_count": len(valid_logits),
                "aggregation_method": "arithmetic_mean_logits",
                "decision_threshold": None if is_mock else effective_threshold,
                "checkpoint_type": det.checkpoint_type,
                "forensic_disclaimer": (
                    "No trained checkpoint is loaded. This output demonstrates the video-processing pipeline only and cannot establish whether the video is authentic or manipulated."
                    if is_mock
                    else (
                        "Video aggregation averages facial frame logits. Consecutive frames are temporally "
                        "correlated and must not be treated as independent observations."
                    )
                ),
            },
            "keyframe_explainability": {
                "frame_index": keyframe_fp.frame_index if keyframe_fp else None,
                "timestamp_sec": round(keyframe_fp.timestamp_sec, 3) if keyframe_fp else None,
                "heatmap_data_url": keyframe_heatmap_b64,
                "overlay_data_url": keyframe_overlay_b64,
                "crop_data_url": keyframe_crop_b64,
                "target_layer_name": explainer.target_layer_name,
            },
            "frames": serialized_frames,
            "examination_timestamp": datetime.now(timezone.utc).isoformat(),
            "is_mock_prediction": is_mock,
        }

    finally:
        # Clean up temporary video file safely
        if os.path.exists(tmp_video_path):
            try:
                os.remove(tmp_video_path)
            except OSError:
                pass


def generate_forensic_narrative(
    media_type: str,
    verdict: str,
    is_mock: bool,
    probability: Optional[float],
    raw_logit: Optional[float] = None,
    filename: str = "evidence",
    frames_count: Optional[int] = None,
    keyframe_idx: Optional[int] = None,
    keyframe_time: Optional[float] = None,
) -> str:
    """Generate an explainable digital forensics narrative interpreting model findings."""
    prob_pct = round(probability * 100, 1) if probability is not None else None
    logit_str = f"{raw_logit:.4f}" if raw_logit is not None else None

    if is_mock:
        if media_type.lower() == "video":
            return (
                f"TEMPORAL PIPELINE DEMONSTRATION NOTICE: Video evidence '{filename}' was processed across "
                f"{frames_count or 16} sampled temporal frames using OpenCV YuNet ONNX face detection and IoU tracking. "
                f"Because no trained checkpoint weights are currently loaded, this output demonstrates the video-processing "
                f"pipeline only. No classification score or authenticity finding is available."
            )
        else:
            return (
                f"DEMONSTRATION MODE NOTICE: Image evidence '{filename}' was received by the image-analysis pipeline. "
                f"No trained checkpoint is loaded, so no classification score, authenticity finding, or Grad-CAM attribution is available."
            )

    if probability is None or logit_str is None:
        return (
            f"FORENSIC ASSESSMENT: The report does not contain a measured model score for '{filename}'. "
            f"No probability-based classification narrative can be generated."
        )

    if verdict.lower() == "fake":
        if media_type.lower() == "video":
            kf_str = f"Peak anomaly localized at Keyframe #{keyframe_idx} (timestamp: {keyframe_time:.2f}s). " if keyframe_idx is not None and keyframe_time is not None else ""
            return (
                f"TEMPORAL FORENSIC ANALYSIS: Video evidence '{filename}' was evaluated across {frames_count or 16} frames. "
                f"Temporal facial crops were tracked via OpenCV YuNet and scored. Arithmetic mean logit pooling produced "
                f"an aggregate continuous logit of {logit_str}, corresponding to a calibrated manipulation probability P(Fake) = {prob_pct}%, "
                f"exceeding the configured decision threshold. {kf_str}Grad-CAM indicates regions that influenced the model score; "
                f"it does not prove which pixels were manipulated."
            )
        else:
            return (
                f"SINGLE-IMAGE FORENSIC ANALYSIS: Facial examination of '{filename}' yielded a continuous logit of {logit_str}, "
                f"corresponding to a model probability P(Fake) = {prob_pct}%, exceeding the configured decision threshold. "
                f"Grad-CAM indicates regions that influenced the score; it does not prove which pixels were manipulated."
            )
    elif verdict.lower() == "real":
        if media_type.lower() == "video":
            return (
                f"TEMPORAL FORENSIC ANALYSIS: Video evidence '{filename}' was evaluated across {frames_count or 16} temporal frames. "
                f"Arithmetic mean logit pooling yielded an aggregate continuous logit of {logit_str}, corresponding to an aggregate "
                f"model probability P(Fake) = {prob_pct}%, below the configured decision threshold. This is a model output, not proof of authenticity."
            )
        else:
            return (
                f"SINGLE-IMAGE FORENSIC ANALYSIS: Facial examination of '{filename}' yielded a continuous logit of {logit_str}, "
                f"corresponding to a model probability P(Fake) = {prob_pct}%, below the configured decision threshold. "
                f"This is a model output, not proof that the image is authentic."
            )
    else:
        return (
            f"FORENSIC ASSESSMENT: Examination of '{filename}' was inconclusive. "
            f"No valid trained-model classification score is available."
        )


@app.get("/api/report/html/{evidence_id}", response_class=HTMLResponse)
def get_html_report(
    evidence_id: str,
    filename: str = Query("evidence.file"),
    sha256: str = Query(""),
    verdict: str = Query("inconclusive"),
    probability: Optional[float] = Query(None, ge=0.0, le=1.0),
    decision_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    media_type: str = Query("image"),
    is_mock: Optional[bool] = Query(None),
    examination_timestamp_utc: Optional[str] = Query(None),
    raw_logit: Optional[float] = Query(None),
    file_size_bytes: Optional[int] = Query(None),
    frames_count: Optional[int] = Query(None),
    keyframe_idx: Optional[int] = Query(None),
    keyframe_time: Optional[float] = Query(None),
) -> str:
    """Generate a clean, printable HTML digital forensic examination report matching Colab analysis structure."""
    det = get_detector()
    
    # Determine effective mock status
    effective_is_mock = det.checkpoint_type != "deepfake_trained" or is_mock is True

    if effective_is_mock:
        badge_label = "INCONCLUSIVE — DEMO MODE"
        verdict_color = "#f59e0b"
        final_verdict = "inconclusive"
    else:
        final_verdict = "inconclusive" if verdict.lower() in ("inconclusive", "unscorable") else verdict.lower()
        badge_label = "INCONCLUSIVE" if final_verdict == "inconclusive" else verdict.upper()
        verdict_color = "#ef4444" if final_verdict == "fake" else ("#10b981" if final_verdict == "real" else "#64748b")

    now_str = examination_timestamp_utc or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    model_label = (
        "EfficientNet-B4 (Demo Mode - Uninitialized Mock Weights)"
        if effective_is_mock
        else f"EfficientNet-B4 (Trained Model - {Path(det.checkpoint_path).name if det.checkpoint_path else 'model_best_fp16.pt'})"
    )

    narrative = generate_forensic_narrative(
        media_type=media_type,
        verdict=final_verdict,
        is_mock=effective_is_mock,
        probability=probability,
        raw_logit=raw_logit,
        filename=filename,
        frames_count=frames_count,
        keyframe_idx=keyframe_idx,
        keyframe_time=keyframe_time,
    )

    demo_banner = ""
    if effective_is_mock:
        demo_banner = """
        <div style="background-color: #fef3c7; border-left: 4px solid #f59e0b; padding: 14px 18px; margin-bottom: 24px; font-size: 13px; color: #92400e; border-radius: 6px; line-height: 1.5;">
            <strong style="display: block; margin-bottom: 4px; font-size: 14px;">DEMO MODE NOTICE</strong>
            No trained checkpoint is loaded. This output demonstrates the image/video-processing pipeline only and cannot establish whether the evidence is authentic or manipulated.
        </div>"""

    size_row = f"<tr><th>Evidence Payload Size</th><td>{file_size_bytes:,} bytes ({(file_size_bytes/1024):.1f} KB)</td></tr>" if file_size_bytes else ""
    logit_row = (
        f"<tr><th>Continuous Logit Score (s)</th><td><code>{raw_logit:.4f}</code></td></tr>"
        if raw_logit is not None and not effective_is_mock
        else ""
    )
    probability_row = (
        f"<tr><th>Probability P(Fake)</th><td><strong>{probability:.4f}</strong> ({probability * 100:.1f}%)</td></tr>"
        if probability is not None and not effective_is_mock
        else ""
    )
    threshold_value = decision_threshold if decision_threshold is not None else det.default_threshold
    threshold_row = (
        f"<tr><th>Decision Threshold</th><td><code>&tau; = {threshold_value:.4f}</code></td></tr>"
        if not effective_is_mock
        else ""
    )
    frames_row = f"<tr><th>Sampled Temporal Frames</th><td>{frames_count} frames (Uniform sampling, OpenCV YuNet tracking)</td></tr>" if (media_type.lower() == "video" and frames_count) else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Digital Forensics Examination Report - {evidence_id}</title>
    <style>
        body {{ font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif; background-color: #f8fafc; color: #1e293b; margin: 0; padding: 32px 16px; }}
        .report-card {{ max-width: 860px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.06); padding: 40px; }}
        .header {{ border-bottom: 2px solid #0f172a; padding-bottom: 20px; margin-bottom: 24px; }}
        .header-top {{ display: flex; justify-content: space-between; align-items: flex-start; }}
        .header h1 {{ font-size: 24px; margin: 0 0 6px 0; color: #0f172a; letter-spacing: -0.02em; }}
        .header p {{ margin: 0; color: #64748b; font-size: 13px; }}
        .badge {{ display: inline-block; padding: 6px 16px; border-radius: 9999px; font-weight: 700; font-size: 13px; letter-spacing: 0.05em; text-transform: uppercase; color: #fff; background-color: {verdict_color}; }}
        .section-title {{ font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #475569; margin: 28px 0 12px 0; border-bottom: 1px solid #f1f5f9; padding-bottom: 6px; }}
        .table {{ width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13.5px; }}
        .table th, .table td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }}
        .table th {{ background-color: #f8fafc; color: #475569; width: 32%; font-weight: 600; }}
        .narrative-box {{ background-color: #f1f5f9; border-left: 4px solid #0284c7; padding: 16px 20px; border-radius: 6px; font-size: 13.5px; line-height: 1.6; color: #0f172a; margin-top: 8px; }}
        .disclaimer-box {{ background-color: #fef2f2; border-left: 4px solid #ef4444; padding: 16px 20px; margin-top: 24px; font-size: 12.5px; color: #991b1b; line-height: 1.6; border-radius: 6px; }}
        .footer {{ margin-top: 36px; font-size: 12px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
        @media print {{ body {{ padding: 0; background: #fff; }} .report-card {{ border: none; box-shadow: none; padding: 0; }} }}
    </style>
</head>
<body>
    <div class="report-card">
        <div class="header">
            <div class="header-top">
                <div>
                    <h1>Digital Forensics Examination Sheet</h1>
                    <p>Multimodal Deepfake Detection and Explainable Digital Forensics System &bull; Academic Forensic Prototype</p>
                </div>
                <div>
                    <span class="badge">{badge_label}</span>
                </div>
            </div>
        </div>

        {demo_banner}

        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; background: #f8fafc; padding: 12px 16px; border-radius: 8px; border: 1px solid #e2e8f0;">
            <div>
                <span style="font-size: 12px; font-weight: 600; text-transform: uppercase; color: #64748b; letter-spacing: 0.05em;">Evidence Identifier:</span>
                <strong style="font-size: 17px; font-family: monospace; color: #0f172a; margin-left: 8px;">{evidence_id}</strong>
            </div>
            <div>
                <span style="font-size: 12px; color: #64748b;">Examination Date:</span>
                <span style="font-size: 13px; font-family: monospace; color: #334155; margin-left: 6px;">{now_str}</span>
            </div>
        </div>

        <div class="section-title">Evidence & Model Provenance</div>
        <table class="table">
            <tr><th>Original File Name</th><td style="font-family: monospace;">{filename}</td></tr>
            <tr><th>Cryptographic Hash (SHA-256)</th><td style="font-family: monospace; font-size: 12px; word-break: break-all;">{sha256}</td></tr>
            <tr><th>Media Type</th><td>{media_type.upper()} ({'Single-Image Facial Analysis' if media_type.lower() == 'image' else 'Temporal Video Sequence Analysis'})</td></tr>
            {size_row}
            {frames_row}
            <tr><th>Classification Verdict</th><td><strong>{badge_label}</strong></td></tr>
            {logit_row}
            {probability_row}
            {threshold_row}
            <tr><th>Model Backbone Architecture</th><td>{model_label}</td></tr>
            <tr><th>Explainability Method</th><td>Grad-CAM targeting <code>model.conv_head</code> (1,792 feature maps)</td></tr>
        </table>

        <div class="section-title">Explainable Forensic Narrative</div>
        <div class="narrative-box">
            {narrative}
        </div>

        <div class="disclaimer-box">
            <strong>Evidentiary Limitations & Forensic Legal Notice:</strong><br>
            Neural network classification scores represent statistical pattern matching and do NOT constitute self-authenticating judicial proof under Federal Rule of Evidence 901/902 or equivalent legal standards. Grad-CAM saliency heatmaps indicate internal receptive field activations that influenced the model, not certified physical tampering masks or compression boundary delineations. Formal legal custody must be established through independent evidentiary documentation.
        </div>

        <div class="footer">
            Final-Year Computer Science and Engineering Capstone Project &bull; Department of Computer Science and Engineering &bull; October 2026
        </div>
    </div>
</body>
</html>"""
    return html


@app.get("/api/report/json/{evidence_id}", response_class=JSONResponse)
def get_json_report(
    evidence_id: str,
    filename: str = Query("evidence.file"),
    sha256: str = Query(""),
    verdict: str = Query("inconclusive"),
    probability: Optional[float] = Query(None, ge=0.0, le=1.0),
    decision_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    media_type: str = Query("image"),
    is_mock: Optional[bool] = Query(None),
    examination_timestamp_utc: Optional[str] = Query(None),
    raw_logit: Optional[float] = Query(None),
    file_size_bytes: Optional[int] = Query(None),
    frames_count: Optional[int] = Query(None),
    keyframe_idx: Optional[int] = Query(None),
    keyframe_time: Optional[float] = Query(None),
) -> Dict[str, Any]:
    """Generate a structured JSON digital forensic examination record matching Colab structure."""
    det = get_detector()
    
    effective_is_mock = det.checkpoint_type != "deepfake_trained" or is_mock is True

    if effective_is_mock:
        final_verdict = "inconclusive"
        verdict_display = "INCONCLUSIVE — DEMO MODE"
    else:
        final_verdict = "inconclusive" if verdict.lower() in ("inconclusive", "unscorable") else verdict.lower()
        verdict_display = "INCONCLUSIVE" if final_verdict == "inconclusive" else verdict.upper()

    now_iso = examination_timestamp_utc or datetime.now(timezone.utc).isoformat()
    narrative = generate_forensic_narrative(
        media_type=media_type,
        verdict=final_verdict,
        is_mock=effective_is_mock,
        probability=probability,
        raw_logit=raw_logit,
        filename=filename,
        frames_count=frames_count,
        keyframe_idx=keyframe_idx,
        keyframe_time=keyframe_time,
    )

    limitations_text = (
        "No trained checkpoint is loaded. This output demonstrates the video-processing pipeline only "
        "and cannot establish whether the video is authentic or manipulated."
        if effective_is_mock
        else (
            "Model classification probabilities represent probabilistic neural inferences and do NOT "
            "constitute definitive legal or judicial proof of manipulation. Saliency heatmaps highlight "
            "spatial receptive fields that influenced the model, not certified physical tampering boundaries."
        )
    )

    return {
        "report_title": "Digital Forensic Examination Sheet",
        "report_version": "1.0.0",
        "academic_project": "Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos",
        "evidence_id": evidence_id,
        "filename": filename,
        "sha256": sha256,
        "media_type": media_type,
        "examination_timestamp_utc": now_iso,
        "evidence_metadata": {
            "evidence_id": evidence_id,
            "filename": filename,
            "sha256": sha256,
            "file_size_bytes": file_size_bytes,
            "media_type": media_type,
            "examination_timestamp_utc": now_iso,
        },
        "model_provenance": {
            "model_architecture": det.model_name,
            "checkpoint_type": det.checkpoint_type,
            "checkpoint_path": det.checkpoint_path,
            "is_mock_prediction": effective_is_mock,
            "model_status": "demo" if effective_is_mock else "trained",
            "device": str(det.device),
            "target_layer_name": "conv_head",
        },
        "classification": {
            "verdict": final_verdict,
            "verdict_display": verdict_display,
            "probability": round(probability, 4) if probability is not None and not effective_is_mock else None,
            "raw_logit": round(raw_logit, 4) if raw_logit is not None and not effective_is_mock else None,
            "checkpoint_type": det.checkpoint_type,
            "is_mock_prediction": effective_is_mock,
            "model_architecture": det.model_name,
            "checkpoint_path": det.checkpoint_path,
            "decision_threshold": (
                None
                if effective_is_mock
                else (decision_threshold if decision_threshold is not None else det.default_threshold)
            ),
        },
        "explainability": {
            "method": "Grad-CAM (Gradient-weighted Class Activation Mapping)",
            "target_layer": "conv_head (1,792 channels)",
            "narrative": narrative,
        },
        "forensic_limitations": limitations_text,
    }
