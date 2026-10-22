"""Core Face Extraction and Identity Tracking for Video Deepfake Forensics.

Adapted from the FaceForensics++ / DFD Kaggle Benchmark (ffb_video.py).
Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos.
Academic Capstone Project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ExtractedFace:
    """Represents a facial crop extracted from a specific video frame,

    maintaining complete evidentiary traceability.
    """

    video_path: str
    frame_index: int
    timestamp_sec: float
    box: Optional[Tuple[float, float, float, float]]  # (x, y, w, h)
    confidence: float
    tracking_status: str  # "detected", "neighbour", "none"
    crop_rgb: Optional[np.ndarray]  # (H, W, 3) uint8 RGB array or None if no face
    original_frame_shape: Tuple[int, int]  # (H, W)


@dataclass
class VideoExtractionResult:
    """Summary of face extraction across all sampled frames of a video."""

    video_path: str
    total_frames: int
    fps: float
    duration_sec: float
    sampled_indices: List[int]
    extracted_faces: List[ExtractedFace]
    valid_face_count: int
    status: str  # "success", "partial", "no_faces_detected", "unreadable_video"
    error_message: Optional[str] = None

    @property
    def num_frames_scanned(self) -> int:
        """Total number of video frames successfully scheduled and decoded."""
        return len(self.sampled_indices)

    @property
    def num_faces_extracted(self) -> int:
        """Number of valid facial crops extracted across sampled frames."""
        return self.valid_face_count

    @property
    def num_unscorable_frames(self) -> int:
        """Number of sampled frames where no face could be extracted or cropped."""
        return max(0, len(self.extracted_faces) - self.valid_face_count)

    @property
    def original_dimensions(self) -> Optional[Tuple[int, int]]:
        """Original video frame dimensions (width, height), if any frame was decoded."""
        for face in self.extracted_faces:
            if face.original_frame_shape:
                return (face.original_frame_shape[1], face.original_frame_shape[0])
        return None


# ==============================================================================
# GEOMETRY & FRAME SAMPLING FUNCTIONS
# ==============================================================================


def sample_indices(n_frames: int, k: int = 16, window: int = 300) -> List[int]:
    """Sample k frame indices distributed uniformly across the first `window` frames.

    Uses bin centering to prevent bias toward the initial or ending frames.
    If the video has fewer frames than the window, sampling spans all available frames.
    """
    k = int(k)
    if k <= 0:
        return []

    total = int(n_frames) if n_frames and n_frames > 0 else int(window)
    w = max(1, min(total, int(window)))

    if k >= w:
        return list(range(w))

    # Center of each of k equal temporal bins
    idx = np.floor((np.arange(k) + 0.5) * w / k).astype(int)
    return sorted(set(int(i) for i in np.clip(idx, 0, w - 1)))


def calculate_iou(boxes: np.ndarray, ref_box: np.ndarray | Tuple[float, float, float, float]) -> np.ndarray:
    """Compute Intersection-over-Union (IoU) of an array of (x, y, w, h) boxes

    against a single reference box.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    ref = np.asarray(ref_box, dtype=np.float64).ravel()

    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float64)

    x1 = np.maximum(boxes[:, 0], ref[0])
    y1 = np.maximum(boxes[:, 1], ref[1])
    x2 = np.minimum(boxes[:, 0] + boxes[:, 2], ref[0] + ref[2])
    y2 = np.minimum(boxes[:, 1] + boxes[:, 3], ref[1] + ref[3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (boxes[:, 2] * boxes[:, 3]) + (ref[2] * ref[3]) - inter

    return inter / np.maximum(union, 1e-9)


def choose_faces(
    detections_per_frame: List[np.ndarray],
    iou_threshold: float = 0.3,
) -> List[Tuple[Optional[np.ndarray], str]]:
    """Associate detected faces across frames to track the same facial identity.

    Rules:
    - In the first frame with detections: selects the largest face.
    - In subsequent frames: selects the detection having the highest IoU with the previously chosen face
      (requiring IoU > iou_threshold). If no detection matches, selects the largest face in that frame.
    - For frames where no face was detected: borrows the bounding box of the nearest valid frame ('neighbour').
    - If no frame has any face detection: returns None with status 'none'.
    """
    chosen: List[Optional[Tuple[np.ndarray, str]]] = []
    prev_box: Optional[np.ndarray] = None

    for b in detections_per_frame:
        if len(b) == 0:
            chosen.append(None)
            continue

        k = None
        if prev_box is not None:
            overlaps = calculate_iou(b[:, :4], prev_box)
            if len(overlaps) > 0 and overlaps.max() > iou_threshold:
                k = int(np.argmax(overlaps))

        if k is None:
            # Fallback to the largest area face box (w * h)
            k = int(np.argmax(b[:, 2] * b[:, 3]))

        selected_box = np.asarray(b[k], dtype=np.float64)
        prev_box = selected_box[:4]
        chosen.append((selected_box, "detected"))

    # Second pass: fill in gaps using nearest neighbor borrowing
    valid_indices = [i for i, c in enumerate(chosen) if c is not None]
    final_output: List[Tuple[Optional[np.ndarray], str]] = []

    for i, c in enumerate(chosen):
        if c is not None:
            final_output.append(c)
        elif valid_indices:
            nearest_idx = min(valid_indices, key=lambda idx: abs(idx - i))
            nearest_box = chosen[nearest_idx][0]
            final_output.append((nearest_box, "neighbour"))
        else:
            final_output.append((None, "none"))

    return final_output


def square_crop(
    img: np.ndarray,
    box: np.ndarray | Tuple[float, float, float, float],
    scale: float = 1.3,
    out_size: int = 224,
) -> np.ndarray:
    """Extract a square face crop centered on the face box with boundary padding.

    - Expands the bounding box by `scale` (default 1.3x).
    - Zero-pads using `cv2.copyMakeBorder` if the box extends beyond frame borders.
    - Resizes cleanly using INTER_AREA (for downsampling) or INTER_CUBIC (for upsampling).
    - Returns an RGB uint8 image of dimensions (out_size, out_size, 3).
    """
    x, y, w, h = [float(v) for v in box[:4]]

    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid bounding box dimensions: w={w}, h={h}")

    cx, cy = x + (w / 2.0), y + (h / 2.0)
    side = max(2, int(round(max(w, h) * float(scale))))
    x0, y0 = int(round(cx - (side / 2.0))), int(round(cy - (side / 2.0)))

    img_h, img_w = img.shape[:2]

    # Crop overlapping region
    sub = img[max(0, y0) : min(img_h, y0 + side), max(0, x0) : min(img_w, x0 + side)]

    # Compute required padding if crop exceeds image bounds
    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, (y0 + side) - img_h)
    pad_right = max(0, (x0 + side) - img_w)

    if sub.size == 0:
        sub = np.zeros((side, side, 3), dtype=np.uint8)
    elif pad_top or pad_left or pad_bottom or pad_right:
        sub = cv2.copyMakeBorder(sub, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    if sub.shape[0] != side or sub.shape[1] != side:
        sub = cv2.resize(sub, (side, side), interpolation=cv2.INTER_LINEAR)

    interp = cv2.INTER_AREA if side > out_size else cv2.INTER_CUBIC
    crop = cv2.resize(sub, (int(out_size), int(out_size)), interpolation=interp)

    # Convert BGR (OpenCV) to RGB (PyTorch standard)
    if crop.ndim == 3 and crop.shape[2] == 3:
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return crop


# ==============================================================================
# FACE DETECTOR ENGINE (YUNET & FALLBACK)
# ==============================================================================


class FaceDetector:
    """Face detector wrapping OpenCV's native ONNX YuNet detector (cv2.FaceDetectorYN)

    with fallback support for OpenCV Haar cascades.
    """

    def __init__(
        self,
        kind: str = "yunet",
        model_path: Optional[str | Path] = None,
        score_threshold: float = 0.6,
        max_detect_side: int = 640,
    ) -> None:
        self.kind = kind.lower()
        self.score_threshold = float(score_threshold)
        self.max_detect_side = int(max_detect_side)
        self.model_path = str(model_path) if model_path else None

        if self.kind == "yunet":
            if not self.model_path or not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"YuNet ONNX model file not found at: {self.model_path}. "
                    "Ensure weights are configured or download face_detection_yunet_2023mar.onnx."
                )
            if not hasattr(cv2, "FaceDetectorYN"):
                raise RuntimeError("Installed OpenCV build lacks cv2.FaceDetectorYN support.")
            self.detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),
                self.score_threshold,
                0.3,
                50,
            )
        elif self.kind == "haar":
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"OpenCV Haar cascade file not found: {cascade_path}")
            self.detector = cv2.CascadeClassifier(cascade_path)
            if self.detector.empty():
                raise RuntimeError("Failed to load OpenCV Haar cascade classifier.")
        else:
            raise ValueError(f"Unsupported detector kind '{kind}'. Expected 'yunet' or 'haar'.")

    def detect(self, img: np.ndarray) -> np.ndarray:
        """Detect faces in an image.

        Returns:
            (N, 5) float32 array: [x, y, w, h, score] for each detected face.
        """
        img_h, img_w = img.shape[:2]
        scale = min(1.0, self.max_detect_side / float(max(img_h, img_w)))

        if scale < 1.0:
            small_w = max(1, int(round(img_w * scale)))
            small_h = max(1, int(round(img_h * scale)))
            small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        else:
            small_img = img

        if self.kind == "yunet":
            self.detector.setInputSize((small_img.shape[1], small_img.shape[0]))
            _, faces = self.detector.detect(small_img)
            if faces is None or len(faces) == 0:
                return np.zeros((0, 5), dtype=np.float32)
            # YuNet returns [x, y, w, h, ... landmarks ..., score (col 14)]
            boxes = np.asarray(faces, dtype=np.float32)[:, [0, 1, 2, 3, 14]].copy()
        else:
            gray = cv2.cvtColor(small_img, cv2.COLOR_BGR2GRAY)
            rects = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
            if len(rects) == 0:
                return np.zeros((0, 5), dtype=np.float32)
            scores = np.ones((len(rects), 1), dtype=np.float32)
            boxes = np.concatenate([np.asarray(rects, dtype=np.float32), scores], axis=1)

        # Rescale boxes back to original image dimensions
        boxes[:, :4] /= scale
        valid_mask = (boxes[:, 2] > 4) & (boxes[:, 3] > 4)
        return boxes[valid_mask]


# ==============================================================================
# HIGH-LEVEL VIDEO EXTRACTION PIPELINE
# ==============================================================================


class FaceTrackerPipeline:
    """Orchestrates video frame decoding, face detection, IoU tracking, and crop extraction."""

    def __init__(
        self,
        yunet_model_path: Optional[str | Path] = None,
        detector_type: str = "yunet",
        score_threshold: float = 0.6,
        max_frames: int = 16,
        sample_window: int = 300,
        crop_scale: float = 1.3,
        crop_size: int = 224,
    ) -> None:
        self.yunet_model_path = yunet_model_path
        self.detector_type = detector_type
        self.score_threshold = score_threshold
        self.max_frames = max_frames
        self.sample_window = sample_window
        self.crop_scale = crop_scale
        self.crop_size = crop_size

        self.detector: Optional[FaceDetector] = None
        if self.yunet_model_path and os.path.exists(str(self.yunet_model_path)):
            self.detector = FaceDetector(
                kind=self.detector_type,
                model_path=self.yunet_model_path,
                score_threshold=self.score_threshold,
            )

    def extract_from_video(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
        custom_detector: Optional[FaceDetector] = None,
    ) -> VideoExtractionResult:
        """Extract facial crops from uniformly sampled frames of a video with identity tracking.

        Args:
            video_path: Path to target video file.
            max_frames: Optional override for the number of frames to sample (defaults to self.max_frames).
            custom_detector: Optional custom or mock face detector.
        """
        path = Path(video_path)
        if not path.is_file():
            return VideoExtractionResult(
                video_path=str(path),
                total_frames=0,
                fps=0.0,
                duration_sec=0.0,
                sampled_indices=[],
                extracted_faces=[],
                valid_face_count=0,
                status="unreadable_video",
                error_message="Video file not found.",
            )

        detector = custom_detector or self.detector
        if detector is None:
            return VideoExtractionResult(
                video_path=str(path),
                total_frames=0,
                fps=0.0,
                duration_sec=0.0,
                sampled_indices=[],
                extracted_faces=[],
                valid_face_count=0,
                status="unreadable_video",
                error_message="No face detector initialized. Provide a valid model_path.",
            )

        cap = cv2.VideoCapture(str(path.resolve()))
        if not cap.isOpened():
            return VideoExtractionResult(
                video_path=str(path),
                total_frames=0,
                fps=0.0,
                duration_sec=0.0,
                sampled_indices=[],
                extracted_faces=[],
                valid_face_count=0,
                status="unreadable_video",
                error_message="OpenCV could not open video stream.",
            )

        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
            duration_sec = float(total_frames) / fps if fps > 0 else 0.0

            k_samples = int(max_frames) if max_frames is not None else self.max_frames
            indices = sample_indices(total_frames, k=k_samples, window=self.sample_window)
            if not indices:
                return VideoExtractionResult(
                    video_path=str(path),
                    total_frames=total_frames,
                    fps=fps,
                    duration_sec=duration_sec,
                    sampled_indices=[],
                    extracted_faces=[],
                    valid_face_count=0,
                    status="unreadable_video",
                    error_message="Zero frames could be scheduled for sampling.",
                )

            # Read target frames sequentially using grab() + retrieve()
            frames_decoded: List[Tuple[int, np.ndarray]] = []
            target_set = set(indices)
            curr_idx = 0
            max_target = max(target_set)

            while curr_idx <= max_target:
                if not cap.grab():
                    break
                if curr_idx in target_set:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None and frame.size > 0:
                        frames_decoded.append((curr_idx, frame))
                curr_idx += 1
        finally:
            cap.release()

        if not frames_decoded:
            return VideoExtractionResult(
                video_path=str(path),
                total_frames=total_frames,
                fps=fps,
                duration_sec=duration_sec,
                sampled_indices=indices,
                extracted_faces=[],
                valid_face_count=0,
                status="unreadable_video",
                error_message="Could not decode any sampled frames from video stream.",
            )

        # Detect candidate faces in each decoded frame
        detections_per_frame = [detector.detect(fr) for _, fr in frames_decoded]

        # Associate and track the most salient face across frames via IoU
        associated_faces = choose_faces(detections_per_frame, iou_threshold=0.3)

        extracted_faces: List[ExtractedFace] = []
        valid_face_count = 0

        for (frame_idx, frame), (chosen_item, track_status) in zip(frames_decoded, associated_faces):
            timestamp = round(float(frame_idx) / fps, 3) if fps > 0 else 0.0
            frame_shape = (frame.shape[0], frame.shape[1])

            if chosen_item is not None:
                box_tuple = (
                    float(chosen_item[0]),
                    float(chosen_item[1]),
                    float(chosen_item[2]),
                    float(chosen_item[3]),
                )
                conf = float(chosen_item[4]) if len(chosen_item) > 4 else 1.0

                try:
                    crop = square_crop(frame, box_tuple, scale=self.crop_scale, out_size=self.crop_size)
                    extracted_faces.append(
                        ExtractedFace(
                            video_path=str(path.resolve()),
                            frame_index=frame_idx,
                            timestamp_sec=timestamp,
                            box=box_tuple,
                            confidence=conf,
                            tracking_status=track_status,
                            crop_rgb=crop,
                            original_frame_shape=frame_shape,
                        )
                    )
                    valid_face_count += 1
                except Exception:
                    extracted_faces.append(
                        ExtractedFace(
                            video_path=str(path.resolve()),
                            frame_index=frame_idx,
                            timestamp_sec=timestamp,
                            box=box_tuple,
                            confidence=conf,
                            tracking_status="failed_crop",
                            crop_rgb=None,
                            original_frame_shape=frame_shape,
                        )
                    )
            else:
                extracted_faces.append(
                    ExtractedFace(
                        video_path=str(path.resolve()),
                        frame_index=frame_idx,
                        timestamp_sec=timestamp,
                        box=None,
                        confidence=0.0,
                        tracking_status="none",
                        crop_rgb=None,
                        original_frame_shape=frame_shape,
                    )
                )

        if valid_face_count == len(frames_decoded):
            status = "success"
        elif valid_face_count > 0:
            status = "partial"
        else:
            status = "no_faces_detected"

        return VideoExtractionResult(
            video_path=str(path.resolve()),
            total_frames=total_frames,
            fps=fps,
            duration_sec=duration_sec,
            sampled_indices=[idx for idx, _ in frames_decoded],
            extracted_faces=extracted_faces,
            valid_face_count=valid_face_count,
            status=status,
            error_message=None if valid_face_count > 0 else "No faces could be detected in sampled frames.",
        )
