"""Unit tests for the Core Face Extraction and Tracking Module.

Tests uniform temporal sampling, IoU association, square face cropping,
missing weights handling, synthetic video decoding, and timestamp traceability.
"""

import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest

from src.video_detection.face_tracker import (
    ExtractedFace,
    FaceDetector,
    FaceTrackerPipeline,
    calculate_iou,
    choose_faces,
    sample_indices,
    square_crop,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for test fixtures."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ==============================================================================
# 1. UNIFORM FRAME SAMPLING TESTS
# ==============================================================================


def test_sample_indices_benchmark_agreement():
    """Verify that sample_indices produces bin-centered indices matching the official benchmark."""
    # 4 frames across 300 window in a 1000 frame video: [37, 112, 187, 262]
    sampled = sample_indices(n_frames=1000, k=4, window=300)
    assert sampled == [37, 112, 187, 262]


def test_sample_indices_edge_cases():
    """Test boundary conditions for frame sampling."""
    # Zero or negative k requested
    assert sample_indices(100, k=0) == []
    assert sample_indices(100, k=-5) == []

    # k exceeds video frame count -> should return all frames
    assert sample_indices(5, k=10) == [0, 1, 2, 3, 4]

    # Video shorter than window
    sampled_short = sample_indices(n_frames=10, k=3, window=300)
    assert len(sampled_short) == 3
    assert max(sampled_short) < 10


# ==============================================================================
# 2. BOUNDING BOX & SQUARE CROP TESTS
# ==============================================================================


def test_square_crop_valid_and_rgb():
    """Verify square crop returns square, centered, RGB uint8 array."""
    # Create a 100x120 test image (BGR: blue background)
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    # Draw a 20x20 white face patch at (50, 40)
    img[40:60, 50:70] = [255, 255, 255]

    box = (50.0, 40.0, 20.0, 20.0)
    crop = square_crop(img, box, scale=1.3, out_size=64)

    assert crop.shape == (64, 64, 3)
    assert crop.dtype == np.uint8
    # Center pixel should be white (255, 255, 255)
    assert crop[32, 32].tolist() == [255, 255, 255]


def test_square_crop_boundary_padding():
    """Verify crop safely zero-pads when face box lies near or exceeds image border."""
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    # Box at top-left corner (0, 0)
    box = (0.0, 0.0, 30.0, 30.0)

    # Scaling 1.5x extends beyond top and left borders
    crop = square_crop(img, box, scale=1.5, out_size=64)

    assert crop.shape == (64, 64, 3)
    # Top-left corner must be zero-padded black (0, 0, 0)
    assert crop[0, 0].tolist() == [0, 0, 0]


def test_square_crop_invalid_box_raises():
    """Verify invalid bounding box (width/height <= 0) raises ValueError."""
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        square_crop(img, (10, 10, 0, 20))
    with pytest.raises(ValueError):
        square_crop(img, (10, 10, 20, -5))


# ==============================================================================
# 3. IOU & IDENTITY ASSOCIATION TESTS
# ==============================================================================


def test_calculate_iou_geometry():
    """Verify IoU calculation on identical, overlapping, and disjoint boxes."""
    box_a = np.array([10.0, 10.0, 20.0, 20.0])  # Area = 400
    box_identical = np.array([[10.0, 10.0, 20.0, 20.0]])
    box_disjoint = np.array([[100.0, 100.0, 20.0, 20.0]])
    box_half_overlap = np.array([[20.0, 10.0, 20.0, 20.0]])  # Overlap 10x20 = 200, Union = 600 -> 0.333

    assert pytest.approx(calculate_iou(box_identical, box_a)[0], rel=1e-4) == 1.0
    assert calculate_iou(box_disjoint, box_a)[0] == 0.0
    assert pytest.approx(calculate_iou(box_half_overlap, box_a)[0], rel=1e-3) == 0.3333


def test_choose_faces_tracking_and_neighbour_borrowing():
    """Test identity tracking: selects largest face, tracks via IoU, and borrows nearest neighbour."""
    # Frame 0: no detection
    frame0_dets = np.zeros((0, 5), dtype=np.float32)
    # Frame 1: 2 candidate faces (small face at (0,0), large face at (50,50))
    frame1_dets = np.array(
        [
            [0.0, 0.0, 10.0, 10.0, 0.9],
            [50.0, 50.0, 30.0, 30.0, 0.85],
        ],
        dtype=np.float32,
    )
    # Frame 2: tracked face moving slightly to (52, 52)
    frame2_dets = np.array(
        [
            [52.0, 52.0, 30.0, 30.0, 0.88],
            [2.0, 2.0, 10.0, 10.0, 0.92],
        ],
        dtype=np.float32,
    )

    associated = choose_faces([frame0_dets, frame1_dets, frame2_dets], iou_threshold=0.3)

    # Frame 1 selects largest face (at 50,50 with w=30)
    assert associated[1][1] == "detected"
    assert associated[1][0][0] == 50.0

    # Frame 0 borrows the face from Frame 1 as neighbour
    assert associated[0][1] == "neighbour"
    assert associated[0][0][0] == 50.0

    # Frame 2 associates with the tracked face at (52,52) via high IoU
    assert associated[2][1] == "detected"
    assert associated[2][0][0] == 52.0


# ==============================================================================
# 4. DETECTOR WEIGHT HANDLING TESTS
# ==============================================================================


def test_missing_yunet_weights_raises_filenotfound():
    """Verify initializing YuNet with missing model path raises clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        FaceDetector(kind="yunet", model_path="models/yunet/nonexistent_file.onnx")
    assert "YuNet ONNX model file not found" in str(exc_info.value)


# ==============================================================================
# 5. VIDEO EXTRACTION PIPELINE END-TO-END TESTS
# ==============================================================================


class MockDetector:
    """Mock face detector returning deterministic face detections for synthetic video testing."""

    def __init__(self, face_box: np.ndarray):
        self.face_box = face_box

    def detect(self, img: np.ndarray) -> np.ndarray:
        return self.face_box.copy()


def test_extract_from_video_with_traceable_timestamps(temp_workspace):
    """Test full extraction pipeline on a synthetic 10-frame video, verifying frame index and timestamps."""
    vid_path = temp_workspace / "synthetic_face_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 20.0
    out = cv2.VideoWriter(str(vid_path), fourcc, fps, (128, 128))

    # Generate 10 frames
    for i in range(10):
        frame = np.full((128, 128, 3), 40, dtype=np.uint8)
        # Draw a synthetic face rectangle
        frame[40:80, 40:80] = 220
        out.write(frame)
    out.release()

    # Configure mock detector returning the (40, 40, 40, 40) box
    mock_box = np.array([[40.0, 40.0, 40.0, 40.0, 0.95]], dtype=np.float32)
    custom_det = MockDetector(mock_box)

    pipeline = FaceTrackerPipeline(
        yunet_model_path=None,
        max_frames=5,
        sample_window=10,
        crop_scale=1.2,
        crop_size=64,
    )

    result = pipeline.extract_from_video(vid_path, custom_detector=custom_det)

    assert result.status == "success"
    assert result.total_frames == 10
    assert result.fps == 20.0
    assert len(result.extracted_faces) == 5
    assert result.valid_face_count == 5

    # Check timestamps and indices traceability
    for face in result.extracted_faces:
        assert face.video_path == str(vid_path.resolve())
        assert face.crop_rgb is not None
        assert face.crop_rgb.shape == (64, 64, 3)
        expected_time = round(float(face.frame_index) / 20.0, 3)
        assert face.timestamp_sec == expected_time
        assert face.tracking_status in ("detected", "neighbour")


def test_extract_from_video_unreadable_or_missing(temp_workspace):
    """Test pipeline gracefully handles unreadable or missing video files."""
    pipeline = FaceTrackerPipeline(yunet_model_path=None, max_frames=4)
    result = pipeline.extract_from_video(temp_workspace / "does_not_exist.mp4")

    assert result.status == "unreadable_video"
    assert result.valid_face_count == 0
    assert "not found" in str(result.error_message).lower()


def test_extract_from_video_no_faces_detected(temp_workspace):
    """Test pipeline handles video containing no detectable faces."""
    vid_path = temp_workspace / "blank_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (64, 64))
    for _ in range(5):
        out.write(np.zeros((64, 64, 3), dtype=np.uint8))
    out.release()

    # Detector that returns zero faces
    empty_detector = MockDetector(np.zeros((0, 5), dtype=np.float32))

    pipeline = FaceTrackerPipeline(yunet_model_path=None, max_frames=3)
    result = pipeline.extract_from_video(vid_path, custom_detector=empty_detector)

    assert result.status == "no_faces_detected"
    assert result.valid_face_count == 0
    assert len(result.extracted_faces) == 3
    # All face crops should be None
    assert all(f.crop_rgb is None for f in result.extracted_faces)
    assert all(f.tracking_status == "none" for f in result.extracted_faces)
