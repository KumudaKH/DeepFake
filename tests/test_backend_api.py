"""Unit tests for the FastAPI REST API backend.

Tests the health check endpoint, image deepfake detection, Grad-CAM visualization
payloads, and HTML forensic report export using FastAPI TestClient.
"""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.backend.app.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """Verify health endpoint returns 200 with complete service and model status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_architecture"] == "efficientnet_b4"
    assert data["input_size"] == 380
    assert "checkpoint_type" in data
    assert data["model_readiness"] == ("trained" if data["is_model_trained"] else "demo")
    assert "device" in data
    assert "disclaimer" in data


def test_api_detect_image_valid_rgb():
    """Verify image detection processes a valid image and returns prediction and Grad-CAM data."""
    # Generate a synthetic RGB image
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    image_bytes = buf.tobytes()

    response = client.post(
        "/api/detect/image",
        files={"file": ("test_face.png", image_bytes, "image/png")},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_face.png"
    assert data["file_size_bytes"] == len(image_bytes)
    assert len(data["sha256"]) == 64  # Valid SHA-256 hex string
    assert data["evidence_id"].startswith("IMG-")

    # Check prediction block
    pred = data["prediction"]
    assert pred["raw_logit"] is None
    assert pred["probability"] is None
    # In demo mode with uninitialized mock model, predicted_label is 'inconclusive'
    assert pred["predicted_label"] == "inconclusive"
    assert "forensic_disclaimer" in pred

    # Check explainability block
    cam = data["explainability"]
    assert cam["heatmap_data_url"] is None
    assert cam["overlay_data_url"] is None
    assert data["examination_timestamp"]
    assert data["model_status"] == "demo"


def test_api_detect_image_empty_file_raises_400():
    """Verify uploading an empty file returns a clean HTTP 400 error."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_html_report_rejects_unverified_verdicts_in_demo_mode():
    """Verify caller-supplied verdicts and scores cannot override missing trained weights."""
    response = client.get(
        "/api/report/html/IMG-TEST12345",
        params={
            "filename": "suspect.jpg",
            "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "verdict": "fake",
            "probability": 0.884,
            "media_type": "image",
        },
    )
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    html_text = response.text
    assert "IMG-TEST12345" in html_text
    assert "suspect.jpg" in html_text
    assert "INCONCLUSIVE — DEMO MODE" in html_text
    assert "0.8840" not in html_text
    assert "Digital Forensics Examination Sheet" in html_text


def test_api_detect_video_missing_yunet_raises_503(monkeypatch):
    """Verify video analysis returns HTTP 503 when YuNet weights are not installed."""
    from pathlib import Path
    from src.backend.app import main

    # Explicitly test missing weights behavior regardless of local filesystem state
    monkeypatch.setattr(main, "YUNET_PATH", Path("models/nonexistent/face_detection_yunet_2023mar.onnx"))

    response = client.post(
        "/api/detect/video",
        files={"file": ("dummy.mp4", b"dummy content", "video/mp4")},
    )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "yunet" in detail.lower()
    assert "face_detection_yunet_2023mar.onnx" in detail


def test_api_json_report_rejects_unverified_verdicts_in_demo_mode():
    """Verify caller-supplied verdicts and scores cannot override missing trained weights."""
    response = client.get(
        "/api/report/json/IMG-TEST999",
        params={
            "filename": "evidence.png",
            "sha256": "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
            "verdict": "real",
            "probability": 0.05,
            "media_type": "image",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence_id"] == "IMG-TEST999"
    assert data["filename"] == "evidence.png"
    assert data["sha256"] == "11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff"
    assert data["classification"]["verdict"] == "inconclusive"
    assert data["classification"]["probability"] is None
    assert data["classification"]["raw_logit"] is None
    assert "forensic_limitations" in data


def test_api_detect_video_pipeline_success(monkeypatch):
    """Verify video analysis endpoint accepts max_frames, calls tracker, and returns complete video verdict."""
    from pathlib import Path
    from src.backend.app import main
    from src.video_detection.face_tracker import ExtractedFace, VideoExtractionResult

    # Ensure YUNET_PATH points to an existing file for this test
    monkeypatch.setattr(main, "YUNET_PATH", Path(__file__))

    dummy_face = ExtractedFace(
        video_path="mock.mp4",
        frame_index=5,
        timestamp_sec=0.25,
        box=(10.0, 10.0, 30.0, 30.0),
        confidence=0.95,
        tracking_status="detected",
        crop_rgb=np.full((64, 64, 3), 120, dtype=np.uint8),
        original_frame_shape=(200, 300),
    )
    mock_result = VideoExtractionResult(
        video_path="mock.mp4",
        total_frames=50,
        fps=20.0,
        duration_sec=2.5,
        sampled_indices=[5],
        extracted_faces=[dummy_face],
        valid_face_count=1,
        status="success",
    )

    class MockTracker:
        detector = True

        def extract_from_video(self, video_path, max_frames=None):
            assert max_frames == 8
            return mock_result

    monkeypatch.setattr(main, "get_tracker", lambda: MockTracker())

    response = client.post(
        "/api/detect/video?max_frames=8",
        files={"file": ("test.mp4", b"dummy video bytes", "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["evidence_id"].startswith("VID-")
    assert data["video_metadata"]["num_frames_scanned"] == 1
    assert data["video_metadata"]["num_faces_extracted"] == 1
    assert data["video_metadata"]["num_unscorable_frames"] == 0
    assert data["video_metadata"]["video_dimensions"] == [300, 200]
    # In demo mode with uninitialized mock model, video predicted_label is 'inconclusive'
    assert data["video_prediction"]["predicted_label"] == "inconclusive"
    assert len(data["frames"]) == 1
    assert data["frames"][0]["frame_index"] == 5
    assert data["frames"][0]["logit"] is None
    assert data["frames"][0]["probability"] is None
    assert data["video_prediction"]["mean_logit"] is None
    assert data["video_prediction"]["probability"] is None
    assert data["is_mock_prediction"] is True


def test_api_html_report_export_demo_mode():
    """Verify HTML report shows INCONCLUSIVE — DEMO MODE when requested in demo mode."""
    response = client.get(
        "/api/report/html/IMG-DEMO123",
        params={
            "filename": "demo.jpg",
            "sha256": "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            "verdict": "inconclusive",
            "media_type": "image",
            "examination_timestamp_utc": "2025-01-02T03:04:05+00:00",
        },
    )
    assert response.status_code == 200
    assert "INCONCLUSIVE — DEMO MODE" in response.text
    assert "DEMO MODE NOTICE" in response.text
    assert "2025-01-02T03:04:05+00:00" in response.text


def test_api_json_report_export_demo_mode():
    """Verify JSON report shows inconclusive demo mode when requested in demo mode."""
    response = client.get(
        "/api/report/json/IMG-DEMO456",
        params={
            "filename": "demo.png",
            "sha256": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
            "verdict": "inconclusive",
            "media_type": "image",
            "probability": 0.5,
            "raw_logit": 2.0,
            "examination_timestamp_utc": "2025-01-02T03:04:05+00:00",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["classification"]["verdict"] == "inconclusive"
    assert data["classification"]["verdict_display"] == "INCONCLUSIVE — DEMO MODE"
    assert data["classification"]["is_mock_prediction"] is True
    assert data["classification"]["probability"] is None
    assert data["classification"]["raw_logit"] is None
    assert data["examination_timestamp_utc"] == "2025-01-02T03:04:05+00:00"
    assert "explainability" in data
    assert "narrative" in data["explainability"]


def test_api_detect_image_corrupt_file_raises_400():
    """Verify uploading a corrupt image file returns HTTP 400."""
    response = client.post(
        "/api/detect/image",
        files={"file": ("corrupt.png", b"not_a_valid_image_bytes", "image/png")},
    )
    assert response.status_code == 400
    assert "could not decode image" in response.json()["detail"].lower()


def test_api_detect_video_empty_file_raises_400(monkeypatch):
    """Verify uploading an empty video file returns HTTP 400."""
    from pathlib import Path
    from src.backend.app import main

    monkeypatch.setattr(main, "YUNET_PATH", Path(__file__))
    response = client.post(
        "/api/detect/video",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_api_detect_video_corrupt_file_raises_400(monkeypatch):
    """Verify uploading a corrupt/unreadable video returns HTTP 400."""
    from pathlib import Path
    from src.backend.app import main
    from src.video_detection.face_tracker import VideoExtractionResult

    monkeypatch.setattr(main, "YUNET_PATH", Path(__file__))

    class MockTracker:
        detector = True

        def extract_from_video(self, video_path, max_frames=None):
            return VideoExtractionResult(
                video_path=video_path,
                total_frames=0,
                fps=0.0,
                duration_sec=0.0,
                sampled_indices=[],
                extracted_faces=[],
                valid_face_count=0,
                status="unreadable_video",
                error_message="Could not open video file.",
            )

    monkeypatch.setattr(main, "get_tracker", lambda: MockTracker())

    response = client.post(
        "/api/detect/video",
        files={"file": ("corrupt.mp4", b"corrupt_video_data_123456789", "video/mp4")},
    )
    assert response.status_code == 400
    assert "could not decode video file" in response.json()["detail"].lower()


def test_api_detect_video_no_faces_returns_unscorable(monkeypatch):
    """Verify video analysis when no faces are detected returns unscorable verdict without crashing."""
    from pathlib import Path
    from src.backend.app import main
    from src.video_detection.face_tracker import ExtractedFace, VideoExtractionResult

    monkeypatch.setattr(main, "YUNET_PATH", Path(__file__))

    dummy_no_face = ExtractedFace(
        video_path="mock.mp4",
        frame_index=0,
        timestamp_sec=0.0,
        box=None,
        confidence=0.0,
        tracking_status="no_face",
        crop_rgb=None,
        original_frame_shape=(200, 300),
    )
    mock_result = VideoExtractionResult(
        video_path="mock.mp4",
        total_frames=10,
        fps=10.0,
        duration_sec=1.0,
        sampled_indices=[0],
        extracted_faces=[dummy_no_face],
        valid_face_count=0,
        status="no_faces_detected",
    )

    class MockTracker:
        detector = True

        def extract_from_video(self, video_path, max_frames=None):
            return mock_result

    monkeypatch.setattr(main, "get_tracker", lambda: MockTracker())

    response = client.post(
        "/api/detect/video",
        files={"file": ("no_face.mp4", b"dummy video bytes", "video/mp4")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["video_prediction"]["predicted_label"] == "unscorable"
    assert data["video_metadata"]["num_faces_extracted"] == 0
    assert data["keyframe_explainability"]["crop_data_url"] is None
