"""Unit tests for the Deepfake Model Architecture Wrapper and Inference Engine.

Tests input validation, evaluation mode determinism, missing/incompatible checkpoint handling,
configurable decision threshold behavior, and video frame metadata preservation.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.image_detection.detector import (
    DeepfakeDetector,
    FramePrediction,
    ImagePrediction,
)
from src.video_detection.face_tracker import ExtractedFace


@pytest.fixture
def mock_detector():
    """Create a lightweight uninitialized detector instance for interface testing without network downloads."""
    # Using allow_uninitialized=True for unit testing interfaces
    return DeepfakeDetector(
        model_name="efficientnet_b4.ra2_in1k",
        checkpoint_path=None,
        device="cpu",
        default_threshold=0.5,
        allow_uninitialized=True,
    )


@pytest.fixture
def temp_workspace():
    """Temporary workspace for saving mock checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ==============================================================================
# 1. INPUT VALIDATION TESTS
# ==============================================================================


def test_preprocess_image_valid_rgb(mock_detector):
    """Test preprocessing converts RGB uint8 image into normalized (1, 3, 224, 224) float32 tensor."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = mock_detector.preprocess_image(img)

    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32
    assert tensor.device == torch.device("cpu")


def test_preprocess_image_invalid_type_raises(mock_detector):
    """Verify passing a non-numpy object raises TypeError."""
    with pytest.raises(TypeError):
        mock_detector.preprocess_image("invalid_string_not_an_array")


def test_preprocess_380px_notebook_resolution():
    """Verify notebook-compatible preprocessing produces the required 380px tensor."""
    detector = DeepfakeDetector(
        model_name="efficientnet_b4",
        allow_uninitialized=True,
        input_size=380,
    )
    tensor = detector.preprocess_image(np.zeros((120, 160, 3), dtype=np.uint8))

    assert tensor.shape == (1, 3, 380, 380)
    assert tensor.dtype == torch.float32


def test_preprocess_image_invalid_channels_raises(mock_detector):
    """Verify passing a 1-channel or 4-channel image raises ValueError."""
    # 2D grayscale
    with pytest.raises(ValueError) as exc:
        mock_detector.preprocess_image(np.zeros((64, 64), dtype=np.uint8))
    assert "Expected 3-channel RGB" in str(exc.value)

    # 4-channel RGBA
    with pytest.raises(ValueError):
        mock_detector.preprocess_image(np.zeros((64, 64, 4), dtype=np.uint8))


def test_preprocess_batch_empty_raises(mock_detector):
    """Verify empty image batch raises ValueError."""
    with pytest.raises(ValueError):
        mock_detector.preprocess_batch([])


# ==============================================================================
# 2. INFERENCE SHAPE & DETERMINISM TESTS
# ==============================================================================


def test_predict_image_output_structure(mock_detector):
    """Verify predict_image returns a complete ImagePrediction with valid probability range [0, 1]."""
    dummy_img = np.full((128, 128, 3), 120, dtype=np.uint8)
    pred = mock_detector.predict_image(dummy_img)

    assert isinstance(pred, ImagePrediction)
    assert isinstance(pred.raw_logit, float)
    assert 0.0 <= pred.probability <= 1.0
    assert pred.predicted_label in ("real", "fake")
    assert pred.decision_threshold == 0.5
    assert pred.is_provisional_threshold is True
    assert pred.model_architecture == "efficientnet_b4.ra2_in1k"
    assert "probabilistic neural inferences" in pred.forensic_disclaimer


def test_predict_image_determinism_in_eval_mode(mock_detector):
    """Verify model inference is fully deterministic in evaluation mode (no stochastic dropout active)."""
    dummy_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    pred1 = mock_detector.predict_image(dummy_img)
    pred2 = mock_detector.predict_image(dummy_img)

    assert pred1.raw_logit == pred2.raw_logit
    assert pred1.probability == pred2.probability
    assert pred1.predicted_label == pred2.predicted_label


def test_predict_batch_throughput(mock_detector):
    """Verify batched inference processes multiple images and preserves order."""
    imgs = [
        np.full((64, 64, 3), 50, dtype=np.uint8),
        np.full((64, 64, 3), 100, dtype=np.uint8),
        np.full((64, 64, 3), 150, dtype=np.uint8),
    ]

    preds = mock_detector.predict_batch(imgs)

    assert len(preds) == 3
    assert all(isinstance(p, ImagePrediction) for p in preds)


# ==============================================================================
# 3. CHECKPOINT LOADING & COMPATIBILITY TESTS
# ==============================================================================


def test_missing_checkpoint_raises_filenotfound():
    """Verify providing a nonexistent checkpoint file raises clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        DeepfakeDetector(checkpoint_path="models/image/nonexistent_model_weights.pt")
    assert "checkpoint not found" in str(exc_info.value).lower()


def test_refuse_silent_uninitialized_execution():
    """Verify detector refuses to silently run inference without weights unless explicitly permitted."""
    with pytest.raises(FileNotFoundError) as exc_info:
        DeepfakeDetector(checkpoint_path=None, allow_uninitialized=False)
    assert "trained deepfake model checkpoint path must be provided" in str(exc_info.value)


def test_incompatible_checkpoint_raises_valueerror(temp_workspace):
    """Verify incomplete state dicts fail exact architecture loading."""
    ckpt_path = temp_workspace / "incompatible_checkpoint.pt"
    # Save a dummy dict that has incompatible keys
    torch.save({"unrelated_layer.weight": torch.randn(10, 10)}, ckpt_path)

    with pytest.raises(ValueError, match="does not exactly match architecture"):
        DeepfakeDetector(checkpoint_path=ckpt_path, checkpoint_type="uninitialized_mock")


def test_raw_state_dict_cannot_mark_model_trained(temp_workspace):
    """Verify raw weights without training provenance do not mark the detector trained."""
    ckpt_path = temp_workspace / "raw_state_dict.pt"
    base_model = DeepfakeDetector(allow_uninitialized=True)
    torch.save(base_model.model.state_dict(), ckpt_path)

    with pytest.raises(ValueError, match="no verified deepfake-training metadata"):
        DeepfakeDetector(checkpoint_path=ckpt_path, checkpoint_type="deepfake_trained")


def test_trained_checkpoint_requires_notebook_state_dict_metadata(temp_workspace):
    """Verify a metadata-wrapped checkpoint must identify its exact export format."""
    ckpt_path = temp_workspace / "wrong_checkpoint_metadata.pt"
    base_model = DeepfakeDetector(
        model_name="efficientnet_b4",
        allow_uninitialized=True,
        input_size=380,
    )
    metadata = {
        "format_version": 1,
        "checkpoint_role": "best_inference",
        "state_dict_format": "unknown",
        "task": "deepfake_binary_classification",
        "model_architecture": "efficientnet_b4",
        "model_config": {"num_classes": 1},
        "input_size": 380,
        "input_transform": "torchvision.transforms.Resize((380,380)), RGB ToTensor, ImageNet normalization",
        "labels": {"real": 0, "fake": 1},
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "best_epoch": 1,
        "validation_metric": {"name": "roc_auc", "value": 0.5},
    }
    torch.save({"state_dict": base_model.model.state_dict(), "metadata": metadata}, ckpt_path)

    with pytest.raises(ValueError, match="state_dict_format"):
        DeepfakeDetector(
            model_name="efficientnet_b4",
            checkpoint_path=ckpt_path,
            checkpoint_type="deepfake_trained",
            input_size=380,
        )


def test_checkpoint_with_wrong_classifier_shape_raises(temp_workspace):
    """Verify a two-logit classifier is rejected by the single-logit inference model."""
    ckpt_path = temp_workspace / "wrong_classifier_shape.pt"
    base_model = DeepfakeDetector(allow_uninitialized=True)
    state_dict = base_model.model.state_dict()
    state_dict["classifier.weight"] = torch.randn(2, 1792)
    torch.save(state_dict, ckpt_path)

    with pytest.raises(ValueError, match="does not exactly match architecture"):
        DeepfakeDetector(checkpoint_path=ckpt_path, checkpoint_type="uninitialized_mock")


def test_valid_mock_checkpoint_loads_without_trained_readiness(temp_workspace):
    """Verify an architecture-compatible test checkpoint remains explicitly untrained."""
    ckpt_path = temp_workspace / "compatible_mock_checkpoint.pt"

    base_model = DeepfakeDetector(allow_uninitialized=True)
    torch.save(base_model.model.state_dict(), ckpt_path)

    detector = DeepfakeDetector(
        checkpoint_path=ckpt_path,
        checkpoint_type="uninitialized_mock",
    )

    assert detector.checkpoint_path == str(ckpt_path.resolve())
    assert detector.checkpoint_type == "uninitialized_mock"


def test_fp16_checkpoint_weights_are_cast_for_cpu_inference(temp_workspace):
    """Verify FP16 state-dict tensors become FP32 when loaded onto CPU."""
    ckpt_path = temp_workspace / "fp16_mock_checkpoint.pt"
    base_model = DeepfakeDetector(allow_uninitialized=True)
    fp16_state_dict = {
        key: value.half() if value.is_floating_point() else value
        for key, value in base_model.model.state_dict().items()
    }
    torch.save(fp16_state_dict, ckpt_path)

    detector = DeepfakeDetector(
        checkpoint_path=ckpt_path,
        checkpoint_type="uninitialized_mock",
        device="cpu",
    )

    assert all(
        not tensor.is_floating_point() or tensor.dtype == torch.float32
        for tensor in detector.model.state_dict().values()
    )


# ==============================================================================
# 4. CONFIGURABLE THRESHOLD BEHAVIOR TESTS
# ==============================================================================


def test_configurable_decision_threshold(mock_detector):
    """Verify changing decision threshold alters predicted label boundary properly."""
    dummy_img = np.full((100, 100, 3), 120, dtype=np.uint8)

    # Using an extreme threshold of 0.0001 (any non-zero probability becomes "fake")
    pred_low_thresh = mock_detector.predict_image(dummy_img, threshold=0.0001)
    # Using an extreme threshold of 0.9999 (almost impossible to trigger "fake")
    pred_high_thresh = mock_detector.predict_image(dummy_img, threshold=0.9999)

    assert pred_low_thresh.decision_threshold == 0.0001
    assert pred_high_thresh.decision_threshold == 0.9999
    assert pred_low_thresh.predicted_label == "fake"
    assert pred_high_thresh.predicted_label == "real"


# ==============================================================================
# 5. VIDEO FRAME SCORING & METADATA PRESERVATION TESTS
# ==============================================================================


def test_predict_video_faces_metadata_preservation(mock_detector):
    """Verify video face predictions preserve exact frame indices, timestamps, and bounding boxes."""
    # Create two extracted faces (one valid face, one frame with no face detected)
    valid_face = ExtractedFace(
        video_path="C:/videos/sample.mp4",
        frame_index=15,
        timestamp_sec=0.60,
        box=(30.0, 30.0, 50.0, 50.0),
        confidence=0.95,
        tracking_status="detected",
        crop_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
        original_frame_shape=(480, 640),
    )
    empty_face = ExtractedFace(
        video_path="C:/videos/sample.mp4",
        frame_index=30,
        timestamp_sec=1.20,
        box=None,
        confidence=0.0,
        tracking_status="none",
        crop_rgb=None,
        original_frame_shape=(480, 640),
    )

    frame_predictions = mock_detector.predict_video_faces([valid_face, empty_face])

    assert len(frame_predictions) == 2

    # Frame 0: Valid face crop
    fp0 = frame_predictions[0]
    assert fp0.video_path == "C:/videos/sample.mp4"
    assert fp0.frame_index == 15
    assert fp0.timestamp_sec == 0.60
    assert fp0.box == (30.0, 30.0, 50.0, 50.0)
    assert fp0.tracking_status == "detected"
    assert fp0.has_face is True
    assert fp0.prediction is not None
    assert isinstance(fp0.prediction.raw_logit, float)

    # Frame 1: Empty face crop (gracefully handled)
    fp1 = frame_predictions[1]
    assert fp1.video_path == "C:/videos/sample.mp4"
    assert fp1.frame_index == 30
    assert fp1.timestamp_sec == 1.20
    assert fp1.has_face is False
    assert fp1.prediction is None
