"""Unit tests for the Grad-CAM Explainability Module.

Tests spatial interpretability heatmap generation, OpenCV RGB overlay creation,
resolution resizing, safe normalization of constant activations, hook cleanup
on success and error, CPU compatibility, and forensic disclaimer safety.
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.explainability.gradcam import (
    GradCAMComputationError,
    GradCAMExplainer,
    GradCAMResult,
)
from src.image_detection.detector import DeepfakeDetector


@pytest.fixture
def mock_detector():
    """Create a lightweight uninitialized detector instance for explainability tests."""
    return DeepfakeDetector(
        model_name="efficientnet_b4.ra2_in1k",
        checkpoint_path=None,
        device="cpu",
        default_threshold=0.5,
        allow_uninitialized=True,
    )


@pytest.fixture
def explainer(mock_detector):
    """Instantiate GradCAMExplainer attached to mock EfficientNet-B4 detector."""
    return GradCAMExplainer(mock_detector)


# ==============================================================================
# 1. INPUT VALIDATION TESTS
# ==============================================================================


def test_input_validation_non_numpy_raises(explainer):
    """Verify non-numpy input raises TypeError."""
    with pytest.raises(TypeError) as exc:
        explainer.generate_cam([[1, 2], [3, 4]])
    assert "Expected numpy.ndarray image" in str(exc.value)


def test_input_validation_wrong_dimensions_raises(explainer):
    """Verify 2D grayscale or 4-channel image raises ValueError."""
    # 2D grayscale
    with pytest.raises(ValueError) as exc1:
        explainer.generate_cam(np.zeros((100, 100), dtype=np.uint8))
    assert "Expected 3-channel RGB image" in str(exc1.value)

    # 4-channel RGBA
    with pytest.raises(ValueError) as exc2:
        explainer.generate_cam(np.zeros((100, 100, 4), dtype=np.uint8))
    assert "Expected 3-channel RGB image" in str(exc2.value)


def test_input_validation_zero_dimension_raises(explainer):
    """Verify image with zero width or height raises ValueError."""
    with pytest.raises(ValueError) as exc:
        explainer.generate_cam(np.zeros((0, 100, 3), dtype=np.uint8))
    assert "Invalid image dimensions" in str(exc.value)


# ==============================================================================
# 2. OUTPUT DIMENSIONS & SHAPE PRESERVATION TESTS
# ==============================================================================


def test_generate_cam_dimension_preservation_non_square(explainer):
    """Verify heatmap and overlay match the exact arbitrary input dimensions (non-square)."""
    # Create non-square test image (height=140, width=190, channels=3)
    orig_h, orig_w = 140, 190
    img = np.full((orig_h, orig_w, 3), 120, dtype=np.uint8)

    result = explainer.generate_cam(img)

    assert isinstance(result, GradCAMResult)
    assert result.heatmap.shape == (orig_h, orig_w)
    assert result.overlay_rgb.shape == (orig_h, orig_w, 3)
    assert result.original_rgb.shape == (orig_h, orig_w, 3)


def test_generate_cam_preserves_original_image(explainer):
    """Verify original input image is not mutated and is returned as a defensive copy."""
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img_copy = img.copy()

    result = explainer.generate_cam(img)

    # Input image passed in should be unchanged
    assert np.array_equal(img, img_copy)
    # Result original_rgb matches input exactly
    assert np.array_equal(result.original_rgb, img_copy)
    # Result original_rgb is a distinct object, not a reference
    assert result.original_rgb is not img


# ==============================================================================
# 3. NUMERIC VALIDITY & SAFE NORMALIZATION TESTS
# ==============================================================================


def test_heatmap_numeric_range_and_types(explainer):
    """Verify heatmap has float32 dtype within [0.0, 1.0] and overlay has uint8 dtype."""
    img = np.random.randint(50, 200, (120, 120, 3), dtype=np.uint8)
    result = explainer.generate_cam(img)

    assert result.heatmap.dtype == np.float32
    assert np.all(result.heatmap >= 0.0)
    assert np.all(result.heatmap <= 1.0)
    assert np.all(np.isfinite(result.heatmap))

    assert result.overlay_rgb.dtype == np.uint8
    assert np.all(result.overlay_rgb >= 0)
    assert np.all(result.overlay_rgb <= 255)

    assert isinstance(result.raw_logit, float)
    assert 0.0 <= result.probability <= 1.0
    assert result.predicted_label in ("real", "fake")


def test_safe_normalization_constant_activations(explainer):
    """Verify constant/zero activation maps normalize safely to uniform zeros without NaNs or Infs."""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)

    # Patch torch.sum in CAM computation to return a completely uniform tensor (spread = 0)
    orig_sum = torch.sum

    def mocked_uniform_sum(tensor, dim=1, keepdim=True):
        res = orig_sum(tensor, dim=dim, keepdim=keepdim)
        return torch.ones_like(res) * 5.0  # Constant uniform value

    with patch("torch.sum", side_effect=mocked_uniform_sum):
        result = explainer.generate_cam(img)

    assert result.heatmap.shape == (64, 64)
    assert np.all(np.isfinite(result.heatmap))
    # When all activations are constant, normalized_cam should safely be all zeros
    assert np.all(result.heatmap == 0.0)
    # Overlay should still be cleanly generated
    assert result.overlay_rgb.shape == (64, 64, 3)
    assert result.overlay_rgb.dtype == np.uint8


# ==============================================================================
# 4. HOOK AND GRADIENT CLEANUP TESTS
# ==============================================================================


def test_hooks_cleanup_on_success(explainer):
    """Verify PyTorch forward and backward hooks are cleanly removed after successful CAM generation."""
    # Before generation, target layer should have no hooks
    assert len(explainer.target_layer._forward_hooks) == 0
    assert len(explainer.target_layer._backward_hooks) == 0

    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    _ = explainer.generate_cam(img)

    # After generation, hooks must be completely removed
    assert len(explainer.target_layer._forward_hooks) == 0
    assert len(explainer.target_layer._backward_hooks) == 0


def test_hooks_cleanup_on_error(explainer):
    """Verify hooks are removed even when an exception occurs during the forward/backward pass."""
    img = np.full((64, 64, 3), 100, dtype=np.uint8)

    # Force a failure during the model forward pass
    with patch.object(explainer.model, "forward", side_effect=RuntimeError("Simulated GPU/CPU failure")):
        with pytest.raises(GradCAMComputationError) as exc:
            explainer.generate_cam(img)
        assert "Simulated GPU/CPU failure" in str(exc.value)

    # Guarantees hooks are cleaned up despite the exception
    assert len(explainer.target_layer._forward_hooks) == 0
    assert len(explainer.target_layer._backward_hooks) == 0


def test_no_grad_leakage_in_model_parameters(explainer):
    """Verify model parameters have gradients cleared after CAM generation."""
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    _ = explainer.generate_cam(img)

    for param in explainer.model.parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0.0)


# ==============================================================================
# 5. MODEL ARCHITECTURE & LAYER HANDLING TESTS
# ==============================================================================


def test_invalid_detector_type_raises():
    """Verify passing a non-DeepfakeDetector instance raises TypeError."""
    with pytest.raises(TypeError) as exc:
        GradCAMExplainer("not_a_detector")
    assert "Expected DeepfakeDetector instance" in str(exc.value)


def test_target_layer_automatic_discovery(mock_detector):
    """Verify default target layer automatically discovers conv_head on EfficientNet-B4."""
    explainer = GradCAMExplainer(mock_detector)
    assert explainer.target_layer_name == "conv_head"
    assert explainer.target_layer is getattr(mock_detector.model, "conv_head")


def test_target_layer_explicit_selection(mock_detector):
    """Verify explicit target layer selection (e.g. bn2) resolves correctly."""
    explainer = GradCAMExplainer(mock_detector, target_layer_name="bn2")
    assert explainer.target_layer_name == "bn2"
    assert explainer.target_layer is getattr(mock_detector.model, "bn2")

    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    result = explainer.generate_cam(img)
    assert result.target_layer_name == "bn2"


def test_target_layer_nonexistent_raises(mock_detector):
    """Verify requesting a nonexistent layer raises ValueError."""
    with pytest.raises(ValueError) as exc:
        GradCAMExplainer(mock_detector, target_layer_name="nonexistent_convolution_layer")
    assert "not found in model architecture" in str(exc.value)


def test_incompatible_model_without_conv_layers_raises():
    """Verify model with no convolutional layers raises GradCAMComputationError."""
    # Create a detector mock whose model is pure MLP (no Conv2d)
    detector = DeepfakeDetector(allow_uninitialized=True)
    detector.model = nn.Sequential(nn.Linear(10, 5), nn.Linear(5, 1))

    with pytest.raises(GradCAMComputationError) as exc:
        GradCAMExplainer(detector)
    assert "Could not locate a suitable convolutional feature layer" in str(exc.value)


# ==============================================================================
# 6. CPU COMPATIBILITY & CONFIGURATION TESTS
# ==============================================================================


def test_cpu_execution_compatibility(explainer):
    """Verify Grad-CAM runs seamlessly on CPU without requiring CUDA or GPU drivers."""
    assert explainer.device == torch.device("cpu")
    img = np.full((80, 80, 3), 150, dtype=np.uint8)

    result = explainer.generate_cam(img)

    assert isinstance(result.heatmap, np.ndarray)
    assert isinstance(result.overlay_rgb, np.ndarray)


def test_configurable_alpha_and_threshold(explainer):
    """Verify custom alpha blend weight and decision threshold behave properly."""
    img = np.full((64, 64, 3), 120, dtype=np.uint8)

    # Custom threshold forcing "fake"
    result_fake = explainer.generate_cam(img, alpha=0.7, threshold=0.0001)
    assert result_fake.predicted_label == "fake"

    # Custom threshold forcing "real"
    result_real = explainer.generate_cam(img, alpha=0.2, threshold=0.9999)
    assert result_real.predicted_label == "real"


# ==============================================================================
# 7. FORENSIC SAFETY & METADATA TESTS
# ==============================================================================


def test_forensic_disclaimer_and_dict_serialization(explainer):
    """Verify forensic disclaimer is present and dictionary representation is JSON-safe."""
    img = np.full((64, 64, 3), 100, dtype=np.uint8)
    result = explainer.generate_cam(img)

    # Check forensic disclaimer content
    assert "highlight spatial features that influenced the model's" in result.forensic_disclaimer
    assert "NOT constitute ground-truth pixel manipulation masks" in result.forensic_disclaimer
    assert "legal proof of forgery" in result.forensic_disclaimer

    # Check serialization
    res_dict = result.to_dict()
    assert isinstance(res_dict, dict)
    assert res_dict["target_layer_name"] == "conv_head"
    assert res_dict["heatmap_shape"] == [64, 64]
    assert res_dict["overlay_shape"] == [64, 64, 3]
    assert isinstance(res_dict["probability"], float)
