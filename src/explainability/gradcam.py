"""Grad-CAM (Gradient-weighted Class Activation Mapping) explainability module.

Generates spatial interpretability heatmaps for EfficientNet-B4 face forgery predictions
without requiring Matplotlib (pure OpenCV, NumPy, and PyTorch implementation).

Adapted from the FaceForensics++ / DFD Kaggle Benchmark.
Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos.
Academic Capstone Project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from src.image_detection.detector import DeepfakeDetector


class GradCAMComputationError(Exception):
    """Raised when Grad-CAM activation maps cannot be computed."""

    pass


@dataclass
class GradCAMResult:
    """Forensic explainability result containing raw heatmap and visual overlays."""

    heatmap: np.ndarray  # (H, W) float32 array normalized to [0.0, 1.0]
    overlay_rgb: np.ndarray  # (H, W, 3) uint8 RGB blended overlay
    original_rgb: np.ndarray  # (H, W, 3) uint8 RGB input image
    target_layer_name: str
    raw_logit: float
    probability: float
    predicted_label: str
    checkpoint_type: str
    forensic_disclaimer: str = (
        "Grad-CAM heatmaps highlight spatial features that influenced the model's "
        "classification decision. They do NOT constitute ground-truth pixel manipulation masks "
        "or legal proof of forgery."
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert scalar metadata to dictionary (arrays excluded from basic JSON)."""
        return {
            "target_layer_name": self.target_layer_name,
            "raw_logit": round(float(self.raw_logit), 4),
            "probability": round(float(self.probability), 4),
            "predicted_label": self.predicted_label,
            "checkpoint_type": self.checkpoint_type,
            "heatmap_shape": list(self.heatmap.shape),
            "overlay_shape": list(self.overlay_rgb.shape),
            "forensic_disclaimer": self.forensic_disclaimer,
        }


class GradCAMExplainer:
    """Computes Grad-CAM spatial interpretability heatmaps for an EfficientNet-B4 detector.

    Hooks the final convolutional feature extractor, calculates gradient-weighted channel
    importance, and generates high-resolution RGB overlays using OpenCV.
    """

    def __init__(
        self,
        detector: DeepfakeDetector,
        target_layer_name: Optional[str] = None,
    ) -> None:
        if not isinstance(detector, DeepfakeDetector):
            raise TypeError(f"Expected DeepfakeDetector instance, got {type(detector).__name__}")

        self.detector = detector
        self.model = detector.model
        self.device = detector.device

        # Locate target convolutional feature layer
        self.target_layer_name, self.target_layer = self._find_target_layer(target_layer_name)

    def _find_target_layer(self, layer_name: Optional[str]) -> Tuple[str, nn.Module]:
        """Locate the target feature layer for gradient hook registration."""
        if layer_name:
            # Check direct attribute or named submodules
            if hasattr(self.model, layer_name):
                return layer_name, getattr(self.model, layer_name)
            for name, module in self.model.named_modules():
                if name == layer_name:
                    return name, module
            raise ValueError(f"Requested target layer '{layer_name}' not found in model architecture.")

        # Default hierarchy for timm EfficientNet models:
        # 1. conv_head (final 1x1 conv mapping 448 -> 1792 channels)
        # 2. bn2 (batch norm after conv_head)
        # 3. Last Conv2d module in the network
        if hasattr(self.model, "conv_head") and isinstance(getattr(self.model, "conv_head"), nn.Module):
            return "conv_head", getattr(self.model, "conv_head")
        if hasattr(self.model, "bn2") and isinstance(getattr(self.model, "bn2"), nn.Module):
            return "bn2", getattr(self.model, "bn2")

        # Fallback search for the last Conv2d module before classifier
        last_conv_name = None
        last_conv_mod = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name = name
                last_conv_mod = module

        if last_conv_mod is not None:
            return last_conv_name, last_conv_mod

        raise GradCAMComputationError("Could not locate a suitable convolutional feature layer in the model.")

    def generate_cam(
        self,
        image_rgb: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
        threshold: Optional[float] = None,
    ) -> GradCAMResult:
        """Generate Grad-CAM spatial heatmap and visual overlay for an RGB image crop.

        Args:
            image_rgb: (H, W, 3) uint8 numpy array in RGB format.
            alpha: Blend weight for the heatmap overlay (0.0 = only image, 1.0 = only heatmap).
            colormap: OpenCV colormap enum (default: cv2.COLORMAP_JET).
            threshold: Optional decision threshold for predicted label.

        Returns:
            GradCAMResult containing normalized heatmap, overlay, and prediction metadata.
        """
        if not isinstance(image_rgb, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray image, got {type(image_rgb).__name__}")

        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {image_rgb.shape}")

        orig_h, orig_w = image_rgb.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            raise ValueError(f"Invalid image dimensions: {orig_w}x{orig_h}")

        # Hook storage containers
        forward_activations: list[torch.Tensor] = []
        backward_gradients: list[torch.Tensor] = []

        def forward_hook(module: nn.Module, input: Any, output: torch.Tensor) -> None:
            forward_activations.append(output)

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
            if grad_output and grad_output[0] is not None:
                backward_gradients.append(grad_output[0])

        fwd_handle = None
        bwd_handle = None

        try:
            # Register hooks
            fwd_handle = self.target_layer.register_forward_hook(forward_hook)
            bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

            # Preprocess image tensor
            input_tensor = self.detector.preprocess_image(image_rgb)
            input_tensor.requires_grad_(True)

            # Ensure model is in eval mode, clear existing gradients
            self.model.eval()
            self.model.zero_grad()

            # Forward pass (gradients enabled for backward pass)
            with torch.enable_grad():
                output = self.model(input_tensor)
                raw_logit = float(output.view(-1)[0].item())
                target_score = output.view(-1)[0]

                # Backpropagate to compute gradients of the manipulation score w.r.t. target layer activations
                target_score.backward()

            if not forward_activations or not backward_gradients:
                raise GradCAMComputationError(
                    "Hooks failed to capture activations or gradients during the forward/backward pass."
                )

            # Extract captured tensors: Shape (1, C, H_feat, W_feat)
            acts = forward_activations[0].detach()
            grads = backward_gradients[0].detach()

            # Global average pooling over spatial dimensions (H_feat, W_feat) to get channel weights alpha_c
            weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Linear combination of feature maps weighted by alpha_c
            cam = torch.sum(weights * acts, dim=1, keepdim=True)  # (1, 1, H_feat, W_feat)

            # Apply ReLU: only features positively contributing to the forgery logit are visualized
            cam = torch.relu(cam)

            # Convert to 2D numpy array
            cam_np = cam.squeeze().cpu().numpy().astype(np.float32)

            # Safe normalization to [0.0, 1.0]
            cam_min = float(np.min(cam_np))
            cam_max = float(np.max(cam_np))
            spread = cam_max - cam_min

            if spread > 1e-8:
                normalized_cam = (cam_np - cam_min) / spread
            else:
                # Constant or near-zero activation: safely return uniform zeros to prevent NaN / division by zero
                normalized_cam = np.zeros_like(cam_np, dtype=np.float32)

            # Resize heatmap back to original input image dimensions
            heatmap_resized = cv2.resize(normalized_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0).astype(np.float32)

            # Generate OpenCV RGB overlay
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
            colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)

            # Blend original RGB image with colored heatmap
            alpha = float(np.clip(alpha, 0.0, 1.0))
            overlay_rgb = cv2.addWeighted(
                image_rgb,
                1.0 - alpha,
                colored_heatmap_rgb,
                alpha,
                0,
            )

            # Compute calibrated probability and label
            probability = float(torch.sigmoid(torch.tensor(raw_logit)).item())
            effective_threshold = float(
                threshold if threshold is not None else self.detector.default_threshold
            )
            predicted_label = "fake" if probability >= effective_threshold else "real"

            return GradCAMResult(
                heatmap=heatmap_resized,
                overlay_rgb=overlay_rgb,
                original_rgb=image_rgb.copy(),
                target_layer_name=self.target_layer_name,
                raw_logit=raw_logit,
                probability=probability,
                predicted_label=predicted_label,
                checkpoint_type=self.detector.checkpoint_type,
            )

        except Exception as e:
            if isinstance(e, (TypeError, ValueError, GradCAMComputationError)):
                raise
            raise GradCAMComputationError(f"Grad-CAM generation failed: {type(e).__name__}: {str(e)}")

        finally:
            # Guarantees hooks are always removed, even if an exception occurs
            if fwd_handle is not None:
                fwd_handle.remove()
            if bwd_handle is not None:
                bwd_handle.remove()
            self.model.zero_grad()
