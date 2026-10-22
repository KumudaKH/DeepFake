"""Model architecture wrapper and inference engine for image and video deepfake detection.

Adapted from the FaceForensics++ / DFD Kaggle Benchmark.
Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos.
Academic Capstone Project.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image

from src.video_detection.face_tracker import ExtractedFace

# Standard ImageNet normalization parameters used by timm EfficientNet architectures
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
DEFAULT_INPUT_SIZE = 224
SUPPORTED_INPUT_TRANSFORM = "torchvision.transforms.Resize((380,380)), RGB ToTensor, ImageNet normalization"


@dataclass
class ImagePrediction:
    """Structured forensic prediction result for a single image or crop."""

    raw_logit: float
    probability: float  # Sigmoid of logit: P(manipulated / fake)
    predicted_label: str  # "real" or "fake"
    decision_threshold: float
    is_provisional_threshold: bool  # True unless calibrated on verified empirical data
    model_architecture: str
    checkpoint_path: Optional[str]
    checkpoint_type: str  # "deepfake_trained", "imagenet_pretrained", "mock", etc.
    device: str
    forensic_disclaimer: str = (
        "Model predictions represent probabilistic neural inferences and do NOT "
        "constitute definitive legal or judicial proof of manipulation."
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FramePrediction:
    """Forensic prediction result for an individual video frame crop,

    preserving complete temporal and evidentiary traceability.
    """

    video_path: str
    frame_index: int
    timestamp_sec: float
    box: Optional[Tuple[float, float, float, float]]
    tracking_status: str  # "detected", "neighbour", "none", etc.
    has_face: bool
    prediction: Optional[ImagePrediction]

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        if self.prediction:
            res["prediction"] = self.prediction.to_dict()
        return res


class DeepfakeDetector:
    """Inference wrapper for EfficientNet deepfake face classification models.

    Supports single-image and batched inference with explicit checkpoint auditing,
    configurable decision thresholds, and temporal metadata preservation.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b4.ra2_in1k",
        checkpoint_path: Optional[str | Path] = None,
        checkpoint_type: str = "deepfake_trained",
        device: str = "cpu",
        default_threshold: float = 0.5,
        allow_uninitialized: bool = False,
        input_size: int = DEFAULT_INPUT_SIZE,
    ) -> None:
        self.model_name = model_name
        self.checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        self.checkpoint_type = checkpoint_type
        self.default_threshold = float(default_threshold)
        if input_size <= 0:
            raise ValueError("input_size must be a positive integer.")
        self.input_size = int(input_size)

        # Configure device (safely defaults to CPU unless CUDA is explicitly requested and available)
        if device.lower().startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device.lower())
        else:
            self.device = torch.device("cpu")

        # Tensor normalization constants on target device
        self.mean_tensor = torch.tensor(DEFAULT_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std_tensor = torch.tensor(DEFAULT_STD, device=self.device).view(1, 3, 1, 1)

        # Build architecture with single output logit (num_classes=1)
        # Always initialize with pretrained=False to avoid silent automatic network downloads
        try:
            self.model: nn.Module = timm.create_model(
                self.model_name,
                pretrained=False,
                num_classes=1,
            )
        except Exception as e:
            raise ValueError(f"Failed to create model architecture '{self.model_name}': {type(e).__name__}: {str(e)}")

        self.model.to(self.device)
        self.model.eval()

        # Load weights if specified, or enforce strict checkpoint presence
        if self.checkpoint_path:
            self.load_checkpoint(self.checkpoint_path, checkpoint_type=self.checkpoint_type)
        elif not allow_uninitialized:
            raise FileNotFoundError(
                "A trained deepfake model checkpoint path must be provided. "
                "The detector refuses to silently run inference using uninitialized random weights. "
                "If testing an uninitialized architecture mock, explicitly pass `allow_uninitialized=True`."
            )
        else:
            self.checkpoint_type = "uninitialized_mock"

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        checkpoint_type: str = "deepfake_trained",
    ) -> None:
        """Load and validate locally available model weights with strict architecture compatibility checks."""
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint not found at: {path.resolve()}. "
                "Ensure trained weights (e.g., model_best_fp16.pt) are present."
            )

        try:
            checkpoint = torch.load(str(path.resolve()), map_location=self.device, weights_only=True)

            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint must be a state-dict mapping or a metadata-wrapped checkpoint.")

            metadata = checkpoint.get("metadata")
            if checkpoint_type == "deepfake_trained":
                if not isinstance(metadata, dict):
                    raise ValueError(
                        "Checkpoint has no verified deepfake-training metadata; refusing to mark it trained."
                    )
                expected_metadata = {
                    "format_version": 1,
                    "checkpoint_role": "best_inference",
                    "state_dict_format": "torch_state_dict_tensor_mapping",
                    "task": "deepfake_binary_classification",
                    "model_architecture": self.model_name,
                    "input_size": self.input_size,
                    "input_transform": SUPPORTED_INPUT_TRANSFORM,
                    "labels": {"real": 0, "fake": 1},
                }
                for key, expected_value in expected_metadata.items():
                    if metadata.get(key) != expected_value:
                        raise ValueError(
                            f"Checkpoint training metadata mismatch for '{key}': "
                            f"expected {expected_value!r}, got {metadata.get(key)!r}."
                        )

                model_config = metadata.get("model_config")
                if not isinstance(model_config, dict) or model_config.get("num_classes") != 1:
                    raise ValueError(
                        "Checkpoint metadata must identify the single-logit classifier head (num_classes=1)."
                    )

                mean = np.asarray(metadata.get("mean"), dtype=np.float64)
                std = np.asarray(metadata.get("std"), dtype=np.float64)
                if (
                    mean.shape != (3,)
                    or std.shape != (3,)
                    or not np.all(np.isfinite(mean))
                    or not np.all(np.isfinite(std))
                    or not np.allclose(mean, DEFAULT_MEAN, rtol=0.0, atol=1e-6)
                    or not np.allclose(std, DEFAULT_STD, rtol=0.0, atol=1e-6)
                ):
                    raise ValueError(
                        "Checkpoint training normalization does not match the detector's RGB preprocessing."
                    )

                best_epoch = metadata.get("best_epoch")
                validation_metric = metadata.get("validation_metric")
                validation_auc = (
                    validation_metric.get("value")
                    if isinstance(validation_metric, dict)
                    and validation_metric.get("name") == "roc_auc"
                    else None
                )
                if (
                    not isinstance(best_epoch, int)
                    or best_epoch < 1
                    or not isinstance(validation_auc, (int, float))
                    or not np.isfinite(validation_auc)
                    or not 0.0 <= validation_auc <= 1.0
                ):
                    raise ValueError(
                        "Checkpoint metadata must record a completed epoch and finite validation ROC-AUC."
                    )

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            if not isinstance(state_dict, dict) or not state_dict:
                raise ValueError("Checkpoint does not contain a non-empty model state dict.")
            if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items()):
                raise ValueError("Checkpoint state dict must contain only string keys and tensor values.")

            clean_state_dict: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                clean_key = k[7:] if k.startswith("module.") else k
                if clean_key in clean_state_dict:
                    raise ValueError(f"Checkpoint contains duplicate state-dict key after prefix cleanup: {clean_key}")
                # Cast FP16 weights to FP32 if running on CPU to prevent CPU FP16 execution errors
                if self.device.type == "cpu" and v.is_floating_point():
                    clean_state_dict[clean_key] = v.float()
                else:
                    clean_state_dict[clean_key] = v

            expected_state_dict = self.model.state_dict()
            missing_keys = expected_state_dict.keys() - clean_state_dict.keys()
            unexpected_keys = clean_state_dict.keys() - expected_state_dict.keys()
            mismatched_shapes = [
                f"{key}: expected {tuple(expected_state_dict[key].shape)}, got {tuple(clean_state_dict[key].shape)}"
                for key in expected_state_dict.keys() & clean_state_dict.keys()
                if expected_state_dict[key].shape != clean_state_dict[key].shape
            ]
            if missing_keys or unexpected_keys or mismatched_shapes:
                raise ValueError(
                    f"Checkpoint at {path.name} does not exactly match architecture '{self.model_name}' "
                    "and its single-logit classifier. "
                    f"Missing keys: {sorted(missing_keys)}; unexpected keys: {sorted(unexpected_keys)}; "
                    f"shape mismatches: {mismatched_shapes}"
                )

            try:
                self.model.load_state_dict(clean_state_dict, strict=True)
            except RuntimeError as exc:
                raise ValueError(
                    f"Checkpoint at {path.name} does not exactly match architecture "
                    f"'{self.model_name}' and its single-logit classifier: {exc}"
                ) from exc

            self.checkpoint_path = str(path.resolve())
            self.checkpoint_type = checkpoint_type
            self.model.eval()

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Incompatible or corrupt checkpoint at {path}: {type(e).__name__}: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Validate and preprocess a single image array into a normalized PyTorch tensor.

        Expects an RGB uint8 image of shape (H, W, 3).
        Returns a normalized float32 tensor of shape (1, 3, input_size, input_size) on self.device.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray image, got {type(image).__name__}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape {image.shape}")

        h, w = image.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image dimensions: {w}x{h}")

        # Match torchvision Resize on the notebook's PIL RGB inputs.
        if h != self.input_size or w != self.input_size:
            resized = np.array(
                Image.fromarray(image).resize(
                    (self.input_size, self.input_size),
                    resample=Image.Resampling.BILINEAR,
                ),
                dtype=np.uint8,
            )
        else:
            resized = image

        # Convert uint8 [0, 255] to float32 [0.0, 1.0]
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(self.device)

        # Apply ImageNet normalization: (x - mean) / std
        return (tensor - self.mean_tensor) / self.std_tensor

    def preprocess_batch(self, images: List[np.ndarray]) -> torch.Tensor:
        """Batch-preprocess multiple RGB face crops into a stacked (B, 3, 224, 224) tensor."""
        if not images:
            raise ValueError("Cannot preprocess an empty list of images.")
        tensors = [self.preprocess_image(img) for img in images]
        return torch.cat(tensors, dim=0)

    @torch.no_grad()
    def predict_image(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
    ) -> ImagePrediction:
        """Perform deepfake forgery prediction on a single facial image.

        Args:
            image: RGB image crop as numpy array (H, W, 3).
            threshold: Optional decision threshold. Defaults to self.default_threshold (0.5).

        Returns:
            ImagePrediction containing raw logit, sigmoid probability, label, and metadata.
        """
        tensor = self.preprocess_image(image)
        self.model.eval()

        raw_output = self.model(tensor)
        logit = float(raw_output.view(-1)[0].item())

        # Sigmoid calibration: P(fake) = 1 / (1 + exp(-s))
        probability = float(torch.sigmoid(torch.tensor(logit)).item())

        effective_threshold = float(threshold if threshold is not None else self.default_threshold)
        predicted_label = "fake" if probability >= effective_threshold else "real"

        return ImagePrediction(
            raw_logit=round(logit, 4),
            probability=round(probability, 4),
            predicted_label=predicted_label,
            decision_threshold=effective_threshold,
            is_provisional_threshold=True,  # Clearly indicates uncalibrated demo threshold
            model_architecture=self.model_name,
            checkpoint_path=self.checkpoint_path,
            checkpoint_type=self.checkpoint_type,
            device=str(self.device),
        )

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[np.ndarray],
        threshold: Optional[float] = None,
    ) -> List[ImagePrediction]:
        """Perform batched inference over multiple face crops simultaneously for high throughput."""
        if not images:
            return []

        batch_tensor = self.preprocess_batch(images)
        self.model.eval()

        raw_outputs = self.model(batch_tensor).view(-1)
        logits = raw_outputs.tolist()
        probabilities = torch.sigmoid(raw_outputs).tolist()

        effective_threshold = float(threshold if threshold is not None else self.default_threshold)
        results: List[ImagePrediction] = []

        for logit_val, prob_val in zip(logits, probabilities):
            pred_label = "fake" if prob_val >= effective_threshold else "real"
            results.append(
                ImagePrediction(
                    raw_logit=round(float(logit_val), 4),
                    probability=round(float(prob_val), 4),
                    predicted_label=pred_label,
                    decision_threshold=effective_threshold,
                    is_provisional_threshold=True,
                    model_architecture=self.model_name,
                    checkpoint_path=self.checkpoint_path,
                    checkpoint_type=self.checkpoint_type,
                    device=str(self.device),
                )
            )

        return results

    def predict_video_faces(
        self,
        extracted_faces: List[ExtractedFace],
        threshold: Optional[float] = None,
    ) -> List[FramePrediction]:
        """Score each extracted face from video sampling, preserving temporal frame metadata.

        Note: Consecutive frame scores are temporally correlated and must not be treated
        as statistically independent observations.
        """
        if not extracted_faces:
            return []

        # Filter valid face crops for batched inference
        valid_indices: List[int] = []
        valid_crops: List[np.ndarray] = []

        for idx, face in enumerate(extracted_faces):
            if face.crop_rgb is not None:
                valid_indices.append(idx)
                valid_crops.append(face.crop_rgb)

        predictions_by_index: Dict[int, ImagePrediction] = {}
        if valid_crops:
            batch_predictions = self.predict_batch(valid_crops, threshold=threshold)
            for original_idx, pred in zip(valid_indices, batch_predictions):
                predictions_by_index[original_idx] = pred

        # Map predictions back to the original ExtractedFace metadata
        frame_predictions: List[FramePrediction] = []
        for idx, face in enumerate(extracted_faces):
            has_face = idx in predictions_by_index
            pred = predictions_by_index.get(idx, None)

            frame_predictions.append(
                FramePrediction(
                    video_path=face.video_path,
                    frame_index=face.frame_index,
                    timestamp_sec=face.timestamp_sec,
                    box=face.box,
                    tracking_status=face.tracking_status,
                    has_face=has_face,
                    prediction=pred,
                )
            )

        return frame_predictions
