"""Explainable AI (XAI) module: Grad-CAM and spatial interpretability heatmaps for facial forgery forensics."""

from src.explainability.gradcam import (
    GradCAMComputationError,
    GradCAMExplainer,
    GradCAMResult,
)

__all__ = [
    "GradCAMExplainer",
    "GradCAMResult",
    "GradCAMComputationError",
]
