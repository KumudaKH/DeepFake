# Model Architecture & Inference Engine Guide
## Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos

**Academic Context:** Final-Year Computer Science and Engineering Capstone Project  
**Author / Examiner Scope:** Image and Video Face Forgery Authentication  
**Module:** [`src/image_detection/detector.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/image_detection/detector.py)  
**Date:** October 2026

---

## 1. Overview and Architecture Specification

The image classification engine uses the **EfficientNet-B4** architecture (`efficientnet_b4` via `timm`) matching the existing training notebook. No genuine trained checkpoint is currently available in the workspace, so the application remains in Demo Mode until verified weights are loaded.

```
+-----------------------------------------------------------------------------------+
|                         INPUT: RGB Image / Crop (380 x 380)                       |
+-----------------------------------------------------------------------------------+
                                          |
                              ImageNet Normalization
                      Mean: (0.485, 0.456, 0.406), Std: (0.229, 0.224, 0.225)
                                          |
+-----------------------------------------------------------------------------------+
|                        EFFICIENTNET-B4 CONVOLUTIONAL BACKBONE                     |
|           Stages 1-7: Mobile Inverted Bottleneck Convolutions (MBConv)            |
|                           Feature Dimension: 1792-d                               |
+-----------------------------------------------------------------------------------+
                                          |
                              Global Average Pooling (1x1)
                                          |
+-----------------------------------------------------------------------------------+
|                          CLASSIFIER HEAD (Single Logit)                           |
|                         nn.Linear(in_features=1792, out_features=1)               |
+-----------------------------------------------------------------------------------+
                                          |
                              Continuous Logit s in R
                                          |
                     Sigmoid Calibration: P(Fake) = 1 / (1 + e^-s)
                                          |
                     Decision Threshold Comparison: P >= Threshold
```

### Architectural Hyperparameters
* **Backbone:** EfficientNet-B4 (`timm`) with ImageNet-pretrained training initialization.
* **Input Resolution:** $380 \times 380 \times 3$ RGB pixels; direct images and tracked video face crops use the same resize.
* **Classifier Head:** Single linear layer mapping the 1,792-dimensional pooled feature vector to a continuous scalar logit $s \in \mathbb{R}$.
* **Loss Function (during training):** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`), inversely weighted to class sizes.

---

## 2. Checkpoint Loading & Security Specifications

In compliance with academic integrity and application security standards:

1. **No Silent Uninitialized Execution**:
   The detector strictly refuses to run inference on uninitialized random weights by default. Attempting to initialize without a valid checkpoint raises an actionable `FileNotFoundError`. For interface unit testing, uninitialized execution must be explicitly authorized via `allow_uninitialized=True`.
2. **Safe Deserialization (`weights_only=True`)**:
   Model checkpoint loading enforces `torch.load(..., weights_only=True)` to prevent arbitrary code execution attacks via malicious pickle payloads.
3. **Architecture Compatibility Validation**:
   A checkpoint is marked `deepfake_trained` only when its metadata identifies the deepfake task, `efficientnet_b4`, the single-logit head, 380-pixel preprocessing, matching normalization, `REAL=0` / `FAKE=1` labels, state-dict packaging, a completed best epoch, and finite validation ROC-AUC. The complete state dict is then loaded strictly; missing, unexpected, or shape-mismatched tensors raise an explicit `ValueError`. Raw state dicts (including ImageNet or random weights) cannot mark the service trained.
4. **Precision & Device Management**:
   Checkpoints trained in FP16 (e.g. `model_best_fp16.pt` from Kaggle 2× T4 GPUs) are automatically cast to FP32 when running on CPU to prevent CPU execution incompatibility. Inference defaults to CPU and enables CUDA only when explicitly requested and available.
5. **Readiness**:
   `/api/health` reports `model_readiness` as `trained` only after metadata and strict state-dict checks succeed. Without such a checkpoint, the service remains in demo mode and its verdict is inconclusive.

---

## 3. Inference Interfaces & Usage

The detector is implemented in [`src/image_detection/detector.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/image_detection/detector.py).

### 3.1 Single-Image Inference

```python
from pathlib import Path
import cv2
from src.image_detection.detector import DeepfakeDetector

# Initialize detector with a verified checkpoint
detector = DeepfakeDetector(
    model_name="efficientnet_b4",
    checkpoint_path="models/image/model_best_fp16.pt",
    device="cpu",
    default_threshold=0.5,
    input_size=380,
)

# Load RGB image; the detector resizes it to 380x380 before ImageNet normalization.
img_bgr = cv2.imread("data/raw/images/sample.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Run inference
prediction = detector.predict_image(img_rgb)
print(f"Logit: {prediction.raw_logit}")
print(f"P(Fake): {prediction.probability:.4f}")
print(f"Verdict: {prediction.predicted_label.upper()}")
```

### 3.2 Batched Video Face Scoring

When processing videos, the face tracker (`src/video_detection/face_tracker.py`) extracts a sequence of temporal crops. The detector scores these crops in a single batched tensor pass while preserving temporal traceability:

```python
from src.video_detection.face_tracker import FaceTrackerPipeline
from src.image_detection.detector import DeepfakeDetector

tracker = FaceTrackerPipeline(yunet_model_path="models/yunet/face_detection_yunet_2023mar.onnx")
result = tracker.extract_from_video("data/raw/videos/sample.mp4", max_frames=16)

# Score all extracted face crops
detector = DeepfakeDetector(
    model_name="efficientnet_b4",
    checkpoint_path="models/image/model_best_fp16.pt",
    input_size=380,
)
frame_predictions = detector.predict_video_faces(result.extracted_faces)

for fp in frame_predictions:
    if fp.has_face:
        print(f"Frame {fp.frame_index} ({fp.timestamp_sec}s): P(Fake) = {fp.prediction.probability:.4f} [{fp.prediction.predicted_label}]")
    else:
        print(f"Frame {fp.frame_index} ({fp.timestamp_sec}s): No face detected (Unscorable)")
```

---

## 4. Threshold Configuration & Empirical Calibration Safety

### Decision Threshold Rules
* **Configurable Parameter:** The decision threshold is passed as an explicit parameter (default $0.5$) rather than being hardcoded into the network forward pass.
* **Provisional Warning Flag:** All predictions explicitly include `is_provisional_threshold=True` until temperature scaling or Platt scaling has been empirically computed on held-out validation data (e.g. using the validation split of FaceForensics++ or 140k Real & Fake Faces).
* **Non-Conclusive Nature:** The system does not claim that a probability score constitutes definitive legal proof of manipulation or personal wrongdoing. Every prediction includes a structured disclaimer string:
  > *"Model predictions represent probabilistic neural inferences and do NOT constitute definitive legal or judicial proof of manipulation."*

---

## 5. Structured Output Schemas

### `ImagePrediction` (Dataclass)
```json
{
  "raw_logit": 2.4512,
  "probability": 0.9207,
  "predicted_label": "fake",
  "decision_threshold": 0.5,
  "is_provisional_threshold": true,
  "model_architecture": "efficientnet_b4",
  "checkpoint_path": "models/image/model_best_fp16.pt",
  "checkpoint_type": "deepfake_trained",
  "device": "cpu",
  "forensic_disclaimer": "Model predictions represent probabilistic neural inferences and do NOT constitute definitive legal or judicial proof of manipulation."
}
```

### `FramePrediction` (Dataclass)
```json
{
  "video_path": "C:\\videos\\sample.mp4",
  "frame_index": 18,
  "timestamp_sec": 0.72,
  "box": [45.0, 32.0, 110.0, 110.0],
  "tracking_status": "detected",
  "has_face": true,
  "prediction": { ... ImagePrediction ... }
}
```

---

## 6. Current Implementation Status & Next Stages

- [x] **Model Wrapper Architecture**: Fully implemented with single logit, ImageNet normalization, and evaluation determinism.
- [x] **Safe Checkpoint Management**: Implemented with `weights_only=True` and strict uninitialized execution guards.
- [x] **Batch & Video Scoring Integration**: Implemented with temporal metadata and missing-face preservation.
- [x] **Unit Test Suite**: 13 comprehensive tests in [`tests/test_image_detector.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/tests/test_image_detector.py) (all 32 repository tests passing).
- [x] **Stage 8**: Implemented the **Grad-CAM Explainability Module** ([`src/explainability/gradcam.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/explainability/gradcam.py)) targeting the final convolutional head of EfficientNet-B4.
- [ ] **Stage 9 (Next)**: Implement the **Temporal Aggregator & Keyframe Selector** (`src/video_detection/temporal_aggregator.py`).
- [ ] **Stage 10**: Implement the **ReportLab Digital Forensics PDF Generator** (`src/forensics/report_generator.py`).
