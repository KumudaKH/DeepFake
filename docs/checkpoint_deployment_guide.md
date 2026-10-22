# Checkpoint Deployment & Model Weight Integration Guide

**Project:** Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos  
**Component:** Model Checkpoint Ingestion & Dynamic Weight Binding  
**Status:** No completed training run or trained checkpoint is present in this workspace

---

## 1. Executive Summary

This document specifies the exact procedure, architectural requirements, key mappings, and verification steps required to transition the deepfake forensics pipeline from **Demo Mode** to **Trained Model**.

The FastAPI backend includes **dynamic checkpoint discovery**: when the trained checkpoint is saved into the project directory, the system automatically binds the trained weights on subsequent requests without requiring a server reboot.

---

## 2. Checkpoint Technical Specifications

### Architecture & Head Conventions
| Parameter | Required Specification | Verification Logic |
| :--- | :--- | :--- |
| **Model Architecture** | `efficientnet_b4` | Instantiated via `timm.create_model(..., pretrained=False, num_classes=1)` |
| **Input Shape** | $(B, 3, 380, 380)$ RGB, bilinear resize, then normalized float32 | Normalization: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)` |
| **Head Parameter Keys** | `classifier.weight` $[1, 1792]$, `classifier.bias` $[1]$ | Verified by `DeepfakeDetector.load_checkpoint()` |
| **Label Convention** | `0 = REAL / AUTHENTIC`, `1 = FAKE / MANIPULATED` | Single continuous logit $s \in \mathbb{R}$ |
| **Probability Calibration** | $P(\text{fake}) = \sigma(s) = \frac{1}{1 + e^{-s}}$ | Binary cross-entropy with logits loss formulation |
| **Weight Precision** | FP16 or FP32 | FP16 weights (`v.half()`) are automatically cast to FP32 on CPU devices |
| **State Dict Packaging** | `{"state_dict": ..., "metadata": ...}` exported by the existing benchmark notebook | Requires `state_dict_format`, training task, architecture, model config, preprocessing, label map, best epoch, and finite validation ROC-AUC; all state-dict keys and shapes must match |

---

## 3. Destination Paths

Save or copy the exported PyTorch checkpoint to the following path:

### Primary Expected Destination:
```
C:\Users\Nethr\OneDrive\Documents\DEEPFAKE\models\image\model_best_fp16.pt
```

### Alternative Accepted Destination:
```
C:\Users\Nethr\OneDrive\Documents\DEEPFAKE\models\image\model_best.pt
```

*(The backend checks these explicitly named best-inference paths only. Resume checkpoints and arbitrary `.pt` files are not loaded.)*

---

## 4. Colab Export & Transfer Walkthrough

When training completes in Google Colab:

### Step 1: Download from Colab
In your Colab notebook, run:
```python
from google.colab import files
files.download("model_best_fp16.pt")
```
Or download directly from the Colab left-hand File Explorer panel under your run output directory.

### Step 2: Copy to Project Folder
Move the downloaded `model_best_fp16.pt` file into:
```
C:\Users\Nethr\OneDrive\Documents\DEEPFAKE\models\image\model_best_fp16.pt
```

### Step 3: Instant Dynamic Reload
- If the FastAPI backend is currently running, refresh the browser at `http://localhost:5173`.
- The top-right navbar will immediately switch from:
  - `[Demo Mode (Mock Weights)]` (Amber) $\to$ `[Trained Model]` (Emerald).
- The dashboard alert switches from "Interface Demonstration Mode" to "Trained Deepfake Model Active".
- The health check at `GET http://127.0.0.1:8000/api/health` will return:
  ```json
  {
    "status": "healthy",
    "checkpoint_type": "deepfake_trained",
    "is_model_trained": true,
    "checkpoint_path": "models\\image\\model_best_fp16.pt"
  }
  ```

---

## 5. YuNet Face Detection Weights (Video Analysis)

Video frame face extraction and identity tracking requires the OpenCV YuNet ONNX model.

### Target Location:
```
C:\Users\Nethr\OneDrive\Documents\DEEPFAKE\models\yunet\face_detection_yunet_2023mar.onnx
```

### Status & Behavior:
- **When Missing:** The frontend displays an actionable amber alert banner on the Video Analysis tab and disables the analysis button. The backend returns a clean HTTP 503 error with installation advice if called directly.
- **When Present:** Video temporal extraction, IoU tracking, and keyframe isolation are automatically unlocked.
- Download from the official OpenCV Zoo:
  ```
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  ```

---

## 6. Current Workspace Finding and Training/Recovery Path

The inspected notebook is `notebooks/ffpp_dfd_face_forgery_benchmark_kaggle_t4x2 (1).ipynb`. It contains saved output from a 380px EfficientNet-B4 image-classification experiment and later epoch logs, but its relevant checkpoint was written to ephemeral Colab storage and is no longer present. `models/image/` contains only `.gitkeep`; no deepfake-trained weights or dataset are available locally. The YuNet ONNX file is only a face detector.

The notebook has been prepared to mount Google Drive explicitly and save `model_best_fp16.pt`, `model_resume.pt`, and training history under `/content/drive/MyDrive/deepfake_checkpoints/`. It saves an improving best checkpoint immediately and preserves an existing best file before replacement. Its existing experiment is retained: `efficientnet_b4`, one-logit head, 380px RGB input, ImageNet normalization, `REAL=0` / `FAKE=1`, AdamW (`1e-5`, weight decay `1e-4`), batch size 16, five epochs, patience 2. Training was not started.

To recover or produce a loadable checkpoint:

1. Check the original Colab runtime, mounted Drive, or exported archives for `model_best_fp16.pt`. A file-size log or a YuNet face-detection model is not a classifier checkpoint. Checkpoint provenance metadata and strict key/shape validation are required.
2. If no checkpoint exists, provide the notebook's configured dataset with `train`, `valid`, and `test` splits and real/fake class folders, open the existing notebook in a GPU-enabled Colab runtime, and run it. This resumes from the latest persistent resume checkpoint when available; it does not automatically download datasets.
3. Copy the notebook-produced `model_best_fp16.pt` into `models/image/model_best_fp16.pt`. Health changes to trained only after the metadata, preprocessing, label mapping, and entire state dict pass validation.
4. Run the notebook's held-out test evaluation after training. It reports accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix at the fixed 0.5 threshold, plus exact cross-split image-hash checks. Exact hashes do not detect near-duplicates, shared identities, or dataset shortcuts; validation ROC-AUC rounded to 1.0000 is not evidence of generalization.
5. Grad-CAM should be verified after loading the trained checkpoint. It highlights regions influential to the model score and does not prove the exact pixels that were manipulated.

## 7. Verification Test Suite

Verify that the model checkpoint loads without regressions:
```powershell
.\.venv\Scripts\python.exe -m pytest -v tests/test_image_detector.py
```
Expected result: All tests pass, validating state dict integrity, logit ranges, and deterministic inference.
