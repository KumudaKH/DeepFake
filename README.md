# Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos

[![Python 3.11](https://img.shields.io/badge/python-3.11.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![timm](https://img.shields.io/badge/timm-0.9+-green.svg)](https://github.com/huggingface/pytorch-image-models)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)](#)

A comprehensive, academic-grade system for detecting image face manipulations and video deepfakes using deep learning, accompanied by transparent interpretability heatmaps (Grad-CAM), temporal identity tracking across video frames, and automated, evidence-backed forensic PDF reports.

Developed as a **Final-Year Computer Science and Engineering Capstone Project**.

---

## 1. Executive Summary & Problem Statement

The rapid evolution of generative deep learning models (Diffusion models, GANs, Neural Head Reenactments, and DeepFaceLab/FaceSwap pipelines) has made hyper-realistic facial forgery widespread. This poses severe societal risks:
- **Misinformation and impersonation** in video broadcasts and social platforms.
- **Biometric authentication bypass** in facial verification gates.
- **Black-box forensic skepticism**: Raw probability outputs lack transparent, audit-ready rationale required in judicial inquiries and security audits.

This project addresses these challenges by developing a multimodal detection system for **images and videos** that couples deep convolutional feature extraction (EfficientNet-B4 / Vision Transformers) with **explainable AI (XAI)** and automated digital forensics reporting. Every forensic assessment provides:
1. Probabilistic classification with calibrated confidence scores.
2. Visual explainability artifacts (Grad-CAM spatial heatmaps isolating facial manipulation artifacts).
3. Video temporal analysis (uniform frame sampling, YuNet face tracking, frame-level anomaly curves, and aggregate video scores).
4. Automated, tamper-evident PDF forensic reports complete with cryptographic file hashes (SHA-256), EXIF container metadata, and explicit methodology disclosures.

---

## 2. Core System Architecture

```
+-----------------------------------------------------------------------------------+
|                                  USER INTERFACE                                   |
|                React / Vite Modern Forensic Dashboard & Upload Portal             |
+-----------------------------------------------------------------------------------+
                                          |
                                    REST API (FastAPI)
                                          |
          +-------------------------------+-------------------------------+
          |                                                               |
+-----------------------+                                       +-----------------------+
|  SINGLE-IMAGE MODULE  |                                       |   VIDEO FORENSICS     |
|  - Preprocessing      |                                       |  - Uniform Sampling   |
|  - YuNet Face Crop    |                                       |  - IoU Face Tracker   |
|  - EfficientNet-B4    |                                       |  - Per-Frame Scoring  |
|  - Grad-CAM Heatmap   |                                       |  - Temporal Pooling   |
|  - Probability Output |                                       |  - Keyframe Extraction|
+-----------------------+                                       +-----------------------+
          |                                                               |
          +-------------------------------+-------------------------------+
                                          |
+-----------------------------------------------------------------------------------+
|                        FORENSIC REPORT GENERATION ENGINE                          |
|                       ReportLab Automated PDF Generator                           |
|  - File SHA-256 Hash & Container Metadata - Calibrated Confidence Assessment       |
|  - Grad-CAM Heatmap Overlays              - Temporal Anomaly Plot & Keyframes     |
|  - Plain-English Forensic Summary         - Methodological Uncertainty & Audit    |
+-----------------------------------------------------------------------------------+
```

### Module A: Single-Image Face Forgery Detection
- **Input**: User-uploaded images (JPEG, PNG, WebP).
- **Processing**: YuNet face detection, spatial normalization ($224 \times 224$ pixels), and ImageNet standardization.
- **Classification**: Binary classification (Likely Authentic vs. Potentially Manipulated).
- **Explainability**: Gradient-weighted Class Activation Mapping (Grad-CAM) computed on the final convolutional layer of EfficientNet-B4.
- **Forensic Distinction**: Distinguishes between model attention heatmaps and verified ground-truth manipulation boundaries.

### Module B: Video Deepfake Detection & Temporal Aggregation
- **Input**: User-uploaded video recordings (MP4, AVI, MOV, MKV, WebM).
- **Frame Sampling**: Uniform temporal sampling ($k = 16$ frames spread evenly over a 300-frame window).
- **Face Tracking**: High-speed ONNX YuNet detector coupled with IoU identity tracking ($IoU > 0.3$) across consecutive frames to maintain consistent subject focus.
- **Neural Inference**: Frame-level inference using fine-tuned EfficientNet-B4.
- **Temporal Aggregation**: Mean logit pooling ($\bar{s} = \frac{1}{k}\sum s_j$) calibrated via logistic sigmoid to produce a video-level manipulation score.
- **Keyframe Saliency**: Automatically identifies the frame exhibiting the highest anomaly logit and computes Grad-CAM for in-depth forensic inspection.

### Module C: Explainable Digital Forensics & PDF Reporting
- **Automated Report Engine**: Generates professional, publication-quality digital forensics reports using ReportLab.
- **Contents**:
  1. File metadata (Name, Size, MIME type, SHA-256 cryptographic hash, analysis timestamp).
  2. Model specification, version, and architecture details.
  3. Calibrated confidence rating with categorical uncertainty boundaries.
  4. Embedded visual evidence (original image/keyframe, Grad-CAM heatmap overlay, and temporal anomaly curves for video).
  5. Plain-English forensic summary and interpretation.
  6. Explicit statement of limitations, non-conclusive disclaimer, and recommended secondary verification methods.

---

## 3. Repository Directory Structure

```
Deepfake_Forensics_System/
├── README.md               # Project documentation and architectural overview
├── .gitignore              # Git ignore rules for deep learning, data, and models
├── requirements.txt        # Python dependency manifest (PyTorch, timm, OpenCV, ReportLab, FastAPI)
├── .env.example            # Environment variable configuration template
├── data/                   # Data directory (kept out of version control)
│   ├── raw/                # Unprocessed source datasets
│   │   ├── images/         # Raw benchmark image datasets (e.g., 140k Real & Fake Faces)
│   │   └── videos/         # Raw benchmark video datasets (e.g., FaceForensics++ c23)
│   └── processed/          # Preprocessed, aligned, and split data
│       ├── images/         # Normalized 224px face crops
│       └── frames/         # Extracted video frames and crop caches
├── models/                 # Model weights and serialized checkpoints
│   ├── image/              # Trained image detection checkpoints & weights
│   ├── video/              # Trained video detection checkpoints & weights
│   └── yunet/              # OpenCV YuNet ONNX face detection weights
├── src/                    # Core Python application source code
│   ├── data/               # Dataset loaders, face alignment, and preprocessing scripts
│   ├── image_detection/    # EfficientNet-B4 architecture, training, and inference for images
│   ├── video_detection/    # Video decoding, YuNet face tracking, and temporal aggregation
│   ├── explainability/     # Grad-CAM, saliency mapping, and visual evidence overlay
│   ├── forensics/          # ReportLab PDF report builder, hashing, and evidence synthesis
│   └── backend/            # FastAPI service application
│       └── app/
│           ├── api/        # REST route definitions (analyze, report, health)
│           ├── services/   # Business logic (inference dispatch, file handlers)
│           └── schemas/    # Pydantic request and response schemas
├── frontend/               # React / Vite forensic dashboard and reporting interface
├── notebooks/              # Jupyter research notebooks (training, evaluation, Kaggle benchmark)
│   └── ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb
├── reports/                # Generated forensic PDF reports output folder
├── docs/                   # Academic project documentation, design specs, review slides
│   └── dataset_selection_report.md
└── tests/                  # Unit and integration test suites
```

---

## 4. Technology Stack

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11.x | Core machine learning, video analysis, backend API, and forensics |
| **Deep Learning** | PyTorch, Torchvision, timm | Neural network modeling, pre-trained EfficientNet-B4, GPU acceleration |
| **Data & Scientific** | NumPy, Pandas, Scikit-learn, SciPy | Data wrangling, calibration metrics, evaluation metrics (AUC, EER, AP) |
| **Computer Vision** | OpenCV (`cv2.FaceDetectorYN`), Pillow | Video decoding, YuNet ONNX face detection, IoU tracking, resizing |
| **Visualization & XAI** | Matplotlib, Seaborn, Grad-CAM | Explanatory heatmap generation and temporal score trajectory plotting |
| **Backend API** | FastAPI, Uvicorn, Pydantic | High-performance asynchronous REST API for image and video analysis |
| **Forensic Reports** | ReportLab | Automated, cryptographically linked PDF report generation |
| **Frontend UI** | React, Vite | Interactive web portal for file analysis and result inspection |
| **Version Control** | Git | Distributed version control and reproducible tracking |

---

## 5. Development Roadmap & Working Method

- [x] **Stage 1: Initial Setup & Environment Verification** (Completed)
- [x] **Stage 2: Dataset Selection, Verification & Benchmark Audit** (Completed)
  - Scope refined to Image and Video Deepfake Forensics.
  - Thorough audit of supplied benchmark notebook (`ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb`).
  - Selection of 140k Real & Fake Faces (image) and FaceForensics++ c23 / Google DFD (video).
  - Formalization of YuNet face tracking, uniform frame sampling, and temporal aggregation.
- [ ] **Stage 3: Environment Setup & Core Utility Library Integration**
  - Configure Python virtual environment (`.venv`).
  - Adapt `ffb_video.py` (YuNet face tracker & sampler) into `src/video_detection/`.
  - Adapt `ffb_metrics.py` (AUC, EER) into `src/forensics/metrics.py`.
- [ ] **Stage 4: Image Detection & Grad-CAM Explainability Module**
  - Implement EfficientNet-B4 model wrapper and inference pipeline.
  - Implement native PyTorch Grad-CAM hook generator with heatmap overlays.
- [ ] **Stage 5: Video Detection & Temporal Aggregation Engine**
  - Implement multi-threaded OpenCV video frame sampler.
  - Implement IoU face tracking across frames.
  - Implement frame scoring, mean pooling, and keyframe saliency selector.
- [ ] **Stage 6: Digital Forensics Engine & PDF Report Generator**
  - Implement SHA-256 cryptographic file hashing and metadata extraction.
  - Build automated PDF generation pipeline using ReportLab.
- [ ] **Stage 7: FastAPI Backend Service**
  - RESTful endpoints for image upload, video upload, analysis, and PDF download.
- [ ] **Stage 8: Frontend Forensic Dashboard & Visualizer**
  - Modern React / Vite UI with drag-and-drop analysis, frame scrubber, and instant PDF download.
- [ ] **Stage 9: Academic Documentation, Evaluation & Project Defense**
