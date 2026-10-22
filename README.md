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

## 5. Development Roadmap & Completed Milestones

- [x] **Stage 1: Initial Setup & Environment Verification** (Completed)
- [x] **Stage 2: Dataset Selection, Verification & Benchmark Audit** (Completed)
  - Scope refined to Image and Video Deepfake Forensics.
  - Thorough audit of benchmark notebook (`ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb`).
  - Selection of 140k Real & Fake Faces (image) and FaceForensics++ c23 / Google DFD (video).
  - Formalization of YuNet face tracking, uniform frame sampling, and temporal aggregation.
- [x] **Stage 3: Core Utility Library Integration & Anti-Leakage Data Splitting** (Completed)
  - Configured Python virtual environment (`.venv`) and pinned dependencies.
  - Adapted YuNet face tracker and uniform frame sampler into `src/video_detection/`.
  - Anti-leakage video pair splitting and dataset validation suite.
- [x] **Stage 4: Image Detection & Grad-CAM Explainability Module** (Completed)
  - Implemented EfficientNet-B4 inference engine with calibrated logit outputs.
  - Implemented PyTorch Grad-CAM hook generator with colorized spatial overlays.
  - Comprehensive unit test suite covering numeric bounds, determinism, and layer hooks.
- [x] **Stage 5: Video Detection & Temporal Aggregation Engine** (Completed)
  - Implemented uniform temporal frame sampling ($k = 16$).
  - Implemented YuNet face detector with IoU identity tracking across consecutive frames.
  - Implemented frame-level anomaly scoring, temporal mean pooling, and keyframe saliency extraction.
- [x] **Stage 6: Digital Forensics Engine & PDF/HTML Report Generator** (Completed)
  - SHA-256 cryptographic file hashing and EXIF container metadata extraction.
  - Calibrated uncertainty thresholds and forensic methodology disclosures.
  - Automated report generation with embedded Grad-CAM overlays and temporal trajectory charts.
- [x] **Stage 7: FastAPI Backend Service** (Completed)
  - RESTful endpoints (`/api/detect/image`, `/api/detect/video`, `/api/report`, `/api/health`).
  - Dynamic model checkpoint discovery with zero-downtime weight binding.
  - Full request validation and error handling.
- [x] **Stage 8: Modern React / Vite Forensic Dashboard** (Completed)
  - Sleek, dark-mode cybersecurity UI built with React, Vite, and TailwindCSS.
  - Real-time drag-and-drop file upload with live progress tracking.
  - Interactive Grad-CAM heatmap viewer, frame scrubber, and forensic report exporter.
- [ ] **Stage 9: Model Training on GPU & Final Defense Preparation** (Next Step)
  - Run the Kaggle / Colab training notebook (`notebooks/ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb`) on GPU.
  - Export `model_best_fp16.pt` and place it in `models/image/`.

---

## 6. Quick Start & Setup Guide

### Prerequisites
- **Python**: 3.11.x (recommended)
- **Node.js**: 18+ and `npm`
- **Git**

### Installation

1. **Clone the repository:**
   ```bash
   git clone <REPO_URL>
   cd DEEPFAKE
   ```

2. **Backend Setup (Python):**
   ```bash
   # Create virtual environment
   python -m venv .venv

   # Activate virtual environment
   # On Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # On Windows (cmd):
   .\.venv\Scripts\activate.bat
   # On Linux/macOS:
   source .venv/bin/activate

   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Frontend Setup (Node.js / React):**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

---

## 7. Running the Application

### One-Click Launch (Windows)
- **Backend**: Double-click `run_backend.bat` (starts FastAPI at `http://127.0.0.1:8000`)
- **Frontend**: Double-click `run_frontend.bat` (starts Vite dev server at `http://localhost:5173`)

### Manual Launch

**Terminal 1 — FastAPI Backend:**
```bash
.\.venv\Scripts\python -m uvicorn src.backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
*Health Check*: Open [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health) in your browser.

**Terminal 2 — React Frontend:**
```bash
cd frontend
npm run dev
```
*Web Dashboard*: Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 8. Running the Test Suite

Execute the complete automated test suite (69 unit & integration tests) covering image detection, video tracking, Grad-CAM hooks, and FastAPI REST endpoints:

```bash
.\.venv\Scripts\pytest -v
```

---

## 9. Checkpoint Deployment & Model Training

The system comes pre-configured with **YuNet ONNX face detection weights** (`models/yunet/face_detection_yunet_2023mar.onnx`) and operates in **Calibrated Demo Mode** out of the box.

To deploy fully trained weights:
1. Run the training notebook `notebooks/ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb` in Kaggle (T4x2 GPU) or Google Colab.
2. Download the resulting checkpoint file `model_best_fp16.pt`.
3. Place it in `models/image/`:
   ```
   models/image/model_best_fp16.pt
   ```
4. The backend automatically discovers and loads the trained weights without requiring a server restart! See [`docs/checkpoint_deployment_guide.md`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/docs/checkpoint_deployment_guide.md) for full details.

---

## 10. How to Continue & Extend This Work

- **Add New Detectors**: Extend `src/image_detection/detector.py` or `src/video_detection/face_tracker.py` to support other backbones (e.g., Vision Transformers, ConvNeXt, MesoNet).
- **Audio Deepfake Detection**: Add audio feature extraction (spectrograms, MFCCs) under `src/audio_detection/` to build a full audiovisual detector.
- **Explainability Enhancements**: Extend `src/explainability/gradcam.py` with Score-CAM, Eigen-CAM, or frequency-domain artifact visualization.
- **Frontend Visualizations**: Modify `frontend/src/pages/` to add custom forensic inspection tools, frequency spectrum graphs, or multi-subject tracking timelines.
