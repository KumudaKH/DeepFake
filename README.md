# Multimodal Deepfake Detection and Explainable Digital Forensics System

[![Python 3.11](https://img.shields.io/badge/python-3.11.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61dafb.svg)](https://reactjs.org/)
[![License: Academic Research](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)](#)

A comprehensive, academic-grade system for detecting image manipulations and synthetic speech using deep learning, accompanied by transparent interpretability heatmaps, acoustic spectrogram analysis, and automated, evidence-backed forensic PDF reports.

Developed as a **Final-Year Computer Science and Engineering Capstone Project**.

---

## 1. Executive Summary & Problem Statement

The rapid evolution of generative deep learning models (Diffusion models, GANs, Neural Voice Clones, and Neural Vocoders) has made high-fidelity synthetic media easily accessible. This presents urgent security challenges:
- **Misinformation and identity fraud** across multimedia platforms.
- **Biometric authentication vulnerabilities** in facial recognition and voice-banking systems.
- **Black-box detection skepticism**: Traditional deep neural networks offer predictions without interpretable rationale, which is insufficient for digital forensics, judicial inquiries, or integrity audits.

This project addresses these challenges by developing a **multimodal detection system** that couples state-of-the-art neural feature extraction with **explainable AI (XAI)** and automated digital forensics reporting. Every forensic assessment provides:
1. Probabilistic classification with calibrated confidence scores.
2. Visual or acoustic explainability artifacts (e.g., Grad-CAM spatial heatmaps, frequency spectrogram anomalies).
3. Plain-English interpretation of detection artifacts.
4. Comprehensive metadata, limitations, and methodological uncertainty disclosures.

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
|  IMAGE FORENSICS      |                                       |   SPEECH FORENSICS    |
|  - Preprocessing      |                                       |  - Preprocessing      |
|  - Face Extraction    |                                       |  - Audio Slicing      |
|  - Deep Classifier    |                                       |  - Acoustic Features  |
|  - Grad-CAM Heatmap   |                                       |  - Deep Classifier    |
|  - Suspicious Regions |                                       |  - Spectrogram / Wave |
+-----------------------+                                       +-----------------------+
          |                                                               |
          +-------------------------------+-------------------------------+
                                          |
+-----------------------------------------------------------------------------------+
|                        FORENSIC REPORT GENERATION ENGINE                          |
|                       ReportLab Automated PDF Generator                           |
|  - File SHA-256 Hash & EXIF Metadata      - Calibrated Confidence Assessment       |
|  - Visual / Acoustic Evidence Overlays    - Methodological Uncertainty & Audit    |
+-----------------------------------------------------------------------------------+
```

### Module A: Image Deepfake Detection
- **Input**: User-uploaded images (JPEG, PNG, WebP).
- **Processing**: Face detection, spatial normalization, and deep feature analysis.
- **Classification**: Binary classification (Likely Authentic vs. Potentially Manipulated).
- **Explainability**: Gradient-weighted Class Activation Mapping (Grad-CAM) to identify salient regions influencing model decisions.
- **Forensic Distinction**: Distinguishes between model saliency explanations and verified bounding-box manipulation locations.

### Module B: Speech Deepfake Detection
- **Input**: User-uploaded audio recordings (WAV, MP3, FLAC, OGG).
- **Processing**: Standardized resampling (16 kHz), silence trimming, and time-frequency transformations (Mel-spectrograms, MFCCs, LFCCs).
- **Classification**: Synthetic/Spoofed Speech vs. Genuine Human Speech detection.
- **Explainability**: Waveform temporal analysis, spectral artifact highlighting, and temporal segment scrutiny where supported.
- **Forensic Interpretation**: Clear narrative explaining acoustic anomalies and spectral inconsistencies.

### Module C: Explainable Digital Forensics & PDF Reporting
- **Automated Report Engine**: Generates professional, publication-quality digital forensics reports using ReportLab.
- **Contents**:
  1. File metadata (Name, Size, MIME type, SHA-256 cryptographic hash, analysis timestamp).
  2. Model specification, version, and architecture.
  3. Calibrated confidence rating with categorical uncertainty categorization.
  4. Embedded visual evidence (Grad-CAM heatmaps, spectrograms, waveforms).
  5. Plain-English forensic summary and interpretation.
  6. Explicit statement of limitations, non-definitive disclaimer, and recommended secondary verification methods.

---

## 3. Repository Directory Structure

```
Deepfake_Forensics_System/
├── README.md               # Project documentation and architectural overview
├── .gitignore              # Git ignore rules for deep learning, data, and models
├── requirements.txt        # Python dependency manifest
├── .env.example            # Environment variable configuration template
├── data/                   # Data directory (kept out of version control)
│   ├── raw/                # Unprocessed source datasets
│   │   ├── images/         # Raw benchmark image datasets (e.g., FaceForensics++)
│   │   └── audio/          # Raw benchmark audio datasets (e.g., ASVspoof)
│   └── processed/          # Preprocessed, aligned, and split data
│       ├── images/         # Normalized face crops and balanced splits
│       └── audio/          # Standardized 16kHz audio slices and feature caches
├── models/                 # Model weights and serialized checkpoints
│   ├── image/              # Trained image detection checkpoints & weights
│   └── audio/              # Trained audio detection checkpoints & weights
├── src/                    # Core Python application source code
│   ├── data/               # Dataset loaders, augmentation, and preprocessing scripts
│   ├── image_detection/    # Model architectures, training, and inference for images
│   ├── audio_detection/    # Feature extraction, architectures, and inference for speech
│   ├── explainability/     # Grad-CAM, saliency mapping, and acoustic visualizers
│   ├── forensics/          # ReportLab PDF report builder and evidence synthesizers
│   └── backend/            # FastAPI service application
│       └── app/
│           ├── api/        # REST route definitions (analyze, report, health)
│           ├── services/   # Business logic (inference dispatch, file handlers)
│           └── schemas/    # Pydantic request and response schemas
├── frontend/               # React / Vite forensic dashboard and reporting interface
├── notebooks/              # Jupyter research notebooks (EDA, experiments, calibration)
├── reports/                # Generated forensic PDF reports output folder
├── docs/                   # Academic project documentation, design specs, review slides
└── tests/                  # Unit and integration test suites
```

---

## 4. Planned Technology Stack

| Layer | Technologies | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11.x | Core machine learning, backend API, and forensics |
| **Deep Learning** | PyTorch, Torchvision, Torchaudio | Neural network modeling, inference, and GPU acceleration |
| **Data & Scientific** | NumPy, Pandas, Scikit-learn, SciPy | Data wrangling, calibration metrics, evaluation metrics |
| **Computer Vision** | OpenCV, Pillow | Image preprocessing, resizing, and colorspace conversions |
| **Audio Processing** | Librosa, SoundFile | Acoustic feature extraction (STFT, Mel-scale, MFCCs) |
| **Visualization** | Matplotlib, Seaborn | Explanatory heatmap generation and spectrogram plotting |
| **Backend API** | FastAPI, Uvicorn, Pydantic | High-performance asynchronous REST API |
| **Forensic Reports** | ReportLab | Automated, cryptographically linked PDF report generation |
| **Frontend UI** | React, Vite | Interactive web portal for file analysis and result inspection |
| **Version Control** | Git | Distributed version control and reproducible tracking |

---

## 5. Development Roadmap & Working Method

Development progresses through disciplined, verifiable academic milestones:

- [x] **Stage 1: Initial Setup & Environment Verification** (Current)
  - Verify Python 3.11 and Git availability.
  - Establish standardized repository structure.
  - Define dependencies, security configuration, and version control rules.
- [ ] **Stage 2: Dataset Selection, Verification & Ingestion Protocols**
  - Evaluate FaceForensics++, ASVspoof, or vetted Kaggle research benchmarks.
  - Verify research licenses, class balance, and test/train leakage prevention.
  - Build reproducible data loaders and preprocessing pipelines.
- [ ] **Stage 3: Image Deepfake Detection & Explainability Module**
  - Implement baseline and transfer-learning architectures (e.g., EfficientNet, ResNet).
  - Integrate Grad-CAM explainability pipeline.
  - Validate temperature scaling / Platt scaling for confidence calibration.
- [ ] **Stage 4: Speech Deepfake Detection & Spectral Analysis Module**
  - Extract acoustic features (Mel-spectrograms / LFCCs).
  - Train genuine vs. synthetic speech classifier (CNN / ResNet / RNN).
  - Generate aligned waveform and spectrogram forensic visualizers.
- [ ] **Stage 5: Digital Forensics Engine & PDF Report Generator**
  - Integrate file integrity hashing (SHA-256) and EXIF extraction.
  - Build automated PDF generation pipeline using ReportLab.
  - Formulate structured narrative explanations and uncertainty boundaries.
- [ ] **Stage 6: FastAPI Backend Integration & Robust File Validation**
  - Implement RESTful endpoints for image, audio, and report generation.
  - Add strict MIME validation, file size limits, and sanitization.
- [ ] **Stage 7: Frontend Forensic Dashboard & Visualizer**
  - Build modern, accessible UI with drag-and-drop analysis.
  - Display interactive explainability overlays and instant PDF download.
- [ ] **Stage 8: Comprehensive Evaluation, Academic Report & Review Defense**
  - Compute ROC-AUC, EER, Precision, Recall, and Confusion Matrices.
  - Compile final academic project documentation and presentation.

---

## 6. Academic Integrity, Limitations & Forensic Disclaimer

1. **Non-Conclusive Nature**: Machine learning predictions represent probabilistic inferences based on training distributions. They do **not** constitute definitive legal or judicial proof of authenticity or manipulation.
2. **Explainability Distinction**: Class activation maps (such as Grad-CAM) highlight features influential to the neural network; they must not be conflated with ground-truth pixel-level manipulation masks unless explicitly trained with spatial supervision.
3. **No Fabrication Policy**: In accordance with academic and scientific standards, this project strictly prohibits the fabrication of experimental metrics, confidence numbers, or forensic artifacts. All reported evaluations are derived from verifiable, reproducible empirical testing.
