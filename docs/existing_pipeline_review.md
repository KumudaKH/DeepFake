# Stage 3: Existing Project Materials & Pipeline Review Report
## Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos

**Academic Context:** Final-Year Computer Science and Engineering Capstone Project  
**Author / Examiner Scope:** Image and Video Face Forgery Authentication  
**Date:** October 2026  
**Status:** Completed Technical Audit of Supplied Materials & Baseline Alignment

---

## 1. Inventory & Verification of Supplied Materials

A search of all reasonable project locations (`c:\Users\Nethr\Downloads`, `c:\Users\Nethr\OneDrive\Documents\DEEPFAKE`, `c:\Users\Nethr\OneDrive\Documents`, and `c:\Users\Nethr\Desktop`) was performed:

| Material Identifier | File Name | Local Status | Physical Path / Status Note |
| :--- | :--- | :--- | :--- |
| **Material 1** | `ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb` | **FOUND** | Located in `C:\Users\Nethr\Downloads\` (288 KB) and copied to [`notebooks/ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/notebooks/ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb). |
| **Material 2** | `ffpp_dfd_benchmark_tests.zip` | **MISSING** | Not present in Downloads, DEEPFAKE workspace, OneDrive, or Desktop. |
| **Material 3** | `ffpp_dfd_benchmark_artifacts.zip` | **MISSING** | Not present in Downloads, DEEPFAKE workspace, OneDrive, or Desktop. |

### Note on the Missing ZIP Archives:
As determined by examining the notebook code (specifically Cell 0 and Cell 24):
* `ffpp_dfd_benchmark_artifacts.zip` is the **output bundle created by the notebook itself** upon completion of a benchmark run. It packages generated PNG/PDF figures, tables T01–T14, evaluation results (`results/`), and manifest metadata (`run_manifest.json`).
* If the user previously executed this notebook on Kaggle, this archive can be retrieved from the notebook's **Output** tab.
* In the absence of the pre-computed ZIP archives, **no results, metrics, or test outputs will be fabricated**. Instead, the robust internal test suite (`_selftest()`) already embedded in the notebook's library cells provides an executable basis for unit verification.

---

## 2. In-Depth Technical Audit of the Benchmark Notebook

The notebook `ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb` is a complete, publication-level benchmarking pipeline designed for Kaggle with 2× NVIDIA T4 GPUs. Below is an exhaustive breakdown of its components:

### 2.1 Supported Datasets & Access Mechanisms
* **Datasets Supported**:
  1. **FaceForensics++ (FF++)**: Compressed versions `c23` (High Quality) and `c40` (Low Quality / compressed). Covers the four classic manipulation methods (`Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`) plus `FaceShifter`.
  2. **Google DeepFake Detection (DFD)**: 363 real videos and 3,068 deepfake videos created with consenting actors.
* **Verified Kaggle Slugs**:
  The notebook natively attaches datasets via `kagglehub.dataset_download(slug)` without requiring local storage:
  - `abbasaliyev/ffpp-core-c23-videos`
  - `xdxd003/ff-c23`
  - `abbasaliyev/ffpp-core-c40-videos`
  - `abbasaliyev/ffpp-dfd-c23-videos`
  - `sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset`
  - `abbasaliyev/ffpp-dfd-c40-videos`
* **Splits**: Implements the official FaceForensics++ source video pair split (`FFPP_SPLITS`):
  - 360 train pairs (720 base videos).
  - 70 validation pairs (140 base videos).
  - 70 test pairs (140 base videos).
  - Verified with cryptographic checksum: `FFPP_SPLITS_SHA256 = ac5a0539ec24b8cfefb19b5a1f6c0aae9b75239a7c0202abe2e8973cb326b9f1`.

### 2.2 Image / Video Preprocessing and Face Extraction Pipeline
* **Video Decoding (`ffb_video.py` $\rightarrow$ `open_video`, `read_frames`)**:
  - Uses OpenCV `cv2.VideoCapture`.
  - Sets `cv2.CAP_PROP_N_THREADS = 1` to disable intra-decoder multi-threading. This allows CPU workers running across multiple processes to decode videos simultaneously without core thrashing.
  - Efficient sequential decoding: Uses `cap.grab()` to advance past unwanted frames rapidly, and invokes `cap.retrieve()` only for targeted frame indices.
* **Temporal Frame Sampling (`sample_indices`)**:
  - Samples $k = 16$ frames spread uniformly across the first $W = 300$ frames.
  - Takes the center of each bin: $\lfloor (i + 0.5) \cdot W / k \rfloor$.
* **Face Detection (`FaceDetector`)**:
  - Primary Detector: OpenCV ONNX **YuNet** (`cv2.FaceDetectorYN`), with weights downloaded and verified from OpenCV Zoo (`face_detection_yunet_2023mar.onnx`, SHA-256 `8f2383e4dd3cf...`).
  - Fallback: OpenCV Haar Cascade Classifier (`haarcascade_frontalface_default.xml`).
  - Inference optimization: Downsamples frames whose maximum dimension exceeds 640 pixels during detection; rescales bounding boxes back to native resolution.
* **Facial Identity Tracking Across Frames (`choose_faces`)**:
  - Computes Intersection-over-Union (IoU) of detected bounding boxes across consecutive frames.
  - Maintains subject continuity by enforcing $IoU > 0.3$ with the previous frame's face.
  - If a frame fails detection, it adopts the bounding box coordinates of the nearest valid frame (`neighbour` borrowing).
* **Face Cropping & Normalization (`square_crop`)**:
  - Expands the bounding box by a 1.3× scale factor (margin).
  - Crops a square region, converts to RGB, and resizes to $224 \times 224$ pixels.
  - Saves crops directly into shared memory-mapped `.npy` files (`CropSet`).

### 2.3 Model Architectures & Training Procedures
* **Model 1: Fine-Tuned EfficientNet-B4**:
  - Architecture: `efficientnet_b4.ra2_in1k` via `timm`.
  - Output: 1 single continuous logit ($num\_classes = 1$).
  - Training hyperparameters (adapted from DeepfakeBench):
    - Optimizer: Adam (learning rate $2 \times 10^{-4}$, weight decay $5 \times 10^{-4}$).
    - Batch size: 32 (with mixed-precision FP16 enabled).
    - Augmentations: Horizontal flip (0.5), rotation $\pm 15^\circ$ (0.5), Gaussian blur (0.5), brightness-contrast (0.5), and JPEG re-compression (0.5).
    - Class balancing: 16 real frames and 4 manipulated frames per training video with loss inversely weighted to class frequencies.
    - Model selection: Best validation video-level AUC (patience 3 epochs).
* **Model 2: CLIP ViT-L/14 Linear Probe**:
  - Architecture: Frozen `openai/clip-vit-large-patch14`.
  - Extracts 768-dimensional L2-normalized image embeddings for each crop.
  - Fits a class-balanced Logistic Regression classifier whose inverse regularization $C$ is optimized on validation video AUC (UnivFD recipe / Ojha et al., CVPR 2023).

### 2.4 Explainability Method (Grad-CAM)
* **Implementation (`stage_gradcam`)**:
  - Operates directly on the fine-tuned EfficientNet-B4 weights (`model_best_fp16.pt`).
  - Targets the final convolutional head (`model.bn2` or `model.conv_head`).
  - Registers forward hooks to capture feature activations $A \in \mathbb{R}^{C \times H \times W}$ and backward hooks to capture gradients $G = \frac{\partial s}{\partial A}$.
  - Computes channel weights via global average pooling $\alpha_c = \frac{1}{Z}\sum_i \sum_j G_{i,j}$ and generates activation heatmaps $L = \text{ReLU}\left(\sum_c \alpha_c A_c\right)$.
  - Overlays the normalized heatmap over the facial crop for forensic visualization.

### 2.5 Video-Level Prediction Aggregation
* **Frame-to-Video Pooling**:
  - Each extracted frame $j \in \{1, \dots, k\}$ receives a raw neural logit $s_j$.
  - Aggregation is performed via **arithmetic mean pooling** of logits across frames:
    $$\bar{s}_{\text{video}} = \frac{1}{k} \sum_{j=1}^k s_j$$
  - The aggregate manipulation probability is computed via the logistic sigmoid:
    $$P_{\text{video}}(\text{Manipulated}) = \sigma(\bar{s}_{\text{video}}) = \frac{1}{1 + e^{-\bar{s}_{\text{video}}}}$$
* **Forensic Keyframe Selection**:
  - The frame with the maximum anomaly logit ($j^* = \arg\max_j s_j$) is selected as the primary visual evidence frame for Grad-CAM explanation.

### 2.6 Evaluation Metrics Engine (`ffb_metrics.py`)
* Vectorized Mann-Whitney U Area Under the ROC Curve (AUC).
* Equal Error Rate (EER) computed via linear interpolation on the ROC curve.
* Average Precision (AP).
* Cluster Bootstrap: 2,000 resamples by source video pairs for FF++ and actor pairs for DFD to compute rigorous 95% confidence intervals.

---

## 3. Reusability Analysis for Our Project

| Subsystem | Reusable Code from Benchmark | Modification / Adaptation Needed for Full System |
| :--- | :--- | :--- |
| **Face Extraction & Tracking** | `ffb_video.py` (`FaceDetector`, `sample_indices`, `choose_faces`, `square_crop`) | **Direct Reuse**: Encapsulate into [`src/video_detection/face_tracker.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/video_detection) with clean class interfaces for the FastAPI backend. |
| **Model Architecture** | EfficientNet-B4 definition (`timm`) and loading routine | **Direct Reuse**: Wrap into [`src/image_detection/detector.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/image_detection) supporting both single-image evaluation and batched frame inference. |
| **Explainability** | Native PyTorch Grad-CAM hooks (`stage_gradcam`) | **Direct Reuse**: Integrate into [`src/explainability/gradcam.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/explainability) with automated colormap rendering (Jet/Turbo overlay) and base64/PNG export for reports and UI. |
| **Video Aggregation** | Mean logit pooling + Keyframe selection | **Direct Reuse**: Implement in [`src/video_detection/temporal_aggregator.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/video_detection) returning both the video score and the keyframe index. |
| **Evaluation Metrics** | `ffb_metrics.py` (vectorized AUC, EER, ROC) | **Direct Reuse**: Adapt into [`src/forensics/metrics.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/forensics) for project report evaluation and test verification. |
| **Execution Framework** | Notebook multi-GPU worker (`ffb_worker.py`) | **Modify**: The multi-GPU queue is specialized for 2x T4 Kaggle runs; for our web application backend, we need asynchronous Python service handlers (`src/backend/app/services/`). |
| **Digital Forensics Engine** | Tables/CSV exports | **New Implementation**: Build the automated ReportLab PDF generator in [`src/forensics/report_generator.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/forensics) that links metadata, hashes, and Grad-CAM visuals. |
| **Web Application** | *None in notebook* | **New Implementation**: Build FastAPI REST routes and React/Vite dashboard. |

---

## 4. Most Practical Dataset Choices for Our 8–12 Hour Timeline

1. **For Single-Image Face Forgery Detection**:
   * **Dataset:** **140k Real and Fake Faces** (NVIDIA FFHQ authentic human faces + StyleGAN synthetic faces).
   * **Size:** ~1.75 GB compressed / ~2.1 GB uncompressed.
   * **Access:** Open/instant download from Kaggle (`xhlulu/140k-real-and-fake-faces`).
   * **Rationale:** Pre-cropped $256 \times 256$ images; eliminates hours of video decoding and face tracking. Allows immediate training and verification of the EfficientNet classifier and Grad-CAM pipeline within the first 1–2 hours.

2. **For Video Deepfake Detection**:
   * **In Cloud (Google Colab / Kaggle)**: Mount `abbasaliyev/ffpp-core-c23-videos` directly via `kagglehub`. The supplied notebook runs its `quick` mode on this dataset in ~15–20 minutes without consuming local disk space.
   * **In Local Development / App Testing**: A curated evaluation test set of 20–50 canonical paired videos from the official test split (10 pristine YouTube, 10 Deepfakes, 10 Face2Face, 10 FaceSwap, 10 NeuralTextures) to test end-to-end video upload, YuNet face tracking, temporal aggregation, and PDF reporting.

---

## 5. Summary of Status & Verification Checklist

- [x] Located `ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb` in Downloads and integrated into [`notebooks/`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/notebooks).
- [x] Verified that `ffpp_dfd_benchmark_tests.zip` and `ffpp_dfd_benchmark_artifacts.zip` are not present on local storage (accurately reported, not invented).
- [x] Fully audited the notebook's YuNet face detection, IoU tracking, EfficientNet-B4 model, Grad-CAM hooks, and temporal aggregation math.
- [x] Mapped all reusable modules directly into the project architecture.
- [x] Preserved existing project structure, zero files deleted, zero mock results fabricated.
