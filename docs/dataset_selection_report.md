# Stage 2: Dataset Selection, Verification & Benchmark Evaluation Report
## Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos

**Academic Context:** Final-Year Computer Science and Engineering Capstone Project  
**Author / Examiner Scope:** Image and Video Face Forgery Authentication  
**Date:** October 2026  
**Status:** Completed Evaluation & Implementation Alignment

---

## 1. Revised Project Scope

In accordance with the updated project direction, **audio and speech detection have been removed from the scope**. The project focuses exclusively on:
1. **Single-Image Face Forgery Detection**: Classifying authentic vs. synthetic/manipulated facial photographs with visual interpretability (Grad-CAM heatmaps).
2. **Video Deepfake Detection & Temporal Aggregation**: Ingesting video recordings, uniformly sampling temporal frames, tracking facial identities across frames, executing frame-level deep neural inference, and aggregating frame predictions into a video-level forensic verdict.
3. **Explainable Digital Forensics Engine**: Synthesizing file metadata (SHA-256 hash, container data), frame inspection summaries, Grad-CAM saliency overlays, uncertainty boundaries, and producing an automated ReportLab PDF forensic report.

---

## 2. In-Depth Evaluation of Supplied Benchmark Materials

The project was provided with the following materials, which have been inspected directly:
* `ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb` (Copied to `notebooks/ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb` and thoroughly analyzed).
* `ffpp_dfd_benchmark_tests.zip` (Test suite / verification bundle).
* `ffpp_dfd_benchmark_artifacts.zip` (Pre-computed benchmark output archive including metrics, tables T01–T14, and figures).

### Technical Audit of `ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb`

| Pipeline Component | Notebook Implementation | Reusability in our System |
| :--- | :--- | :--- |
| **Supported Benchmark Datasets** | **FaceForensics++ (FF++)** (c23 and c40 compressions) across all 4 classic methods (`Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`) plus `FaceShifter`, and **Google DeepFake Detection (DFD)**. | **Directly compatible**. Uses the official 360 train / 70 validation / 70 test video pair split (`FFPP_SPLITS_SHA256 = ac5a053...`). |
| **Verified Kaggle Slugs** | The notebook natively attaches datasets via `kagglehub`: <br>• `abbasaliyev/ffpp-core-c23-videos`<br>• `xdxd003/ff-c23`<br>• `abbasaliyev/ffpp-dfd-c23-videos`<br>• `sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset` | **Directly usable in Google Colab / Kaggle**. Avoids manual downloading or authorization wait times. |
| **Video Decoding & Frame Extraction** | Defined in `ffb_src/ffb_video.py`. Uses OpenCV `cv2.VideoCapture` with thread contention management. Implements `sample_indices(n_frames, k, window)`: samples $k$ frames uniformly across the first `window` (default 300) frames without bias toward the start or end. | **100% Reusable**. Production-ready, pure NumPy/OpenCV without heavyweight dependencies. |
| **Face Detection & Tracking** | Defined in `ffb_src/ffb_video.py`. Implements `FaceDetector` wrapping OpenCV's high-speed ONNX YuNet detector (`cv2.FaceDetectorYN`) with automatic weight fetching from OpenCV Zoo (`face_detection_yunet_2023mar.onnx`) and Haar fallback. Implements `choose_faces(dets)` using IoU tracking ($IoU > 0.3$) across frames to prevent identity-jumping in multi-person videos. | **100% Reusable**. Solves face localization and tracking cleanly on CPU/GPU. |
| **Model Architectures** | 1. **EfficientNet-B4** (`efficientnet_b4.ra2_in1k` via `timm`): Fine-tuned with single output logit on 224px crops with Adam, lr $2\times 10^{-4}$, weight decay $5\times 10^{-4}$, and augmentations (flip, rotation, blur, brightness-contrast, JPEG compression).<br>2. **CLIP ViT-L/14** (`openai/clip-vit-large-patch14`): Frozen linear probe (Ojha et al., CVPR 2023). | **Core Deep Learning Engine**. The EfficientNet-B4 pipeline can be directly integrated into `src/image_detection/` and `src/video_detection/`. |
| **Explainability Method** | Native PyTorch **Grad-CAM** implemented in `stage_gradcam()` attaching forward and backward hooks to the last convolutional head (`bn2` / `conv_head`) of EfficientNet-B4. | **100% Reusable**. Directly integrated into `src/explainability/gradcam.py` for both single images and video keyframes. |
| **Video Aggregation Method** | Averages the individual frame logit scores across the $k$ sampled frames per video: $\bar{s} = \frac{1}{k} \sum_{i=1}^k s_i$. The final video forgery probability is computed via calibrated logistic sigmoid: $P(\text{Manipulated}) = \sigma(\bar{s})$. | **Standard Forensic Aggregation**. Documented and directly implemented in `src/video_detection/temporal_aggregator.py`. |
| **Evaluation Metrics** | Defined in `ffb_src/ffb_metrics.py`. Fully vectorized computation of Area Under the ROC Curve (AUC), Equal Error Rate (EER), Average Precision (AP), and 2,000-resample cluster bootstrap confidence intervals. | **100% Reusable**. Ideal for computing official evaluation tables for the project report and paper. |

---

## 3. Evaluation of Candidate Datasets for Image & Video Forensics

### 3.1 Single-Image Face Forgery Candidates

| Dataset | Official Source | Access Mode | Size | Image Format | Labels & Structure | Feasibility for 8–12h Prototype |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **140k Real & Fake Faces** *(Recommended)* | [Kaggle Hub](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | **Open / Immediate** (Kaggle API or Web) | **~1.75 GB** (zip) / ~2.1 GB uncompressed | 256×256 standardized PNG | Binary (0 = Real / FFHQ, 1 = Fake / StyleGAN). Pre-split into `train/`, `valid/`, `test/`. | **Optimal**. Zero wait time, pre-cropped faces, directly ingestible by PyTorch `ImageFolder`. Perfect for validating single-image detection and Grad-CAM. |
| **FaceForensics++ Extracted Frames** | Community Kaggle Mirrors | Open on Kaggle | 4 GB to 12 GB | Varied JPEG/PNG crops | Real vs Fake (FF++ methods) | **Moderate**. Requires verifying frame resolution and ensuring no identity overlap across splits. |
| **Celeb-DF v2 Extracted Frames** | Manual frame extraction from Celeb-DF | Approval form required | >30 GB | Varied PNG | Real vs Fake | **Poor**. Download latency and extraction overhead violates timeline. |

### 3.2 Video Face Forgery Candidates

| Dataset | Official Source | Access Mode | Size | Format | Manipulation Methods | Feasibility for 8–12h Prototype |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FaceForensics++ (c23 Core Videos)** *(Recommended)* | [Kaggle: abbasaliyev/ffpp-core-c23-videos](https://www.kaggle.com/datasets/abbasaliyev/ffpp-core-c23-videos) | **Verified Kaggle Mount** via `kagglehub` (or direct download) | ~18 GB (Full set) | MP4 videos (H.264, c23 HQ) | Deepfakes, Face2Face, FaceSwap, NeuralTextures, Pristine YouTube | **Optimal in Colab/Kaggle**. The supplied notebook is pre-configured to mount this exact slug. For local testing, a curated subset of 20–50 paired test videos provides instant verification of the entire video pipeline. |
| **Google DeepFake Detection (DFD)** | [Kaggle: abbasaliyev/ffpp-dfd-c23-videos](https://www.kaggle.com/datasets/abbasaliyev/ffpp-dfd-c23-videos) | **Verified Kaggle Mount** via `kagglehub` | ~15 GB | MP4 videos (c23) | High-quality deepfakes generated by Google/Jigsaw with consenting actors | **High**. Officially supported by the supplied notebook as an independent zero-shot generalization test set. |
| **Celeb-DF v2** | [GitHub: yuezunli/celeb-deepfakeforensics](https://github.com/yuezunli/celeb-deepfakeforensics) | **Restricted**: Requires email authorization form | ~36 GB video | MP4 videos | Real YouTube vs. Celeb-synthesis DeepFakes | **Infeasible for 8–12h**. Access request approval delay and large download footprint. |

---

## 4. Final Dataset & Prototype Strategy Recommendations

### Recommended Image Dataset: **"140k Real and Fake Faces"**
* **Role**: Primary benchmark for single-image face classification and Grad-CAM explainability evaluation.
* **Why**: Instantly accessible via Kaggle (~1.75 GB), balanced 1:1, standardized $256 \times 256$ resolution, zero extraction overhead.
* **Execution**: Allows instant training and validation of the EfficientNet image classifier, establishing baseline accuracy and Grad-CAM visualization within the first 1–2 hours.

### Recommended Video Dataset: **FaceForensics++ (c23) & Google DFD (via Verified Kaggle Mirrors)**
* **Role**: Primary benchmark for video-level face forgery detection, temporal frame sampling, identity tracking, and video-level score aggregation.
* **Why**: Directly supported by the supplied benchmark notebook (`abbasaliyev/ffpp-core-c23-videos` and `abbasaliyev/ffpp-dfd-c23-videos`).
* **Execution**:
  - In **Google Colab / Kaggle**: Mount the verified Kaggle datasets via `kagglehub` or `/kaggle/input` using the supplied notebook's `quick` test or rehearsal mode.
  - In the **Local Application**: A targeted test suite of 20–50 canonical paired videos from the official test split (10 pristine YouTube, 10 Deepfakes, 10 Face2Face, 10 FaceSwap, 10 NeuralTextures) is used to validate end-to-end video upload, YuNet face extraction, temporal aggregation, and PDF forensic reporting.

---

## 5. Temporal Frame Sampling and Prediction Aggregation Specification

For video deepfake analysis, the system implements a mathematically rigorous aggregation pipeline:

1. **Uniform Temporal Window Sampling**:
   For a video with $N$ total frames and a sampling target of $k$ frames (default $k = 16$, window $W = \min(N, 300)$):
   $$\text{indices} = \left\{ \left\lfloor (i + 0.5) \cdot \frac{W}{k} \right\rfloor : i \in \{0, 1, \dots, k-1\} \right\}$$
   This prevents sampling bias towards initial title screens or ending credits.

2. **Facial Tracking & Consistency**:
   For each sampled frame $j \in \{1, \dots, k\}$, the YuNet detector detects all candidate faces $\{b_1, \dots, b_m\}$. The primary face $b^*$ is selected using IoU tracking against the anchor face from previous frames:
   $$b^* = \arg\max_{b} \text{IoU}(b, b_{\text{anchor}}), \quad \text{requiring } \text{IoU} > 0.3$$
   If no detection satisfies the threshold, the spatial coordinates of the nearest valid neighbour frame are preserved.

3. **Frame-Level Neural Scoring**:
   Each cropped face $x_j$ is resized to $224 \times 224$, normalized via ImageNet statistics, and fed to the fine-tuned EfficientNet-B4 to output a continuous logit $s_j \in \mathbb{R}$.

4. **Temporal Score Aggregation**:
   The aggregate video logit $\bar{s}$ is computed via arithmetic mean pooling across valid frames:
   $$\bar{s} = \frac{1}{k} \sum_{j=1}^k s_j$$
   The calibrated video-level manipulation probability $P_{\text{video}}(\text{Manipulated})$ is given by:
   $$P_{\text{video}}(\text{Manipulated}) = \frac{1}{1 + e^{-\bar{s}}}$$

5. **Keyframe Saliency Selection**:
   The frame exhibiting the highest anomaly logit ($j^* = \arg\max_j s_j$) is automatically selected as the **Forensic Keyframe**. Grad-CAM is computed on this keyframe to localize the spatial forgery artifacts for the forensic report.

---

## 6. Anti-Leakage Partitioning Strategy

We enforce the official FaceForensics++ source video pair split (`FFPP_SPLITS`):
* **Train Partition (360 source video pairs / 720 base videos)**: All manipulated variants derived from these 360 pairs are quarantined strictly to training.
* **Validation Partition (70 source video pairs / 140 base videos)**: Used exclusively for early stopping and model selection.
* **Evaluation Test Partition (70 source video pairs / 140 base videos)**: Strictly held out. Zero actor, background, or source video overlap with Train or Val.

---

## 7. Reusable Code Assets Checklist from Supplied Materials

- [x] `ffb_video.py` $\rightarrow$ Adapted into [`src/video_detection/face_tracker.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/video_detection) (YuNet detector, IoU face tracker, frame sampler).
- [x] `ffb_metrics.py` $\rightarrow$ Adapted into [`src/forensics/metrics.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/forensics) (Vectorized AUC, EER, and calibration metrics).
- [x] `stage_gradcam` $\rightarrow$ Adapted into [`src/explainability/gradcam.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/explainability) (EfficientNet gradient-weighted activation mapping).
- [x] `ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb` $\rightarrow$ Staged under [`notebooks/ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/notebooks/ffpp_dfd_face_forgery_benchmark_kaggle_t4x2.ipynb) as the primary GPU training notebook.
