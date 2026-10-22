# Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos

**Department:** Department of Computer Science and Engineering  
**Curriculum Requirement:** Final-Year Capstone Project / Major Technical Project  
**Date:** October 2026  
**System Status:** Full Stack Prototype Operational (FastAPI + React/Vite + PyTorch + OpenCV)  

---

## Abstract

With the proliferation of deep generative modeling techniques (including diffusion autoencoders, Generative Adversarial Networks, and neural face swapping algorithms), digital video and photographic evidence can no longer be evaluated through perceptual inspection alone. This capstone engineering project designs, implements, and evaluates an end-to-end multimodal deepfake forensics and interpretability system.

The core computational architecture combines:
1. A fine-tuned **EfficientNet-B4** convolutional neural network with continuous logit output for image and facial crop manipulation classification;
2. An **OpenCV YuNet ONNX** and IoU-based temporal face tracker capable of tracking identity across sampled video frames and pooling facial logits via arithmetic mean aggregation;
3. A gradient-weighted class activation mapping (**Grad-CAM**) explainer targeting terminal convolutional feature layers (`conv_head`, 1,792 dimensions) to provide spatial visual attribution for investigative triage;
4. A cryptographically anchored forensics reporting framework producing **SHA-256** hashed digital examination sheets in JSON and printable HTML formats;
5. A high-performance **React + Vite + Tailwind CSS** digital forensics console wired directly into a **FastAPI** asynchronous REST backend.

---

## 1. System Architecture & Pipeline Formulation

```
[Uploaded Evidence Media (Image or Video)]
                   │
                   ▼
     ┌─────────────────────────────┐
     │ SHA-256 Cryptographic Hash  │  (Zero-fabrication digital integrity)
     └─────────────┬───────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
 [Image Pathway]        [Video Pathway]
       │                       │
       │                Uniform Temporal Sampling (k=16 frames, window=300)
       │                       │
       │                OpenCV YuNet ONNX Facial Crop Extraction
       │                       │
       │                IoU-based Inter-Frame Identity Tracking
       │                       │
       └───────────┬───────────┘
                   │
                   ▼
     ┌─────────────────────────────┐
     │  EfficientNet-B4 Backbone   │  (Input: 224×224×3 RGB, ImageNet normalized)
     │  conv_head (1,792 channels) │
     │  Linear Head (1 logit)      │
     └─────────────┬───────────────┘
                   │
       ┌───────────┴───────────┐
       ▼                       ▼
 [Logit Calibration]    [Grad-CAM Attribution]
  P(fake) = σ(s)         A_k = ReLU(Σ α_k * A^k)
       │                       │
       └───────────┬───────────┘
                   │
                   ▼
 ┌───────────────────────────────────┐
 │ Digital Forensics Examination Log │
 │ (JSON & HTML Downloadable Sheets) │
 └───────────────────────────────────┘
```

---

## 2. Mathematical Formulations

### 2.1 Continuous Binary Logit Projection & Probability Calibration
Rather than predicting uncalibrated multi-class softmax probabilities, the terminal classifier employs a single continuous scalar logit $s \in \mathbb{R}$:
$$s = \mathbf{w}^T \mathbf{f} + b$$
where $\mathbf{f} \in \mathbb{R}^{1792}$ represents the globally pooled feature vector from `conv_head`, $\mathbf{w} \in \mathbb{R}^{1792}$ represents `classifier.weight`, and $b \in \mathbb{R}$ is `classifier.bias`.

The posterior probability that an ingested facial sample represents synthetic manipulation ($y=1$) is given by the standard logistic sigmoid function:
$$P(\text{fake} \mid \mathbf{x}) = \sigma(s) = \frac{1}{1 + e^{-s}}$$

The classification decision rule parameterized by threshold $\tau \in [0, 1]$ (default $\tau = 0.5$) is:
$$\hat{y} = \begin{cases} 1 \ (\text{FAKE}), & \text{if } P(\text{fake}) \ge \tau \\ 0 \ (\text{REAL}), & \text{otherwise} \end{cases}$$

### 2.2 Temporal Video Frame Sampling & Logit Pooling
For a video $V$ comprising $N$ frames, frames are sampled uniformly using bin-centered temporal indexes over a processing window $W = \min(N, 300)$ with frame budget $k = 16$:
$$t_i = \left\lfloor \left(i + \frac{1}{2}\right) \frac{W}{k} \right\rfloor, \quad i \in \{0, 1, \dots, k-1\}$$

For frames containing verified tracked faces, individual frame logits $s_i$ are extracted. Aggregated video-level probability is derived via arithmetic mean pooling over valid logits:
$$\bar{s} = \frac{1}{|M|} \sum_{i \in M} s_i, \quad P_V(\text{fake}) = \sigma(\bar{s})$$
where $M = \{i \mid \text{frame } i \text{ contains a valid face crop}\}$.

The **Peak Anomaly Keyframe** isolating the temporal frame exhibiting highest manipulation evidence is identified by:
$$i^* = \arg\max_{i \in M} s_i$$

### 2.3 Grad-CAM Explainability Formulation
To explain the spatial regions influencing the model's prediction, Gradient-weighted Class Activation Mapping computes the gradient of the continuous logit score $s$ with respect to feature activation maps $A^k \in \mathbb{R}^{U \times V}$ of the final convolutional layer `conv_head` ($U=V=7$):
$$\alpha_k = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial s}{\partial A_{i,j}^k}$$
where $Z = U \times V = 49$.

The spatial attribution heatmap $L_{\text{Grad-CAM}} \in \mathbb{R}^{U \times V}$ is generated via a rectified linear combination of feature maps:
$$L_{\text{Grad-CAM}} = \operatorname{ReLU}\left( \sum_{k=1}^K \alpha_k A^k \right)$$

Applying $\operatorname{ReLU}$ ensures that only feature channels that **positively contribute** to the manipulation score $s$ are visualized. The map is subsequently normalized to $[0.0, 1.0]$, bilinearly upsampled to $224 \times 224$, colorized with a Jet colormap, and blended with the original crop:
$$I_{\text{overlay}} = \alpha \cdot I_{\text{Jet}} + (1 - \alpha) \cdot I_{\text{crop}}$$

---

## 3. Experimental Training Progression (Google Colab T4 GPU)

The EfficientNet-B4 network is fine-tuned on the FaceForensics++ / Deepfake Detection (FF++/DFD) curated facial dataset using AdamW optimizer ($\text{lr} = 10^{-4}$ with cosine annealing) and binary cross-entropy with logits loss.

### Empirical Validation Table
| Training Epoch | Train Loss | Validation Loss | Validation ROC-AUC | Best Checkpoint Status |
| :---: | :---: | :---: | :---: | :---: |
| **Epoch 1** | `0.2567` | `0.0585` | `0.9981` | Initial baseline saved |
| **Epoch 2** | `0.0528` | `0.0207` | **`0.9998`** | **`model_best_fp16.pt` (Saved)** |
| **Epoch 3** | Underway | Underway | Pending | Cosine schedule convergence |

*Key Findings:* Validation ROC-AUC achieved $0.9998$ by Epoch 2 with validation loss decreasing to $0.0207$, confirming discriminative feature convergence across facial artifact boundaries without actor leakage.

---

## 4. Verification & Software Engineering Standards

### 4.1 PyTest Verification Suite
The system is protected by 56 comprehensive automated tests spanning all modular layers:
- `tests/test_backend_api.py` (6 tests): Health endpoints, image inference, Grad-CAM payloads, 400 empty file handling, missing YuNet 503 response, structured JSON report export.
- `tests/test_image_detector.py` (15 tests): ImageNet normalization, FP16-to-FP32 CPU promotion, logit determinism, uninitialized execution prevention, missing weight detection.
- `tests/test_gradcam.py` (18 tests): Hook registration and cleanup, dimension invariance, zero-gradient leakage, numeric stability with constant activations.
- `tests/test_face_tracker.py` (10 tests): Frame sampling geometry, IoU calculations, crop boundary padding, missing YuNet exception handling.
- `tests/test_prepare_data.py` (7 tests): Deduplication, disjoint actor pair splitting, SHA-256 computation.

**Test Execution Result:** `56 passed in 16.19s` (100% pass rate).

### 4.2 Production Frontend Build
- Bundler: Vite v5.4.21
- Modules Transformed: 1,586
- Production JS Bundle: 234.20 kB (gzip: 63.66 kB)
- Production CSS: 25.86 kB (gzip: 5.26 kB)
- Build Status: Zero errors, zero regressions.

---

## 5. Forensic Evidentiary Limitations & Legal Custody Notice

Digital forensic practitioners and evaluators must adhere to the following procedural standards:
1. **Probabilistic Nature:** Neural network outputs represent statistical pattern inference and do not constitute self-authenticating judicial proof under Federal Rule of Evidence 901/902 or equivalent legal standards.
2. **Receptive Field Attribution:** Grad-CAM heatmaps highlight internal neural receptive activations; they must not be conflated with certified physical tampering masks or compression boundary delineations.
3. **Cryptographic Provenance:** Every evidence item processed by this system is anchored by its cryptographic SHA-256 digest, exact native dimensions, temporal frame index, and execution timestamps to maintain evidentiary chain of custody.
