# Grad-CAM Explainability & Digital Forensics Guide
## Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos

**Academic Context:** Final-Year Computer Science and Engineering Capstone Project  
**Author / Examiner Scope:** Explainable AI (XAI) for Facial Forgery Authentication  
**Module:** [`src/explainability/gradcam.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/explainability/gradcam.py)  
**Date:** October 2026

**Current verification status:** The trained checkpoint is unavailable in this workspace. The guide describes the Grad-CAM implementation; attribution with the trained checkpoint cannot be verified until compatible weights are recovered or trained.

---

## 1. Executive Summary

In digital forensics and academic research, deep learning detectors are frequently criticized as opaque "black boxes." A binary verdict (e.g., *94% Fake*) is insufficient for courtroom scrutiny, forensic audits, or human-in-the-loop verification. 

The **Grad-CAM (Gradient-weighted Class Activation Mapping)** module provides spatial explainability for the **EfficientNet-B4** face forgery detector without introducing any external plotting dependencies (strictly Matplotlib-free, utilizing high-performance OpenCV and NumPy routines). It identifies the specific facial regions—such as blending boundaries, unnatural skin texture smoothing, or ocular distortions—that influenced the neural network's forgery verdict.

**Interpretation limit:** Grad-CAM highlights coarse regions that influenced the loaded model's score. It does not establish which exact pixels were altered, prove that any region was manipulated, or independently validate the classification.

```
+--------------------------------------------------------------------------------------------------+
|                                    GRAD-CAM PROCESSING PIPELINE                                  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +-----------------------+           Forward Pass            +------------------------------+   |
|   |  RGB Facial Crop      | --------------------------------> | EfficientNet-B4 Backbone     |   |
|   |  (H x W x 3)          |                                   | Target Layer: `conv_head`    |   |
|   +-----------------------+                                   +------------------------------+   |
|               |                                                               |                  |
|               |                                                Forward Hook: Activations A^k     |
|               |                                                Shape: (1, 1792, 7, 7)            |
|               |                                                               |                  |
|               |                                                               v                  |
|               |                                               +------------------------------+   |
|               |                                               | Binary Logit Score s in R    |   |
|               |                                               +------------------------------+   |
|               |                                                               |                  |
|               |                       Backward Pass                           | Backpropagate    |
|               |                       dL / dA^k                               v target logit     |
|               |                       Backward Hook: Gradients        +----------------------+   |
|               |                                                       | Global Avg Pooling   |   |
|               |                                                       | Channel Weights a_k  |   |
|               |                                                       +----------------------+   |
|               |                                                               |                  |
|               |                                                       Linear Combination         |
|               |                                                       + ReLU Filter              |
|               |                                                               |                  |
|               |                                                               v                  |
|               |                                                       Raw CAM Heatmap (7 x 7)    |
|               |                                                               |                  |
|               |                                                       Safe Normalization [0, 1]  |
|               |                                                       (Zero-spread protected)    |
|               |                                                               |                  |
|               v                                                               v                  |
|   +-----------------------+    cv2.applyColorMap(COLORMAP_JET)        +----------------------+   |
|   | Original Image Preserved | <--- cv2.addWeighted(alpha=0.4) <--- | Resized Heatmap      |   |
|   | (Defensive Copy)      |                                           | (H x W) float32      |   |
|   +-----------------------+                                           +----------------------+   |
|               |                                                               |                  |
|               +-------------------------------+-------------------------------+                  |
|                                               |                                                  |
|                                               v                                                  |
|                               +-------------------------------+                                  |
|                               |   GradCAMResult Dataclass     |                                  |
|                               | - heatmap: (H, W) float32     |                                  |
|                               | - overlay_rgb: (H, W, 3)      |                                  |
|                               | - original_rgb: (H, W, 3)     |                                  |
|                               | - probability & logit metadata|                                  |
|                               | - forensic disclaimer         |                                  |
|                               +-------------------------------+                                  |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Mathematical Formulation & Architecture Integration

### 2.1 Grad-CAM Formulation for Single-Logit Binary Forgery Classification

Standard multi-class Grad-CAM computes gradients with respect to class score $y^c$. In our binary forgery classification pipeline, the network outputs a single continuous logit $s \in \mathbb{R}$ before sigmoid calibration:

$$P(\text{fake}) = \sigma(s) = \frac{1}{1 + e^{-s}}$$

Because $\sigma(s)$ is strictly monotonic, maximizing $s$ directly maximizes the predicted probability of facial manipulation.

1. **Neuron Importance Weights ($\alpha_k$):**  
   The importance weight $\alpha_k$ for channel $k$ in target convolutional feature map $A^k$ is computed by global average pooling the gradients of logit $s$:
   $$\alpha_k = \frac{1}{Z} \sum_{i=1}^{U} \sum_{j=1}^{V} \frac{\partial s}{\partial A_{i,j}^k}$$
   where $Z = U \times V$ represents the spatial dimensions ($7 \times 7 = 49$) of the target feature map.

2. **Weighted Saliency Combination:**  
   The raw spatial class activation map is obtained by taking the linear combination of forward activation maps $A^k$, weighted by their corresponding importance $\alpha_k$:
   $$L_{\text{Grad-CAM}} = \text{ReLU}\left( \sum_{k=1}^{K} \alpha_k A^k \right)$$

3. **Role of the Rectified Linear Unit (ReLU):**  
   The $\text{ReLU}$ operation retains positive contributions to the selected model logit. This is a visualization of model influence, not evidence that a highlighted region is manipulated.

---

## 3. Selected Model Layer: `model.conv_head`

In convolutional neural networks, earlier layers capture low-level visual primitives (edges, corners, color gradients), while deeper layers encode complex semantic concepts and spatial context.

### 3.1 EfficientNet-B4 Target Layer Hierarchy

Inspection of the notebook-compatible `efficientnet_b4` model reveals the terminal feature extraction hierarchy:

| Layer Identifier | Module Type | Output Channels | Spatial Resolution (at $380 \times 380$ input) | Selection Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **`model.conv_head`** | `nn.Conv2d(448, 1792, kernel_size=(1, 1))` | **1,792** | **$7 \times 7$** | **Primary Choice**: Final convolutional projection containing richest semantic manipulation abstractions before pooling. |
| `model.bn2` | `nn.BatchNormAct2d` | 1,792 | $7 \times 7$ | Secondary fallback: Post-convolution normalized activations. |
| `model.blocks[-1]` | `InvertedResidual` (MBConv) | 448 | $7 \times 7$ | Tertiary fallback: Pre-head inverted bottleneck block. |

`conv_head` provides the optimal balance between high-level forensic artifact awareness (such as boundary inconsistencies from face swapping or frequency anomalies from GAN generators) and spatial localization.

---

## 4. Matplotlib-Free Visual Generation Pipeline

Due to Windows Application Control restrictions on specific native C-extension DLLs (`_c_internal_utils.pyd`), Matplotlib cannot be safely imported in this execution environment. 

[`src/explainability/gradcam.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/explainability/gradcam.py) implements a pure OpenCV (`cv2`) and NumPy visual generation pipeline that delivers superior execution speed, zero memory leakage, and native RGB alignment.

### 4.1 Safe Heatmap Normalization

Raw Grad-CAM activations vary widely in scale depending on model confidence and gradient magnitudes. Standard min-max normalization:

$$\text{CAM}_{\text{norm}} = \frac{\text{CAM} - \min(\text{CAM})}{\max(\text{CAM}) - \min(\text{CAM})}$$

fails catastrophically when feature activations or gradients are constant (e.g. uniform synthetic inputs or completely zeroed gradient paths), yielding $\frac{0}{0} = \text{NaN}$.

**Implemented Safeguard:**
```python
spread = cam_max - cam_min
if spread > 1e-8:
    normalized_cam = (cam_np - cam_min) / spread
else:
    # Constant or near-zero activations safely resolve to uniform zeros
    normalized_cam = np.zeros_like(cam_np, dtype=np.float32)
```

### 4.2 Interpolation and Color Mapping

1. **Spatial Interpolation:**  
   The $7 \times 7$ normalized activation grid is bilinearly upsampled to match the exact input dimensions $(H, W)$ using `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`.
2. **Pseudocolor Rendering:**  
   The float32 heatmap is scaled to `uint8` $[0, 255]$ and colored using OpenCV's standard forensic colormap:
   ```python
   heatmap_uint8 = np.uint8(255 * heatmap_resized)
   colored_heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
   colored_heatmap_rgb = cv2.cvtColor(colored_heatmap_bgr, cv2.COLOR_BGR2RGB)
   ```
3. **Alpha Blending:**  
   The colored heatmap is blended over the original RGB image crop:
   ```python
   overlay_rgb = cv2.addWeighted(image_rgb, 1.0 - alpha, colored_heatmap_rgb, alpha, 0)
   ```
4. **Defensive Image Preservation:**  
   The original RGB crop is defensively copied (`image_rgb.copy()`) and returned alongside the raw normalized float32 heatmap and the blended overlay as distinct arrays.

---

## 5. Hook Lifecycle & Gradient Safety

PyTorch forward and backward hooks retain references in internal module dictionaries (`_forward_hooks`, `_backward_hooks`). If not meticulously deregistered, repeated calls cause memory bloat and unintended backpropagation side effects.

### Guaranteed Cleanup Guarantee

`GradCAMExplainer.generate_cam` guarantees hook removal using Python's `try ... finally` semantics:

```python
fwd_handle = self.target_layer.register_forward_hook(forward_hook)
bwd_handle = self.target_layer.register_full_backward_hook(backward_hook)

try:
    with torch.enable_grad():
        output = self.model(input_tensor)
        output.view(-1)[0].backward()
    ...
finally:
    if fwd_handle is not None:
        fwd_handle.remove()
    if bwd_handle is not None:
        bwd_handle.remove()
    self.model.zero_grad()
```

Even if an out-of-memory error, invalid dimension, or unexpected exception occurs during tensor operations, both hooks are deleted and model gradients are reset to zero before the exception propagates.

---

## 6. Checkpoint Safety & Scientific Integrity

### 6.1 Requirement of Genuine Deepfake Checkpoints

> [!IMPORTANT]
> **ImageNet-only pretrained weights DO NOT provide meaningful deepfake explanations.**

1. **Feature Space Divergence:**  
   ImageNet weights are trained to distinguish 1,000 biological and object categories (e.g., golden retriever vs. acoustic guitar). Saliency maps from ImageNet weights highlight dominant natural features (eyes, nose, mouth contours) simply because they define facial morphology.
2. **Forensic Interpretation Validity:**  
   A model must be trained on authentic manipulated datasets (FaceForensics++, Celeb-DF, DFD) to learn the high-frequency residual artifacts, affine warping seams, and blending boundaries specific to deepfake synthesis.
3. **Strict Checkpoint Auditing:**  
   The detector records `checkpoint_type` in both the prediction and the Grad-CAM result (`"deepfake_trained"` vs `"uninitialized_mock"`). If uninitialized weights are used in development testing, `is_provisional_threshold=True` and mock metadata are explicitly recorded.

---

## 7. Limitations & Evidentiary Interpretation

> [!CAUTION]
> **Grad-CAM heatmaps highlight regions influencing neural classification; they DO NOT constitute ground-truth manipulation masks or legal proof of forgery.**

Forensic practitioners and academic examiners must consider the following technical limitations:

| Limitation | Technical Reason | Forensic Implication |
| :--- | :--- | :--- |
| **Coarse Spatial Resolution** | The final convolutional layer operates at a coarse spatial grid. Upsampling to the $380 \times 380$ input size produces smooth heatmaps. | Cannot resolve exact pixel-level manipulation boundaries or fine hair-strand blending artifacts. |
| **Sensitivity to Compression** | JPEG quantization noise and video codec blockiness (H.264/H.265 at c40) alter convolutional activation patterns. | Heavily compressed genuine faces may show spurious heatmaps around block boundaries. |
| **Non-Exclusive Saliency** | Grad-CAM reveals which areas the network attended to, not why that area was generated synthetically. | An active region near the mouth might reflect synthesis distortion, facial hair asymmetry, or an unusual shadow. |
| **Admissibility Standard** | Under federal rules of evidence (e.g. Daubert standard in the US or ISO/IEC 27037 digital evidence guidelines), neural heatmaps alone do not satisfy evidentiary repeatability. | Saliency maps must serve as an investigative lead to guide microscopic or frequency-domain verification, never sole legal proof. |

---

## 8. Usage Example & Integration

```python
import cv2
from src.image_detection.detector import DeepfakeDetector
from src.explainability.gradcam import GradCAMExplainer

# 1. Initialize detector with verified trained weights
detector = DeepfakeDetector(
    model_name="efficientnet_b4",
    checkpoint_path="models/image/model_best_fp16.pt",
    device="cpu",
    input_size=380,
)

# 2. Attach Grad-CAM explainer
explainer = GradCAMExplainer(detector)

# 3. Load target RGB facial crop
img_bgr = cv2.imread("data/raw/images/suspect_face.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 4. Generate spatial explainability result
result = explainer.generate_cam(img_rgb, alpha=0.4)

# 5. Inspect structured metadata
print(f"Target Layer: {result.target_layer_name}")
print(f"Logit: {result.raw_logit:.4f} | Probability: {result.probability:.4f} [{result.predicted_label.upper()}]")
print(f"Disclaimer: {result.forensic_disclaimer}")

# 6. Save forensic overlay using OpenCV (no Matplotlib needed)
overlay_bgr = cv2.cvtColor(result.overlay_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite("reports/figures/gradcam_overlay.png", overlay_bgr)
```

---

## 9. Verification Summary

The complete module test suite in [`tests/test_gradcam.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/tests/test_gradcam.py) verifies:
- [x] Dimension preservation across non-square arbitrary inputs.
- [x] Unmutated preservation of original input image arrays.
- [x] Numeric range compliance ($[0.0, 1.0]$ float32 heatmaps, $[0, 255]$ uint8 overlays).
- [x] Zero-spread safe normalization without NaNs or Infs.
- [x] Complete hook and gradient cleanup on both successful execution and simulated hardware errors.
- [x] CPU execution compatibility without CUDA dependencies.
- [x] Automatic target layer discovery (`conv_head`) and fallback hierarchy.
- [x] Rejection of non-convolutional architectures with clear exceptions.
- [x] Inclusion of statutory forensic evidentiary disclaimers.
