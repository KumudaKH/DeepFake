# Dataset Preparation and Ingestion Guide
## Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos

**Academic Context:** Final-Year Computer Science and Engineering Capstone Project  
**Author / Examiner Scope:** Image and Video Face Forgery Authentication  
**Module:** [`src/data/prepare_data.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/data/prepare_data.py)  
**Date:** October 2026

---

## 1. Overview and Engineering Objectives

This guide establishes the standardized dataset validation, ingestion, and splitting protocols for our multimodal forensics system. Before training deep convolutional networks (such as EfficientNet-B4) or extracting facial activation maps (Grad-CAM), all multimedia files must pass through strict validation checks:
1. **Container & Stream Verification**: Confirm that image headers (JPEG, PNG, WebP) and video containers (MP4, AVI, MOV) are uncorrupted and decodable via Pillow and OpenCV.
2. **Cryptographic Hashing**: Compute SHA-256 hashes for every file to establish a verifiable chain-of-custody and eliminate duplicate samples across splits.
3. **Strict Anti-Leakage Partitioning**: Ensure that related visual material (the same facial subject, source video, or manipulation pair) never crosses between training, validation, or testing partitions.

---

## 2. Expected Dataset Folder Structures

The system supports two standardized layout modes in `data/raw/`:

### 2.1 Single-Image Dataset Layout ("140k Real and Fake Faces")
Recommended benchmark: [`xhlulu/140k-real-and-fake-faces`](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

```
data/raw/images/
├── real_vs_fake/
│   ├── train/
│   │   ├── real/      # Authentic facial portraits (e.g., Flickr-Faces-HQ / FFHQ)
│   │   └── fake/      # Synthetic facial portraits (e.g., StyleGAN)
│   ├── valid/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
```
* **Format:** $256 \times 256$ normalized PNG or JPEG files.
* **Label Ingestion:** Inferred automatically from directory names (`real/` $\rightarrow$ label 0, `fake/` $\rightarrow$ label 1).
* **Deduplication:** Files sharing identical SHA-256 digests are automatically quarantined.

---

### 2.2 Video Deepfake Dataset Layout (FaceForensics++ & Google DFD)
Recommended benchmark: [`abbasaliyev/ffpp-core-c23-videos`](https://www.kaggle.com/datasets/abbasaliyev/ffpp-core-c23-videos) and [`abbasaliyev/ffpp-dfd-c23-videos`](https://www.kaggle.com/datasets/abbasaliyev/ffpp-dfd-c23-videos)

```
data/raw/videos/
├── original_sequences/
│   └── youtube/
│       └── c23/
│           └── videos/
│               ├── 071.mp4             # Authentic YouTube video (Source A)
│               ├── 054.mp4             # Authentic YouTube video (Source B)
│               └── ...
└── manipulated_sequences/
    ├── Deepfakes/
    │   └── c23/
    │       └── videos/
    │           ├── 071_054.mp4         # Deepfake swap: Source A into B
    │           └── ...
    ├── Face2Face/
    │   └── c23/videos/
    ├── FaceSwap/
    │   └── c23/videos/
    └── NeuralTextures/
        └── c23/videos/
```
* **Format:** MP4 (H.264 video streams, c23 high-quality compression).
* **Naming Conventions:**
  - `(\d{3}).mp4` (e.g., `071.mp4`): Pristine authentic video $\rightarrow$ label 0, method = `"original"`.
  - `(\d{3})_(\d{3}).mp4` (e.g., `071_054.mp4`): Manipulated video $\rightarrow$ label 1, method = subfolder name (`"Deepfakes"`, `"Face2Face"`, etc.), source pair = `"071_054"`.

---

## 3. Anti-Leakage Partitioning Protocol

### The Data Leakage Threat in Video Forensics
In video deepfake detection, data leakage causes artificially inflated validation accuracy that collapses on real-world test sets:
* **Frame-Level Leakage:** If individual frames from the same video are distributed across train and test sets, the model learns the specific background, lighting, or person's identity rather than deepfake boundary artifacts.
* **Source-Pair Leakage:** If the pristine video `071.mp4` is placed in the training set and its manipulated counterpart `071_054.mp4` is placed in the test set, the classifier can identify the manipulation simply by recognizing the familiar actor or scene.

### Enforced Protection Mechanism
Our ingestion pipeline strictly enforces **Source-Pair Quarantining**:
1. All video frames extracted from `071.mp4` remain grouped together with the video.
2. The pristine video `071.mp4` and all its derived manipulations (`071_054.mp4` under Deepfakes, Face2Face, FaceSwap, and NeuralTextures) are locked to the **same split**:
   - `FFPP_SPLITS["train"]`: 360 source video pairs (720 base videos + corresponding manipulations).
   - `FFPP_SPLITS["val"]`: 70 source video pairs (140 base videos + corresponding manipulations).
   - `FFPP_SPLITS["test"]`: 70 source video pairs (140 base videos + corresponding manipulations).
3. The evaluation test set has **zero actor, scene, or photographer overlap** with the training or validation sets.

---

## 4. Ingestion & Validation Execution Guide

The validation module is located in [`src/data/prepare_data.py`](file:///c:/Users/Nethr/OneDrive/Documents/DEEPFAKE/src/data/prepare_data.py).

### 4.1 Running the Ingestion Programmatically

```python
from pathlib import Path
from src.data.prepare_data import (
    scan_image_directory,
    validate_video_file,
    assign_video_splits_by_source_pair,
    export_dataset_manifest,
)

# 1. Scan and validate an image directory
image_dir = Path("data/raw/images")
if image_dir.exists():
    samples, summary = scan_image_directory(image_dir)
    print(f"Scanned {summary.valid_samples} valid images (Real: {summary.real_count}, Fake: {summary.fake_count})")
    print(f"Duplicates removed: {summary.duplicate_count}")

    # Export forensic manifest
    export_dataset_manifest(samples, "data/processed/images/image_manifest.json")

# 2. Validate an uploaded video file
video_path = Path("data/raw/videos/sample.mp4")
if video_path.exists():
    video_sample = validate_video_file(video_path)
    if video_sample.is_valid:
        print(f"Video {video_sample.filename}: {video_sample.num_frames} frames, {video_sample.fps} FPS, {video_sample.duration_sec}s")
    else:
        print(f"Invalid video: {video_sample.error_message}")
```

### 4.2 Executing Unit Tests

To verify that image validation, OpenCV video container parsing, deduplication, and anti-leakage splitting operate properly without errors:

```powershell
# Run from the project root using the virtual environment
.\.venv\Scripts\python.exe -m pytest tests/test_prepare_data.py -v
```

Expected output:
```text
tests/test_prepare_data.py::test_validate_image_file_valid PASSED        [ 12%]
tests/test_prepare_data.py::test_validate_image_file_corrupt PASSED      [ 25%]
tests/test_prepare_data.py::test_validate_image_file_unsupported_extension PASSED [ 37%]
tests/test_prepare_data.py::test_scan_image_directory_and_deduplication PASSED [ 50%]
tests/test_prepare_data.py::test_create_reproducible_image_split PASSED  [ 62%]
tests/test_prepare_data.py::test_parse_video_metadata_from_path PASSED   [ 75%]
tests/test_prepare_data.py::test_validate_video_file_synthetic PASSED    [ 87%]
tests/test_prepare_data.py::test_assign_video_splits_by_source_pair_no_leakage PASSED [100%]
============================== 8 passed ==============================
```

---

## 5. Metadata Schema (`manifest.json`)

When an ingested dataset is validated, an audit-ready manifest is generated:

```json
[
  {
    "filepath": "C:\\Users\\Nethr\\OneDrive\\Documents\\DEEPFAKE\\data\\raw\\images\\train\\real\\0001.png",
    "filename": "0001.png",
    "label": 0,
    "label_name": "real",
    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "width": 256,
    "height": 256,
    "format": "PNG",
    "channels": 3,
    "is_valid": true,
    "error_message": null
  }
]
```

This manifest provides the exact evidentiary chain-of-custody that will be referenced by the automated ReportLab forensic PDF generator in subsequent stages.
