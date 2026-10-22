"""Dataset ingestion, validation, and split preparation protocols for image and video deepfake detection.

Multimodal Deepfake Detection and Explainable Digital Forensics System for Images and Videos.
Academic Capstone Project.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

# Supported formats
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# Regex patterns matching FaceForensics++ video naming conventions
RX_FFPP_REAL = re.compile(r"^(\d{3})$")
RX_FFPP_FAKE = re.compile(r"^(\d{3})_(\d{3})$")


@dataclass
class ImageSample:
    """Represents a validated image file in the dataset."""

    filepath: str
    filename: str
    label: int  # 0 = authentic / real, 1 = manipulated / fake
    label_name: str  # "real" or "fake"
    sha256: str
    width: int
    height: int
    format: str
    channels: int
    is_valid: bool
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoSample:
    """Represents a validated video file in the dataset."""

    filepath: str
    filename: str
    label: int  # 0 = authentic / real, 1 = manipulated / fake
    label_name: str  # "real" or "fake"
    source_pair: Optional[str]  # e.g., "071_054" or "071" for FF++
    manipulation_method: str  # "original", "Deepfakes", "Face2Face", etc.
    num_frames: int
    fps: float
    width: int
    height: int
    duration_sec: float
    is_valid: bool
    split_assignment: Optional[str] = None  # "train", "val", or "test"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetSummary:
    """Summary statistics for an ingested dataset."""

    modality: str  # "image" or "video"
    total_files_scanned: int
    valid_samples: int
    invalid_samples: int
    real_count: int
    fake_count: int
    duplicate_count: int
    class_balance_ratio: float  # real / fake ratio
    split_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_file_sha256(filepath: Path | str, chunk_size: int = 65536) -> str:
    """Compute cryptographic SHA-256 hash of a file for digital forensics and deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


# ==============================================================================
# IMAGE DATASET INGESTION & VALIDATION
# ==============================================================================


def validate_image_file(filepath: Path | str, label: int) -> ImageSample:
    """Thoroughly validate image file integrity, metadata, and dimensions without trusting extensions alone."""
    path = Path(filepath)
    label_name = "fake" if label == 1 else "real"

    if not path.is_file():
        return ImageSample(
            filepath=str(path),
            filename=path.name,
            label=label,
            label_name=label_name,
            sha256="",
            width=0,
            height=0,
            format="",
            channels=0,
            is_valid=False,
            error_message="File does not exist or is not a regular file.",
        )

    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return ImageSample(
            filepath=str(path),
            filename=path.name,
            label=label,
            label_name=label_name,
            sha256="",
            width=0,
            height=0,
            format=path.suffix.lower(),
            channels=0,
            is_valid=False,
            error_message=f"Unsupported image extension: {path.suffix}",
        )

    try:
        file_hash = compute_file_sha256(path)
        with Image.open(path) as img:
            # Force verification of file headers and image stream
            img.verify()

        # Reopen to read dimensions and mode after verify() closes the file
        with Image.open(path) as img:
            width, height = img.size
            img_format = img.format or path.suffix.replace(".", "").upper()
            channels = len(img.getbands())

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {width}x{height}")

        return ImageSample(
            filepath=str(path.resolve()),
            filename=path.name,
            label=label,
            label_name=label_name,
            sha256=file_hash,
            width=width,
            height=height,
            format=img_format,
            channels=channels,
            is_valid=True,
            error_message=None,
        )
    except Exception as e:
        return ImageSample(
            filepath=str(path),
            filename=path.name,
            label=label,
            label_name=label_name,
            sha256="",
            width=0,
            height=0,
            format="",
            channels=0,
            is_valid=False,
            error_message=f"Corrupt or unreadable image: {type(e).__name__}: {str(e)}",
        )


def scan_image_directory(
    root_dir: Path | str,
    real_subdirs: Tuple[str, ...] = ("real", "authentic", "0"),
    fake_subdirs: Tuple[str, ...] = ("fake", "manipulated", "synthetic", "1"),
) -> Tuple[List[ImageSample], DatasetSummary]:
    """Recursively scan an image dataset directory, infer labels from folder structure,

    and detect duplicates using SHA-256 hashes.
    """
    root = Path(root_dir)
    samples: List[ImageSample] = []
    seen_hashes: Set[str] = set()
    duplicate_count = 0

    if not root.exists():
        raise FileNotFoundError(f"Image dataset directory not found: {root}")

    for current_root, _, files in os.walk(root):
        folder_name = Path(current_root).name.lower()
        if folder_name in real_subdirs:
            folder_label = 0
        elif folder_name in fake_subdirs:
            folder_label = 1
        else:
            # Search parent folder hierarchy if nested (e.g. train/real/)
            parts = [p.lower() for p in Path(current_root).parts]
            if any(r in parts for r in real_subdirs):
                folder_label = 0
            elif any(f in parts for f in fake_subdirs):
                folder_label = 1
            else:
                continue

        for f in files:
            path = Path(current_root) / f
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                sample = validate_image_file(path, label=folder_label)
                if sample.is_valid:
                    if sample.sha256 in seen_hashes:
                        duplicate_count += 1
                        continue
                    seen_hashes.add(sample.sha256)
                samples.append(sample)

    valid_samples = [s for s in samples if s.is_valid]
    real_count = sum(1 for s in valid_samples if s.label == 0)
    fake_count = sum(1 for s in valid_samples if s.label == 1)
    balance_ratio = float(real_count) / float(fake_count) if fake_count > 0 else 0.0

    summary = DatasetSummary(
        modality="image",
        total_files_scanned=len(samples) + duplicate_count,
        valid_samples=len(valid_samples),
        invalid_samples=len(samples) - len(valid_samples),
        real_count=real_count,
        fake_count=fake_count,
        duplicate_count=duplicate_count,
        class_balance_ratio=round(balance_ratio, 4),
        split_counts={},
    )
    return samples, summary


def create_reproducible_image_split(
    samples: List[ImageSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[ImageSample]]:
    """Partition valid image samples into reproducible, stratified Train, Validation, and Test splits.

    Guarantees no evaluation leakage and equal class balance across all splits.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    valid_samples = [s for s in samples if s.is_valid]
    real_samples = [s for s in valid_samples if s.label == 0]
    fake_samples = [s for s in valid_samples if s.label == 1]

    rng = np.random.default_rng(seed)
    rng.shuffle(real_samples)
    rng.shuffle(fake_samples)

    def _split_list(items: List[ImageSample]) -> Tuple[List[ImageSample], List[ImageSample], List[ImageSample]]:
        n = len(items)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]

    real_train, real_val, real_test = _split_list(real_samples)
    fake_train, fake_val, fake_test = _split_list(fake_samples)

    splits = {
        "train": real_train + fake_train,
        "val": real_val + fake_val,
        "test": real_test + fake_test,
    }

    # Shuffle each split so classes are interspersed
    for split_key in splits:
        rng.shuffle(splits[split_key])

    return splits


# ==============================================================================
# VIDEO DATASET INGESTION & VALIDATION
# ==============================================================================


def parse_video_metadata_from_path(filepath: Path | str) -> Tuple[int, str, Optional[str]]:
    """Extract label, manipulation method, and source-pair ID from path naming conventions (e.g.

    FaceForensics++ or standard folder structures).
    """
    path = Path(filepath)
    stem = path.stem
    parent_folder = path.parent.name.lower()

    # Check for FF++ Real pattern: 3 digits e.g. "071"
    if RX_FFPP_REAL.match(stem):
        return 0, "original", stem

    # Check for FF++ Fake pattern: 3 digits _ 3 digits e.g. "071_054"
    if RX_FFPP_FAKE.match(stem):
        # Determine manipulation method from folder path
        for method in ("deepfakes", "face2face", "faceswap", "neuraltextures", "faceshifter"):
            if method in str(path).lower():
                return 1, method, stem
        return 1, parent_folder, stem

    # Fallback to directory structure inspection
    parts = [p.lower() for p in path.parts]
    if any(r in parts for r in ("real", "authentic", "youtube", "pristine")):
        return 0, "original", stem
    if any(f in parts for f in ("fake", "manipulated", "synthetic", "deepfake")):
        return 1, parent_folder, stem

    # Default unknown
    return 1, parent_folder, stem


def validate_video_file(filepath: Path | str, label: Optional[int] = None) -> VideoSample:
    """Validate video container readability, stream metadata, and dimensions via OpenCV."""
    path = Path(filepath)
    inferred_label, method, source_pair = parse_video_metadata_from_path(path)
    final_label = label if label is not None else inferred_label
    label_name = "fake" if final_label == 1 else "real"

    if not path.is_file():
        return VideoSample(
            filepath=str(path),
            filename=path.name,
            label=final_label,
            label_name=label_name,
            source_pair=source_pair,
            manipulation_method=method,
            num_frames=0,
            fps=0.0,
            width=0,
            height=0,
            duration_sec=0.0,
            is_valid=False,
            error_message="Video file does not exist.",
        )

    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        return VideoSample(
            filepath=str(path),
            filename=path.name,
            label=final_label,
            label_name=label_name,
            source_pair=source_pair,
            manipulation_method=method,
            num_frames=0,
            fps=0.0,
            width=0,
            height=0,
            duration_sec=0.0,
            is_valid=False,
            error_message=f"Unsupported video extension: {path.suffix}",
        )

    cap = cv2.VideoCapture(str(path.resolve()))
    try:
        if not cap.isOpened():
            return VideoSample(
                filepath=str(path),
                filename=path.name,
                label=final_label,
                label_name=label_name,
                source_pair=source_pair,
                manipulation_method=method,
                num_frames=0,
                fps=0.0,
                width=0,
                height=0,
                duration_sec=0.0,
                is_valid=False,
                error_message="OpenCV could not open video stream (corrupt container or missing codec).",
            )

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        # Confirm at least the first frame is decodable
        grabbed = cap.grab()
        if not grabbed:
            return VideoSample(
                filepath=str(path),
                filename=path.name,
                label=final_label,
                label_name=label_name,
                source_pair=source_pair,
                manipulation_method=method,
                num_frames=num_frames,
                fps=fps,
                width=width,
                height=height,
                duration_sec=0.0,
                is_valid=False,
                error_message="Video container opened, but failed to grab first frame stream.",
            )

        duration_sec = float(num_frames) / fps if fps > 0 else 0.0

        return VideoSample(
            filepath=str(path.resolve()),
            filename=path.name,
            label=final_label,
            label_name=label_name,
            source_pair=source_pair,
            manipulation_method=method,
            num_frames=num_frames,
            fps=round(fps, 2),
            width=width,
            height=height,
            duration_sec=round(duration_sec, 2),
            is_valid=True,
            error_message=None,
        )
    except Exception as e:
        return VideoSample(
            filepath=str(path),
            filename=path.name,
            label=final_label,
            label_name=label_name,
            source_pair=source_pair,
            manipulation_method=method,
            num_frames=0,
            fps=0.0,
            width=0,
            height=0,
            duration_sec=0.0,
            is_valid=False,
            error_message=f"Error reading video properties: {type(e).__name__}: {str(e)}",
        )
    finally:
        cap.release()


def assign_video_splits_by_source_pair(
    video_samples: List[VideoSample],
    official_splits: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, List[VideoSample]]:
    """Assign video samples to splits based on source-pair IDs.

    CRITICAL RULE: Videos belonging to the same source pair (original and all derived manipulations)
    MUST NEVER be split across train, validation, or test. Frames from the same video are never separated.
    """
    valid_samples = [v for v in video_samples if v.is_valid]

    # If official FaceForensics++ splits JSON is supplied, use it
    if official_splits:
        split_map: Dict[str, str] = {}
        for split_name, pair_list in official_splits.items():
            for pair in pair_list:
                split_map[pair] = split_name
                # Also map the single constituent video IDs e.g. "071" from "071_054"
                if "_" in pair:
                    a, b = pair.split("_")
                    split_map[a] = split_name
                    split_map[b] = split_name

        result: Dict[str, List[VideoSample]] = {"train": [], "val": [], "test": [], "unassigned": []}
        for s in valid_samples:
            assigned = None
            if s.source_pair and s.source_pair in split_map:
                assigned = split_map[s.source_pair]
            elif s.filename.split(".")[0] in split_map:
                assigned = split_map[s.filename.split(".")[0]]

            if assigned:
                s.split_assignment = assigned
                result[assigned].append(s)
            else:
                s.split_assignment = "unassigned"
                result["unassigned"].append(s)
        return result

    # Dynamic Pair-Grouped Split: Group by source_pair to prevent data leakage
    pair_groups: Dict[str, List[VideoSample]] = {}
    for s in valid_samples:
        key = s.source_pair or s.filename
        pair_groups.setdefault(key, []).append(s)

    unique_pairs = sorted(pair_groups.keys())
    rng = np.random.default_rng(42)
    rng.shuffle(unique_pairs)

    n = len(unique_pairs)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))

    train_pairs = set(unique_pairs[:n_train])
    val_pairs = set(unique_pairs[n_train : n_train + n_val])
    test_pairs = set(unique_pairs[n_train + n_val :])

    result = {"train": [], "val": [], "test": []}
    for pair_key, group in pair_groups.items():
        if pair_key in train_pairs:
            target = "train"
        elif pair_key in val_pairs:
            target = "val"
        else:
            target = "test"

        for s in group:
            s.split_assignment = target
            result[target].append(s)

    return result


def export_dataset_manifest(
    samples: List[ImageSample] | List[VideoSample],
    output_path: Path | str,
) -> None:
    """Save validated dataset metadata manifest to JSON for reproducibility and forensic auditing."""
    records = [s.to_dict() for s in samples]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
