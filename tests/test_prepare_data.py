"""Unit tests for dataset ingestion, validation, and split preparation protocols.

Tests image and video validation functions using small, isolated temporary files.
"""

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.data.prepare_data import (
    ImageSample,
    VideoSample,
    assign_video_splits_by_source_pair,
    compute_file_sha256,
    create_reproducible_image_split,
    parse_video_metadata_from_path,
    scan_image_directory,
    validate_image_file,
    validate_video_file,
)


@pytest.fixture
def temp_workspace():
    """Create a temporary directory for test fixtures and clean up after test execution."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_validate_image_file_valid(temp_workspace):
    """Test validation of a genuine, properly formatted image file."""
    img_path = temp_workspace / "sample_real.png"
    # Create a small 64x64 RGB image
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:30, 10:30] = [255, 128, 64]
    Image.fromarray(arr).save(img_path, format="PNG")

    sample = validate_image_file(img_path, label=0)

    assert sample.is_valid is True
    assert sample.width == 64
    assert sample.height == 64
    assert sample.format == "PNG"
    assert sample.channels == 3
    assert sample.label == 0
    assert sample.label_name == "real"
    assert len(sample.sha256) == 64
    assert sample.error_message is None


def test_validate_image_file_corrupt(temp_workspace):
    """Test validation handles corrupt/truncated image files gracefully without crashing."""
    corrupt_path = temp_workspace / "corrupt.png"
    with open(corrupt_path, "wb") as f:
        f.write(b"NOT_A_VALID_PNG_HEADER_RANDOM_GARBAGE_BYTES")

    sample = validate_image_file(corrupt_path, label=1)

    assert sample.is_valid is False
    assert sample.width == 0
    assert sample.height == 0
    assert "Corrupt or unreadable" in str(sample.error_message)


def test_validate_image_file_unsupported_extension(temp_workspace):
    """Test validation rejects unsupported file extensions."""
    txt_path = temp_workspace / "data.txt"
    txt_path.write_text("hello world")

    sample = validate_image_file(txt_path, label=0)
    assert sample.is_valid is False
    assert "Unsupported image extension" in str(sample.error_message)


def test_scan_image_directory_and_deduplication(temp_workspace):
    """Test recursive image directory scanning and SHA-256 deduplication."""
    real_dir = temp_workspace / "train" / "real"
    fake_dir = temp_workspace / "train" / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)

    # Create 2 unique real images
    arr1 = np.full((32, 32, 3), 50, dtype=np.uint8)
    arr2 = np.full((32, 32, 3), 100, dtype=np.uint8)
    Image.fromarray(arr1).save(real_dir / "real1.png")
    Image.fromarray(arr2).save(real_dir / "real2.png")

    # Create 1 fake image and 1 duplicate of fake1
    arr3 = np.full((32, 32, 3), 150, dtype=np.uint8)
    Image.fromarray(arr3).save(fake_dir / "fake1.png")
    Image.fromarray(arr3).save(fake_dir / "fake1_duplicate.png")

    samples, summary = scan_image_directory(temp_workspace)

    assert summary.valid_samples == 3
    assert summary.real_count == 2
    assert summary.fake_count == 1
    assert summary.duplicate_count == 1
    assert summary.class_balance_ratio == 2.0


def test_create_reproducible_image_split():
    """Test deterministic stratified image dataset partitioning."""
    samples = []
    # 20 real samples
    for i in range(20):
        samples.append(
            ImageSample(
                filepath=f"path/real_{i}.png",
                filename=f"real_{i}.png",
                label=0,
                label_name="real",
                sha256=f"hash_real_{i}",
                width=64,
                height=64,
                format="PNG",
                channels=3,
                is_valid=True,
            )
        )
    # 20 fake samples
    for i in range(20):
        samples.append(
            ImageSample(
                filepath=f"path/fake_{i}.png",
                filename=f"fake_{i}.png",
                label=1,
                label_name="fake",
                sha256=f"hash_fake_{i}",
                width=64,
                height=64,
                format="PNG",
                channels=3,
                is_valid=True,
            )
        )

    splits = create_reproducible_image_split(samples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42)

    assert len(splits["train"]) == 28  # 14 real, 14 fake
    assert len(splits["val"]) == 6  # 3 real, 3 fake
    assert len(splits["test"]) == 6  # 3 real, 3 fake

    # Verify identical reproducibility with same seed
    splits_repeat = create_reproducible_image_split(samples, seed=42)
    assert [s.filename for s in splits["train"]] == [s.filename for s in splits_repeat["train"]]


def test_parse_video_metadata_from_path():
    """Test label and source-pair parsing from FaceForensics++ paths."""
    # FF++ pristine video
    label, method, pair = parse_video_metadata_from_path("data/raw/videos/original_sequences/youtube/c23/videos/071.mp4")
    assert label == 0
    assert method == "original"
    assert pair == "071"

    # FF++ manipulated video
    label, method, pair = parse_video_metadata_from_path("data/raw/videos/manipulated_sequences/Deepfakes/c23/videos/071_054.mp4")
    assert label == 1
    assert method == "deepfakes"
    assert pair == "071_054"


def test_validate_video_file_synthetic(temp_workspace):
    """Test video validation using a small synthetic MP4 video generated via OpenCV."""
    vid_path = temp_workspace / "test_synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (64, 64))

    # Write 5 frames of dummy content
    for _ in range(5):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[20:40, 20:40] = 200
        out.write(frame)
    out.release()

    sample = validate_video_file(vid_path, label=0)

    assert sample.is_valid is True
    assert sample.width == 64
    assert sample.height == 64
    assert sample.fps == 10.0
    assert sample.num_frames == 5
    assert sample.duration_sec == 0.5
    assert sample.error_message is None


def test_assign_video_splits_by_source_pair_no_leakage():
    """Test that videos sharing the same source pair are NEVER separated across splits (zero leakage)."""
    samples = [
        VideoSample(
            filepath="vid1.mp4",
            filename="071.mp4",
            label=0,
            label_name="real",
            source_pair="071_054",
            manipulation_method="original",
            num_frames=100,
            fps=25.0,
            width=224,
            height=224,
            duration_sec=4.0,
            is_valid=True,
        ),
        VideoSample(
            filepath="vid2.mp4",
            filename="071_054.mp4",
            label=1,
            label_name="fake",
            source_pair="071_054",
            manipulation_method="Deepfakes",
            num_frames=100,
            fps=25.0,
            width=224,
            height=224,
            duration_sec=4.0,
            is_valid=True,
        ),
        VideoSample(
            filepath="vid3.mp4",
            filename="087.mp4",
            label=0,
            label_name="real",
            source_pair="087_081",
            manipulation_method="original",
            num_frames=100,
            fps=25.0,
            width=224,
            height=224,
            duration_sec=4.0,
            is_valid=True,
        ),
    ]

    official_splits = {
        "train": ["071_054"],
        "val": ["087_081"],
        "test": [],
    }

    result = assign_video_splits_by_source_pair(samples, official_splits=official_splits)

    # 071.mp4 (real) and 071_054.mp4 (fake) share pair 071_054 and MUST BOTH be in train
    train_filenames = [s.filename for s in result["train"]]
    assert "071.mp4" in train_filenames
    assert "071_054.mp4" in train_filenames

    # 087.mp4 must be in val
    val_filenames = [s.filename for s in result["val"]]
    assert "087.mp4" in val_filenames
