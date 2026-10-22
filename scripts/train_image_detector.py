"""Train the app-compatible EfficientNet-B4 detector on paired face-frame data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.image_detection.detector import DEFAULT_MEAN, DEFAULT_STD, SUPPORTED_INPUT_TRANSFORM

IMAGE_SIZE = 380
FRAMES_PER_SOURCE_VIDEO = 20
ARCHITECTURE = "efficientnet_b4"
SEED = 42


class FaceFrameDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], transform: Any) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, label


def load_samples(dataset_root: Path) -> dict[str, list[tuple[Path, int, int]]]:
    by_class: dict[str, list[tuple[Path, int, int]]] = {}
    for class_name, label in (("real", 0), ("fake", 1)):
        directory = dataset_root / class_name
        if not directory.is_dir():
            raise FileNotFoundError(f"Expected class directory: {directory}")
        samples = []
        for path in sorted(directory.glob("*.jpg")):
            if not path.stem.isdigit():
                raise ValueError(f"Expected numeric frame filename, got {path.name}")
            frame_id = int(path.stem)
            with Image.open(path) as image:
                image.verify()
            samples.append((path, label, frame_id))
        if not samples:
            raise ValueError(f"No JPG images found in {directory}")
        by_class[class_name] = samples

    real_ids = {frame_id for _, _, frame_id in by_class["real"]}
    fake_ids = {frame_id for _, _, frame_id in by_class["fake"]}
    if real_ids != fake_ids:
        raise ValueError("Real and fake folders must contain the same paired numeric frame IDs.")
    return by_class


def grouped_split(
    by_class: dict[str, list[tuple[Path, int, int]]],
) -> tuple[dict[str, list[tuple[Path, int]]], dict[str, Any]]:
    frame_ids = {frame_id for _, _, frame_id in by_class["real"]}
    group_ids = sorted((frame_id - 1) // FRAMES_PER_SOURCE_VIDEO for frame_id in frame_ids)
    group_ids = sorted(set(group_ids))
    shuffled_groups = list(group_ids)
    random.Random(SEED).shuffle(shuffled_groups)

    test_count = max(1, round(len(group_ids) * 0.15))
    validation_count = max(1, round(len(group_ids) * 0.15))
    test_groups = set(shuffled_groups[:test_count])
    validation_groups = set(shuffled_groups[test_count : test_count + validation_count])
    train_groups = set(shuffled_groups[test_count + validation_count :])
    if not train_groups:
        raise ValueError("Not enough source-video groups to make train/validation/test splits.")

    samples: dict[str, list[tuple[Path, int]]] = {"train": [], "valid": [], "test": []}
    groups = {"train": train_groups, "valid": validation_groups, "test": test_groups}
    for class_samples in by_class.values():
        for path, label, frame_id in class_samples:
            group_id = (frame_id - 1) // FRAMES_PER_SOURCE_VIDEO
            split = next(name for name, split_groups in groups.items() if group_id in split_groups)
            samples[split].append((path, label))

    split_hashes: dict[str, set[str]] = {}
    for split, split_samples in samples.items():
        digests = set()
        for path, _ in split_samples:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in digests:
                raise ValueError(f"Exact duplicate image exists within {split}: {path.name}")
            digests.add(digest)
        split_hashes[split] = digests
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = split_hashes[left] & split_hashes[right]
        if overlap:
            raise ValueError(f"Exact duplicate image content crosses {left}/{right} splits ({len(overlap)} files).")

    audit = {
        "split_method": "paired numeric frame IDs grouped in consecutive 20-frame source-video blocks",
        "seed": SEED,
        "frames_per_source_video_assumption": FRAMES_PER_SOURCE_VIDEO,
        "groups_per_split": {name: len(value) for name, value in groups.items()},
        "images_per_split": {name: len(value) for name, value in samples.items()},
        "exact_content_duplicates_across_splits": 0,
    }
    return samples, audit


def atomic_save(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(handle)
    try:
        torch.save(payload, temporary_name)
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.remove(temporary_name)


def save_best_checkpoint(
    model: nn.Module,
    destination: Path,
    epoch: int,
    validation_auc: float,
    args: argparse.Namespace,
    split_audit: dict[str, Any],
) -> None:
    state_dict = {
        key: value.detach().cpu().half() if value.is_floating_point() else value.detach().cpu()
        for key, value in model.state_dict().items()
    }
    metadata = {
        "format_version": 1,
        "checkpoint_role": "best_inference",
        "state_dict_format": "torch_state_dict_tensor_mapping",
        "task": "deepfake_binary_classification",
        "model_architecture": ARCHITECTURE,
        "model_config": {
            "num_classes": 1,
            "pretrained_initialization": True,
            "optimizer": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "dataset_source": "Josin2000/DEEPFAKE-DETECTION Dataset/face",
            "split_audit": split_audit,
        },
        "input_size": IMAGE_SIZE,
        "input_transform": SUPPORTED_INPUT_TRANSFORM,
        "mean": list(DEFAULT_MEAN),
        "std": list(DEFAULT_STD),
        "labels": {"real": 0, "fake": 1},
        "dataset": "Josin2000/DEEPFAKE-DETECTION Dataset/face",
        "best_epoch": epoch,
        "validation_metric": {"name": "roc_auc", "value": validation_auc},
    }
    atomic_save(
        {
            "state_dict": state_dict,
            "metadata": metadata,
            "epoch": epoch,
            "best_epoch": epoch,
            "best_val_auc": validation_auc,
        },
        destination,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    with torch.inference_mode():
        for images, batch_labels in tqdm(loader, desc="Held-out test", leave=False):
            logits = model(images.to(device)).reshape(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.tolist())

    predictions = [int(score >= 0.5) for score in scores]
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_fake": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall_fake": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1_fake": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix_real_fake": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "decision_threshold": 0.5,
        "label_order": ["real", "fake"],
        "sample_count": len(labels),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Folder containing real/ and fake/.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/image/model_best_fp16.pt"))
    parser.add_argument("--metrics-output", type=Path, default=Path("models/image/training_evaluation.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)

    by_class = load_samples(args.dataset_root)
    samples, split_audit = grouped_split(by_class)
    print("Device:", device)
    print("Split audit:", json.dumps(split_audit, sort_keys=True))

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )
    train_loader = DataLoader(
        FaceFrameDataset(samples["train"], train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        FaceFrameDataset(samples["valid"], eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        FaceFrameDataset(samples["test"], eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    if len(set(label for _, label in samples["train"])) != 2:
        raise ValueError("Training split must contain both real and fake images.")

    model = timm.create_model(ARCHITECTURE, pretrained=True, num_classes=1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    if args.checkpoint.is_file():
        backup = args.checkpoint.with_name(
            args.checkpoint.name
            + ".previous_"
            + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        shutil.copy2(args.checkpoint, backup)
        print("Preserved existing checkpoint:", backup)

    best_auc = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        count = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images = images.to(device)
            labels = labels.float().to(device).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        model.eval()
        validation_labels: list[int] = []
        validation_scores: list[float] = []
        with torch.inference_mode():
            for images, labels in validation_loader:
                logits = model(images.to(device)).reshape(-1)
                validation_scores.extend(torch.sigmoid(logits).cpu().tolist())
                validation_labels.extend(labels.tolist())
        validation_auc = float(roc_auc_score(validation_labels, validation_scores))
        train_loss = running_loss / count
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "validation_roc_auc": validation_auc}
        )
        print(
            f"Epoch {epoch}/{args.epochs}: train_loss={train_loss:.4f}, "
            f"validation_roc_auc={validation_auc:.4f}"
        )

        if validation_auc > best_auc:
            best_auc = validation_auc
            epochs_without_improvement = 0
            save_best_checkpoint(
                model, args.checkpoint, epoch, best_auc, args, split_audit
            )
            print("Saved best checkpoint immediately:", args.checkpoint)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping: validation ROC-AUC did not improve.")
                break

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    metrics = evaluate(model, test_loader, device)
    result = {
        "dataset_source": "Josin2000/DEEPFAKE-DETECTION Dataset/face",
        "dataset_root": str(args.dataset_root.resolve()),
        "model_architecture": ARCHITECTURE,
        "input_size": IMAGE_SIZE,
        "labels": {"real": 0, "fake": 1},
        "split_audit": split_audit,
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint["best_epoch"],
        "best_validation_roc_auc": checkpoint["best_val_auc"],
        "training_history": history,
        "test_metrics": metrics,
        "caveats": [
            "The source dataset preparation extracts 20 frames per video; this split groups consecutive numeric IDs in blocks of 20.",
            "The repository does not provide original video identifiers or subject identities, so cross-video identity leakage cannot be ruled out.",
            "This small source dataset evaluation does not establish generalization to other deepfake methods or real-world media.",
        ],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Held-out test metrics:", json.dumps(metrics, sort_keys=True))
    print("Evaluation saved:", args.metrics_output)


if __name__ == "__main__":
    main()
