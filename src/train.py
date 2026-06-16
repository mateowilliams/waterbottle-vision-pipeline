"""(Re)training pipeline for the fill-level classifier.

Reproduces the EDA notebook end to end, but as a reproducible, automatable
CLI script:

  1. Builds the dataset from data/<class>/*  (each subfolder = one class).
  2. Stratified train/val/test split (no leakage, fixed seed).
  3. ResNet18 (transfer learning, frozen backbone) + class-imbalance handling
     via class weights in the loss.
  4. Trains, selects the best model by validation accuracy.
  5. Evaluates on test (accuracy, classification report, confusion matrix).
  6. Saves the versioned artifact, the metrics, and the drift baseline
     (reference_stats.json) consumed by the monitoring dashboard.

Usage:
  python -m src.train --epochs 15 --version waterbottle_resnet18_v2
  python -m src.train --reference-only --artifact artifacts/waterbottle_resnet18_v1.pt

Automation: this script is the "retraining" step. It can be triggered by cron
or CI (see README) when monitoring detects drift or enough new labels pile up.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from src.app.monitoring import NUM_BINS, confidence_histogram

ROOT = Path(__file__).resolve().parents[1]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def list_images(data_dir: Path, exclude: Tuple[str, ...] = ("uploads",)) -> pd.DataFrame:
    """Each direct subfolder of data_dir (except `exclude`) is a class."""
    rows = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if class_dir.name in exclude:
            continue
        for p in class_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rows.append({"path": str(p), "label": class_dir.name})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No images found in {data_dir}")
    return df


class ImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_id: Dict[str, int], tfms):
        self.df = df.reset_index(drop=True)
        self.label_to_id = label_to_id
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGBA").convert("RGB")
        return self.tfms(img), self.label_to_id[row["label"]]


def make_transforms(img_size: int):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tfms, eval_tfms


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(num_classes: int) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    return model


def run_epoch(model, dl, loss_fn, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    all_preds, all_true = [], []
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        if train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if train:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total_n += xb.size(0)
        all_preds.append(preds.cpu().numpy())
        all_true.append(yb.cpu().numpy())
    return (
        total_loss / total_n,
        total_correct / total_n,
        np.concatenate(all_true),
        np.concatenate(all_preds),
    )


# --------------------------------------------------------------------------
# Drift baseline
# --------------------------------------------------------------------------
def compute_reference_stats(
    model, eval_tfms, id_to_label: Dict[int, str], df: pd.DataFrame, device, model_version: str
) -> dict:
    """Run the model over `df` and summarize class distribution and confidence.

    Serves as the baseline for the production-monitoring PSI.
    """
    ds = ImageDataset(df, {v: k for k, v in id_to_label.items()}, eval_tfms)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    preds: List[int] = []
    confs: List[float] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in dl:
            probs = torch.softmax(model(xb.to(device)), dim=1)
            conf, pred = probs.max(dim=1)
            preds.extend(pred.cpu().tolist())
            confs.extend(conf.cpu().tolist())

    n = len(preds)
    counts: Dict[str, int] = {}
    for p in preds:
        counts[id_to_label[p]] = counts.get(id_to_label[p], 0) + 1
    class_distribution = {k: round(v / n, 6) for k, v in counts.items()}

    return {
        "model_version": model_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_reference": n,
        "class_distribution": class_distribution,
        "confidence_mean": round(sum(confs) / n, 6) if n else None,
        "confidence_hist": confidence_histogram(confs, NUM_BINS),
        "num_bins": NUM_BINS,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="(Re)train the water-bottle classifier")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--version", default=None, help="Version name (default: auto-increment)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--val-size", type=float, default=0.1)
    ap.add_argument("--test-size", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true", help="Disable class-imbalance handling")
    ap.add_argument("--reference-only", action="store_true", help="Only regenerate reference_stats.json")
    ap.add_argument("--artifact", default=None, help="Artifact to use with --reference-only")
    return ap.parse_args()


def next_version(artifacts_dir: Path) -> str:
    existing = list(artifacts_dir.glob("waterbottle_resnet18_v*.pt"))
    nums = []
    for p in existing:
        tail = p.stem.split("_v")[-1]
        if tail.isdigit():
            nums.append(int(tail))
    return f"waterbottle_resnet18_v{(max(nums) + 1) if nums else 1}"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    set_seed(args.seed)
    _, eval_tfms = make_transforms(args.img_size)

    # ---- Reference-only mode (no retraining) ----
    if args.reference_only:
        artifact_path = Path(args.artifact) if args.artifact else (artifacts_dir / "waterbottle_resnet18_v1.pt")
        ckpt = torch.load(str(artifact_path), map_location="cpu")
        id_to_label = {int(v): str(k) for k, v in ckpt["label_to_id"].items()}
        model = build_model(int(ckpt["num_classes"]))
        model.load_state_dict(ckpt["state_dict"])
        version = str(ckpt.get("model_version") or artifact_path.stem)
        df = list_images(data_dir)
        ref = compute_reference_stats(model, eval_tfms, id_to_label, df, device, version)
        out = artifacts_dir / "reference_stats.json"
        out.write_text(json.dumps(ref, indent=2), encoding="utf-8")
        print(f"[reference-only] baseline saved to {out} (n={ref['n_reference']})")
        return

    # ---- Full training ----
    version = args.version or next_version(artifacts_dir)
    df = list_images(data_dir)
    print(f"Dataset: {len(df)} images")
    print(df["label"].value_counts().to_string())

    labels = sorted(df["label"].unique())
    label_to_id = {lbl: i for i, lbl in enumerate(labels)}
    id_to_label = {i: lbl for lbl, i in label_to_id.items()}
    num_classes = len(labels)

    # Stratified train / val / test split.
    rel_val = args.val_size / (args.val_size + args.test_size)
    train_df, temp_df = train_test_split(
        df, test_size=args.val_size + args.test_size, random_state=args.seed, stratify=df["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - rel_val), random_state=args.seed, stratify=temp_df["label"]
    )
    print(f"Splits -> train {len(train_df)} | val {len(val_df)} | test {len(test_df)}")

    train_tfms, _ = make_transforms(args.img_size)
    train_dl = DataLoader(ImageDataset(train_df, label_to_id, train_tfms), batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(ImageDataset(val_df, label_to_id, eval_tfms), batch_size=args.batch_size)
    test_dl = DataLoader(ImageDataset(test_df, label_to_id, eval_tfms), batch_size=args.batch_size)

    model = build_model(num_classes).to(device)

    # Class-imbalance handling: class weights inversely proportional to frequency.
    if args.no_class_weights:
        weight = None
        print("Class weights: disabled")
    else:
        counts = train_df["label"].map(label_to_id).value_counts().sort_index()
        freqs = np.array([counts.get(i, 0) for i in range(num_classes)], dtype="float64")
        w = freqs.sum() / (num_classes * np.maximum(freqs, 1))
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        print("Class weights:", {id_to_label[i]: round(float(w[i]), 3) for i in range(num_classes)})

    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

    best_val_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_dl, loss_fn, optimizer, device, train=True)
        va_loss, va_acc, _, _ = run_epoch(model, val_dl, loss_fn, optimizer, device, train=False)
        print(f"Epoch {epoch}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_acc > best_val_acc:
            best_val_acc, best_state = va_acc, copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)  # best model by validation
    print(f"Best val acc: {best_val_acc:.4f}")

    # Test evaluation.
    te_loss, te_acc, y_true, y_pred = run_epoch(model, test_dl, loss_fn, optimizer, device, train=False)
    target_names = [id_to_label[i] for i in range(num_classes)]
    report = classification_report(y_true, y_pred, target_names=target_names, digits=3, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print(f"\nTEST loss {te_loss:.4f} acc {te_acc:.4f}\n")
    print("Confusion matrix (rows=true, cols=pred):\n", cm, "\n")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=3, zero_division=0))

    # Save artifact (in the format the API expects).
    artifact = {
        "model_name": "resnet18",
        "model_version": version,
        "num_classes": num_classes,
        "label_to_id": label_to_id,
        "img_size": args.img_size,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
        "state_dict": model.state_dict(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    artifact_path = artifacts_dir / f"{version}.pt"
    torch.save(artifact, artifact_path)
    print("Artifact saved:", artifact_path)

    # Save metrics.
    metrics = {
        "model_version": version,
        "test_accuracy": round(float(te_acc), 4),
        "best_val_accuracy": round(float(best_val_acc), 4),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": target_names,
        "splits": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "trained_at": artifact["trained_at"],
    }
    metrics_path = artifacts_dir / f"metrics_{version}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Metrics saved:", metrics_path)

    # Drift baseline over train+val (data seen during training).
    ref_df = pd.concat([train_df, val_df], ignore_index=True)
    ref = compute_reference_stats(model, eval_tfms, id_to_label, ref_df, device, version)
    ref_path = artifacts_dir / "reference_stats.json"
    ref_path.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print("Drift baseline saved:", ref_path)


if __name__ == "__main__":
    main()
