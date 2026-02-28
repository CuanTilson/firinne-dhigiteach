from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from backend.models.training.manifest_dataset import LABEL_MAP, ManifestImageDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Model A on cleaned CSV manifests."
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_a"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(LABEL_MAP))
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    tp = fp = tn = fn = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)

            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            tp += ((preds == 1) & (labels == 1)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = correct / total if total else 0.0

    return {
        "loss": running_loss / total if total else 0.0,
        "accuracy": accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "total_samples": total,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform, eval_transform = build_transforms(args.image_size)

    train_dataset = ManifestImageDataset(args.train_manifest, transform=train_transform)
    val_dataset = ManifestImageDataset(args.val_manifest, transform=eval_transform)
    test_dataset = ManifestImageDataset(args.test_manifest, transform=eval_transform)
    external_dataset = (
        ManifestImageDataset(args.external_manifest, transform=eval_transform)
        if args.external_manifest
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    external_loader = (
        DataLoader(
            external_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        if external_dataset is not None
        else None
    )

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_weights = output_dir / "model_a_best.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples if total_samples else 0.0
        val_metrics = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": val_metrics,
            }
        )
        print(
            f"[INFO] epoch={epoch} train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1_ai={val_metrics['f1_ai']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_weights)

    model.load_state_dict(torch.load(best_weights, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    external_metrics = evaluate(model, external_loader, device) if external_loader else None

    run_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
        },
        "manifests": {
            "train": str(args.train_manifest),
            "val": str(args.val_manifest),
            "test": str(args.test_manifest),
            "external": str(args.external_manifest) if args.external_manifest else None,
        },
        "history": history,
        "test_metrics": test_metrics,
        "external_metrics": external_metrics,
        "weights_sha256": sha256_file(best_weights),
    }

    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(f"[OK] Saved best weights -> {best_weights}")
    print(f"[OK] Saved run manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
