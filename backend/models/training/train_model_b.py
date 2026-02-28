from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


NON_FEATURE_COLUMNS = {
    "record_id",
    "filename",
    "created_at",
    "classification",
    "target_label",
}
LABEL_MAP = {"real": 0, "ai": 1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a first-pass logistic Model B on exported forensic feature rows."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/model_b_baseline"))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--exclude-features",
        nargs="*",
        default=[],
        help="Exact feature names to exclude from training.",
    )
    parser.add_argument(
        "--exclude-prefixes",
        nargs="*",
        default=[],
        help="Feature name prefixes to exclude from training.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("target_label") in LABEL_MAP]
        return rows, list(reader.fieldnames or [])


def split_rows(rows: list[dict], train_ratio: float, val_ratio: float, seed: int):
    grouped = {"real": [], "ai": []}
    for row in rows:
        grouped[row["target_label"]].append(row)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for label, label_rows in grouped.items():
        rng.shuffle(label_rows)
        n = len(label_rows)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        splits["train"].extend(label_rows[:n_train])
        splits["val"].extend(label_rows[n_train:n_train + n_val])
        splits["test"].extend(label_rows[n_train + n_val:])
        print(f"[INFO] label={label}: train={n_train}, val={n_val}, test={n - n_train - n_val}")
    return splits


def build_feature_names(
    fieldnames: list[str], exclude_features: list[str], exclude_prefixes: list[str]
) -> list[str]:
    excluded = set(exclude_features)
    feature_names: list[str] = []
    for name in fieldnames:
        if name in NON_FEATURE_COLUMNS or name in excluded:
            continue
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        feature_names.append(name)
    if not feature_names:
        raise ValueError("No features remain after applying exclusions.")
    return feature_names


def build_matrix(rows: list[dict], feature_names: list[str]):
    x = np.array(
        [[float(row.get(name, 0.0) or 0.0) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    y = np.array([LABEL_MAP[row["target_label"]] for row in rows], dtype=np.int64)
    return x, y


def standardize(train_x, other_xs: list[np.ndarray]):
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0] = 1.0
    transformed = [(arr - mean) / std for arr in [train_x, *other_xs]]
    return transformed, mean, std


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, num_workers: int):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total = correct = tp = fp = tn = fn = 0
    running_loss = 0.0
    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device).float()
            logits = model(features).squeeze(1)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            labels_long = labels.long()

            batch_size = labels.size(0)
            total += batch_size
            running_loss += loss.item() * batch_size
            correct += (preds == labels_long).sum().item()
            tp += ((preds == 1) & (labels_long == 1)).sum().item()
            fp += ((preds == 1) & (labels_long == 0)).sum().item()
            tn += ((preds == 0) & (labels_long == 0)).sum().item()
            fn += ((preds == 0) & (labels_long == 1)).sum().item()

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
    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 0.001:
        raise ValueError("train/val/test ratios must sum to 1.0")

    set_seed(args.seed)
    rows, fieldnames = load_rows(args.features_csv)
    if len(rows) < 20:
        raise ValueError("Not enough labeled feature rows to train Model B.")

    splits = split_rows(rows, args.train_ratio, args.val_ratio, args.seed)
    feature_names = build_feature_names(fieldnames, args.exclude_features, args.exclude_prefixes)
    train_x, train_y = build_matrix(splits["train"], feature_names)
    val_x, val_y = build_matrix(splits["val"], feature_names)
    test_x, test_y = build_matrix(splits["test"], feature_names)
    (train_x, val_x, test_x), mean, std = standardize(train_x, [val_x, test_x])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(train_x, train_y, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_x, val_y, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_x, test_y, args.batch_size, False, args.num_workers)

    model = nn.Linear(train_x.shape[1], 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_weights = output_dir / "model_b_best.pt"
    best_val_acc = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_samples = 0
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad(set_to_none=True)
            logits = model(features).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        train_loss = total_loss / total_samples if total_samples else 0.0
        val_metrics = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_metrics": val_metrics})
        if epoch == 1 or epoch % 20 == 0 or epoch == args.epochs:
            print(
                f"[INFO] epoch={epoch} train_loss={train_loss:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} val_f1_ai={val_metrics['f1_ai']:.4f}"
            )
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(model.state_dict(), best_weights)

    model.load_state_dict(torch.load(best_weights, map_location=device))
    test_metrics = evaluate(model, test_loader, device)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_workers": args.num_workers,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "input_features_csv": str(args.features_csv),
        "feature_names": feature_names,
        "excluded_features": args.exclude_features,
        "excluded_prefixes": args.exclude_prefixes,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "history": history,
        "test_metrics": test_metrics,
        "weights_sha256": sha256_file(best_weights),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] Saved best weights -> {best_weights}")
    print(f"[OK] Saved run manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
