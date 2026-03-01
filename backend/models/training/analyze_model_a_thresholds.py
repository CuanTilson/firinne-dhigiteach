from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from backend.models.training.manifest_dataset import ManifestImageDataset
from backend.models.training.train_model_a import build_model, build_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ROC-AUC and threshold calibration outputs for a trained Model A checkpoint."
    )
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="*",
        default=[0.3, 0.5, 0.7],
        help="Probability thresholds to evaluate for the positive AI class.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def roc_auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks = ranks[positives].sum()
    auc = (pos_ranks - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count)
    return float(auc)


def confusion_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, float | int]:
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "accuracy": accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total_samples": int(len(labels)),
    }


def evaluate_scores(
    manifest_path: Path,
    *,
    weights_path: Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    _, eval_transform = build_transforms(image_size)
    dataset = ManifestImageDataset(manifest_path, transform=eval_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model().to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_scores.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_labels), np.concatenate(all_scores)


def threshold_rows(labels: np.ndarray, scores: np.ndarray, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(np.int64)
        row = {"threshold": float(threshold)}
        row.update(confusion_metrics(labels, preds))
        rows.append(row)
    return rows


def band_policy(real_max: float, ai_min: float) -> dict[str, float | str]:
    return {
        "likely_real_max": real_max,
        "likely_ai_min": ai_min,
        "inconclusive_band": f"({real_max:.2f}, {ai_min:.2f})",
    }


def write_markdown(
    output_path: Path,
    run_manifest: dict,
    auc_values: dict[str, float],
    validation_thresholds: list[dict],
    chosen_policy: dict[str, float | str],
) -> None:
    lines = [
        "# Model A Threshold Calibration",
        "",
        f"- Source run manifest: `{run_manifest.get('created_at')}`",
        f"- Weights SHA-256: `{run_manifest.get('weights_sha256')}`",
        "",
        "## ROC-AUC",
        f"- Validation ROC-AUC: `{auc_values['validation']:.4f}`",
        f"- Test ROC-AUC: `{auc_values['test']:.4f}`",
    ]
    if "external" in auc_values:
        lines.append(f"- External ROC-AUC: `{auc_values['external']:.4f}`")

    lines.extend(
        [
            "",
            "## Validation Threshold Sweep",
            "",
            "| Threshold | Accuracy | Precision (AI) | Recall (AI) | F1 (AI) | FPR | FNR |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in validation_thresholds:
        lines.append(
            f"| {row['threshold']:.2f} | {row['accuracy']:.4f} | {row['precision_ai']:.4f} | "
            f"{row['recall_ai']:.4f} | {row['f1_ai']:.4f} | {row['false_positive_rate']:.4f} | "
            f"{row['false_negative_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Draft Classification Policy",
            f"- `Likely Real` if `p(ai) <= {chosen_policy['likely_real_max']:.2f}`",
            f"- `Likely AI` if `p(ai) >= {chosen_policy['likely_ai_min']:.2f}`",
            f"- `Inconclusive` for `{chosen_policy['inconclusive_band']}`",
            "",
            "## Interpretation",
            "- The Week 3 wording remains conservative.",
            "- The 0.30 / 0.70 bands remain acceptable as a draft decision policy for the prototype.",
            "- These thresholds are now backed by a validation ROC-AUC and threshold sweep artifact rather than protocol text alone.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_manifest = load_json(args.run_manifest)
    config = run_manifest.get("config", {})
    manifests = run_manifest.get("manifests", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or (args.run_manifest.parent / "exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_args = {
        "weights_path": args.weights,
        "image_size": int(config.get("image_size", 224)),
        "batch_size": int(args.batch_size or config.get("batch_size", 16)),
        "num_workers": int(
            args.num_workers if args.num_workers is not None else config.get("num_workers", 0)
        ),
        "device": device,
    }

    val_labels, val_scores = evaluate_scores(Path(manifests["val"]), **eval_args)
    test_labels, test_scores = evaluate_scores(Path(manifests["test"]), **eval_args)
    auc_values = {
        "validation": roc_auc_score(val_labels, val_scores),
        "test": roc_auc_score(test_labels, test_scores),
    }
    external_manifest = manifests.get("external")
    if external_manifest and not args.skip_external:
        ext_labels, ext_scores = evaluate_scores(Path(external_manifest), **eval_args)
        auc_values["external"] = roc_auc_score(ext_labels, ext_scores)

    validation_thresholds = threshold_rows(val_labels, val_scores, list(args.thresholds))
    chosen_policy = band_policy(0.30, 0.70)

    (output_dir / "roc_auc_summary.json").write_text(
        json.dumps(auc_values, indent=2), encoding="utf-8"
    )
    (output_dir / "threshold_calibration.json").write_text(
        json.dumps(
            {
                "validation_thresholds": validation_thresholds,
                "chosen_policy": chosen_policy,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(
        output_dir / "threshold_calibration.md",
        run_manifest,
        auc_values,
        validation_thresholds,
        chosen_policy,
    )

    print(f"[OK] Wrote ROC-AUC summary -> {output_dir / 'roc_auc_summary.json'}")
    print(f"[OK] Wrote threshold calibration -> {output_dir / 'threshold_calibration.json'}")
    print(f"[OK] Wrote threshold note -> {output_dir / 'threshold_calibration.md'}")


if __name__ == "__main__":
    main()
