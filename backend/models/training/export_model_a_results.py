from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Model A run_manifest metrics into report-friendly files."
    )
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def derive_rates(confusion: dict) -> dict[str, float]:
    tn = confusion.get("tn", 0)
    fp = confusion.get("fp", 0)
    fn = confusion.get("fn", 0)
    tp = confusion.get("tp", 0)
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (fn + tp) if (fn + tp) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "specificity": specificity,
    }


def metric_row(name: str, metrics: dict | None) -> dict | None:
    if not metrics:
        return None
    confusion = metrics.get("confusion_matrix", {})
    derived = derive_rates(confusion)
    return {
        "dataset": name,
        "loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "precision_ai": metrics.get("precision_ai"),
        "recall_ai": metrics.get("recall_ai"),
        "f1_ai": metrics.get("f1_ai"),
        "false_positive_rate": derived["false_positive_rate"],
        "false_negative_rate": derived["false_negative_rate"],
        "specificity": derived["specificity"],
        "tn": confusion.get("tn"),
        "fp": confusion.get("fp"),
        "fn": confusion.get("fn"),
        "tp": confusion.get("tp"),
        "total_samples": metrics.get("total_samples"),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(path: Path, manifest: dict, rows: list[dict]) -> None:
    best_val = manifest["history"][-1]["val_metrics"] if manifest.get("history") else {}
    external_row = next((row for row in rows if row["dataset"] == "external"), None)
    test_row = next((row for row in rows if row["dataset"] == "test"), None)

    lines = [
        "# Model A Evaluation Summary",
        "",
        "## Run Metadata",
        f"- Device: `{manifest.get('device')}`",
        f"- Seed: `{manifest.get('seed')}`",
        f"- Epochs: `{manifest.get('config', {}).get('epochs')}`",
        f"- Batch size: `{manifest.get('config', {}).get('batch_size')}`",
        f"- Image size: `{manifest.get('config', {}).get('image_size')}`",
        f"- Learning rate: `{manifest.get('config', {}).get('lr')}`",
        f"- Weights SHA-256: `{manifest.get('weights_sha256')}`",
        "",
        "## Key Findings",
    ]

    if test_row:
        lines.extend(
            [
                f"- In-domain GenImage test accuracy: `{test_row['accuracy']:.4f}`",
                f"- In-domain GenImage AI-class F1: `{test_row['f1_ai']:.4f}`",
            ]
        )
    if external_row:
        lines.extend(
            [
                f"- External Kaggle accuracy: `{external_row['accuracy']:.4f}`",
                f"- External Kaggle AI-class F1: `{external_row['f1_ai']:.4f}`",
            ]
        )
    if test_row and external_row:
        accuracy_gap = test_row["accuracy"] - external_row["accuracy"]
        f1_gap = test_row["f1_ai"] - external_row["f1_ai"]
        lines.extend(
            [
                f"- Accuracy gap (in-domain vs external): `{accuracy_gap:.4f}`",
                f"- F1 gap (in-domain vs external): `{f1_gap:.4f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Validation Endpoint",
            f"- Final validation accuracy: `{best_val.get('accuracy', 0.0):.4f}`",
            f"- Final validation AI-class F1: `{best_val.get('f1_ai', 0.0):.4f}`",
            "",
            "## Interpretation",
            "- The baseline learns the curated GenImage subset well.",
            "- External generalization remains weak, indicating substantial dataset shift.",
            "- The result supports the next project phase: better evaluation reporting, broader data coverage, and Model B learned fusion with forensic features.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.run_manifest)
    output_dir = args.output_dir or (args.run_manifest.parent / "exports")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    history = manifest.get("history", [])
    if history:
        rows.append(metric_row("validation_final", history[-1]["val_metrics"]))
    rows.append(metric_row("test", manifest.get("test_metrics")))
    external = metric_row("external", manifest.get("external_metrics"))
    if external is not None:
        rows.append(external)
    rows = [row for row in rows if row is not None]

    write_csv(output_dir / "metrics_summary.csv", rows)
    write_csv(
        output_dir / "confusion_matrices.csv",
        [
            {
                "dataset": row["dataset"],
                "tn": row["tn"],
                "fp": row["fp"],
                "fn": row["fn"],
                "tp": row["tp"],
            }
            for row in rows
        ],
    )
    write_markdown_summary(output_dir / "evaluation_summary.md", manifest, rows)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    print(f"[OK] Exported metrics -> {output_dir / 'metrics_summary.csv'}")
    print(f"[OK] Exported confusion matrices -> {output_dir / 'confusion_matrices.csv'}")
    print(f"[OK] Exported markdown summary -> {output_dir / 'evaluation_summary.md'}")


if __name__ == "__main__":
    main()
