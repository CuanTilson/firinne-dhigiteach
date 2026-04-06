from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "artifacts" / "chapter6_evaluation"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_metrics_rows() -> list[dict]:
    rows: list[dict] = []

    model_a_sources = [
        (
            "Model A baseline",
            ROOT / "artifacts" / "model_a_baseline_gpu" / "exports" / "metrics_summary.json",
            "present_in_repo",
        ),
        (
            "Model A v2.1",
            ROOT / "artifacts" / "model_a_v2_1_gpu" / "exports" / "metrics_summary.json",
            "present_in_repo",
        ),
    ]

    for model_name, path, availability in model_a_sources:
        metrics = _load_json(path)
        for item in metrics:
            dataset = item["dataset"]
            dataset_label = {
                "validation_final": "Validation",
                "test": "GenImage test",
                "external": "Kaggle external",
            }.get(dataset, dataset)
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_label,
                    "accuracy": item["accuracy"],
                    "precision_ai": item["precision_ai"],
                    "recall_ai": item["recall_ai"],
                    "f1_ai": item["f1_ai"],
                    "source_file": str(path.relative_to(ROOT)),
                    "status": availability,
                }
            )

    model_b_path = ROOT / "artifacts" / "model_b_comparison_kaggle_250" / "model_b_comparison.json"
    model_b = _load_json(model_b_path)
    for run in model_b["model_b_runs"]:
        tm = run["test_metrics"]
        rows.append(
            {
                "model": run["run_name"],
                "dataset": "Kaggle-250 feature split (test)",
                "accuracy": tm["accuracy"],
                "precision_ai": tm["precision_ai"],
                "recall_ai": tm["recall_ai"],
                "f1_ai": tm["f1_ai"],
                "source_file": str(model_b_path.relative_to(ROOT)),
                "status": "present_in_repo",
            }
        )

    return rows


def build_confusion_rows() -> list[dict]:
    rows: list[dict] = []

    model_a_conf_sources = [
        (
            "Model A baseline",
            ROOT / "artifacts" / "model_a_baseline_gpu" / "exports" / "confusion_matrices.csv",
        ),
        (
            "Model A v2.1",
            ROOT / "artifacts" / "model_a_v2_1_gpu" / "exports" / "confusion_matrices.csv",
        ),
    ]
    for model_name, path in model_a_conf_sources:
        for item in _load_csv_rows(path):
            dataset_label = {
                "validation_final": "Validation",
                "test": "GenImage test",
                "external": "Kaggle external",
            }.get(item["dataset"], item["dataset"])
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_label,
                    "tn": int(item["tn"]),
                    "fp": int(item["fp"]),
                    "fn": int(item["fn"]),
                    "tp": int(item["tp"]),
                    "source_file": str(path.relative_to(ROOT)),
                    "status": "present_in_repo",
                }
            )

    model_b_path = ROOT / "artifacts" / "model_b_comparison_kaggle_250" / "model_b_comparison.json"
    model_b = _load_json(model_b_path)
    for run in model_b["model_b_runs"]:
        cm = run["test_metrics"]["confusion_matrix"]
        rows.append(
            {
                "model": run["run_name"],
                "dataset": "Kaggle-250 feature split (test)",
                "tn": int(cm["tn"]),
                "fp": int(cm["fp"]),
                "fn": int(cm["fn"]),
                "tp": int(cm["tp"]),
                "source_file": str(model_b_path.relative_to(ROOT)),
                "status": "present_in_repo",
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(metrics_rows: list[dict], confusion_rows: list[dict]) -> None:
    md_path = OUTPUT_DIR / "chapter6_evaluation_summary.md"
    lines = [
        "# Chapter 6 Evaluation Pack",
        "",
        "This pack consolidates frozen evaluation artefacts already present in the repository.",
        "No retraining was performed. Metrics and confusion counts were read from saved outputs.",
        "",
        "## Metrics Summary",
        "",
        "| Model | Dataset | Accuracy | Precision | Recall | F1 | Source | Status |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in metrics_rows:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {float(row['accuracy']):.4f} | "
            f"{float(row['precision_ai']):.4f} | {float(row['recall_ai']):.4f} | "
            f"{float(row['f1_ai']):.4f} | `{row['source_file']}` | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Confusion Matrix Summary",
            "",
            "| Model | Dataset | TN | FP | FN | TP | Source | Status |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in confusion_rows:
        lines.append(
            f"| {row['model']} | {row['dataset']} | {row['tn']} | {row['fp']} | "
            f"{row['fn']} | {row['tp']} | `{row['source_file']}` | {row['status']} |"
        )

    lines.extend(
        [
            "",
            "## Availability Notes",
            "",
            "- Model A baseline metrics and confusion matrices were already present in frozen exports.",
            "- Model A v2.1 metrics and confusion matrices were already present in frozen exports.",
            "- Model B metrics and confusion counts were already present in the saved comparison JSON.",
            "- No additional external predictions were reconstructed.",
            "- ROC-AUC is available separately in the original frozen artefacts, but is not repeated in this Chapter 6 summary pack.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    metrics_rows = build_metrics_rows()
    confusion_rows = build_confusion_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_DIR / "chapter6_metrics_summary.csv", metrics_rows)
    write_csv(OUTPUT_DIR / "chapter6_confusion_summary.csv", confusion_rows)
    (OUTPUT_DIR / "chapter6_metrics_summary.json").write_text(
        json.dumps(metrics_rows, indent=2), encoding="utf-8"
    )
    (OUTPUT_DIR / "chapter6_confusion_summary.json").write_text(
        json.dumps(confusion_rows, indent=2), encoding="utf-8"
    )
    write_markdown(metrics_rows, confusion_rows)
    print(f"[OK] Wrote metrics summary -> {OUTPUT_DIR / 'chapter6_metrics_summary.csv'}")
    print(f"[OK] Wrote confusion summary -> {OUTPUT_DIR / 'chapter6_confusion_summary.csv'}")
    print(f"[OK] Wrote markdown summary -> {OUTPUT_DIR / 'chapter6_evaluation_summary.md'}")


if __name__ == "__main__":
    main()
