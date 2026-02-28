from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from backend.models.training.train_model_b import LABEL_MAP, load_rows, split_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a comparison table for rule-based, ML-only, and Model B runs."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--model-b-run", type=Path, action="append", required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model_b_comparison"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    return parser.parse_args()


def confusion_counts(preds: list[int], labels: list[int]) -> dict[str, int]:
    tp = fp = tn = fn = 0
    for pred, label in zip(preds, labels):
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}


def metrics_from_predictions(preds: list[int], labels: list[int]) -> dict[str, float | int | dict]:
    total = len(labels)
    cm = confusion_counts(preds, labels)
    tp = cm["tp"]
    fp = cm["fp"]
    tn = cm["tn"]
    fn = cm["fn"]
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision_ai": precision,
        "recall_ai": recall,
        "f1_ai": f1,
        "confusion_matrix": cm,
        "total_samples": total,
    }


def compare_on_test_split(features_csv: Path, seed: int, train_ratio: float, val_ratio: float) -> dict:
    rows, _ = load_rows(features_csv)
    splits = split_rows(rows, train_ratio, val_ratio, seed)
    test_rows = splits["test"]
    labels = [LABEL_MAP[row["target_label"]] for row in test_rows]

    rule_preds = []
    ml_preds = []
    for row in test_rows:
        classification = str(row.get("classification", "")).lower()
        rule_preds.append(1 if "ai" in classification else 0)
        ml_prob = float(row.get("ml_probability", 0.0) or 0.0)
        ml_preds.append(1 if ml_prob >= 0.5 else 0)

    return {
        "rule_based": metrics_from_predictions(rule_preds, labels),
        "ml_probability_0_5": metrics_from_predictions(ml_preds, labels),
        "test_rows": len(test_rows),
    }


def load_model_b_metrics(run_manifest: Path) -> dict:
    data = json.loads(run_manifest.read_text(encoding="utf-8"))
    return {
        "run_name": run_manifest.parent.name,
        "feature_count": len(data.get("feature_names", [])),
        "excluded_prefixes": data.get("excluded_prefixes", []),
        "excluded_features": data.get("excluded_features", []),
        "test_metrics": data["test_metrics"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, baseline_metrics: dict, model_b_runs: list[dict]) -> None:
    lines = [
        "# Model B Comparison",
        "",
        "## Same-Split Baselines",
        "",
        "| System | Accuracy | AI Precision | AI Recall | AI F1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| Rule-based fusion | {baseline_metrics['rule_based']['accuracy']:.3f} | "
            f"{baseline_metrics['rule_based']['precision_ai']:.3f} | "
            f"{baseline_metrics['rule_based']['recall_ai']:.3f} | "
            f"{baseline_metrics['rule_based']['f1_ai']:.3f} |"
        ),
        (
            f"| ML probability >= 0.5 | {baseline_metrics['ml_probability_0_5']['accuracy']:.3f} | "
            f"{baseline_metrics['ml_probability_0_5']['precision_ai']:.3f} | "
            f"{baseline_metrics['ml_probability_0_5']['recall_ai']:.3f} | "
            f"{baseline_metrics['ml_probability_0_5']['f1_ai']:.3f} |"
        ),
        "",
        "## Model B Runs",
        "",
        "| Run | Features | Accuracy | AI F1 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for run in model_b_runs:
        tm = run["test_metrics"]
        lines.append(
            f"| `{run['run_name']}` | {run['feature_count']} | "
            f"{tm['accuracy']:.3f} | {tm['f1_ai']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_metrics = compare_on_test_split(
        args.features_csv,
        args.seed,
        args.train_ratio,
        args.val_ratio,
    )
    model_b_runs = [load_model_b_metrics(path) for path in args.model_b_run]

    comparison_rows = [
        {
            "system": "rule_based_fusion",
            **baseline_metrics["rule_based"],
        },
        {
            "system": "ml_probability_0_5",
            **baseline_metrics["ml_probability_0_5"],
        },
    ]
    for run in model_b_runs:
        comparison_rows.append(
            {
                "system": run["run_name"],
                "feature_count": run["feature_count"],
                "excluded_prefixes": ",".join(run["excluded_prefixes"]),
                "excluded_features": ",".join(run["excluded_features"]),
                **run["test_metrics"],
            }
        )

    write_csv(output_dir / "model_b_comparison.csv", comparison_rows)
    write_markdown(output_dir / "model_b_comparison.md", baseline_metrics, model_b_runs)
    (output_dir / "model_b_comparison.json").write_text(
        json.dumps(
            {
                "input_features_csv": str(args.features_csv),
                "baseline_metrics": baseline_metrics,
                "model_b_runs": model_b_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] Wrote comparison CSV -> {output_dir / 'model_b_comparison.csv'}")
    print(f"[OK] Wrote comparison JSON -> {output_dir / 'model_b_comparison.json'}")
    print(f"[OK] Wrote comparison note -> {output_dir / 'model_b_comparison.md'}")


if __name__ == "__main__":
    main()
