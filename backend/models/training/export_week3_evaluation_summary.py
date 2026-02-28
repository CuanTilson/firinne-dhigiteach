from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a consolidated Week 3 evaluation summary for Model A and Model B."
    )
    parser.add_argument(
        "--model-a-run",
        type=Path,
        default=Path("artifacts/model_a_baseline_gpu/run_manifest.json"),
    )
    parser.add_argument(
        "--model-b-comparison",
        type=Path,
        default=Path("artifacts/model_b_comparison_kaggle_250/model_b_comparison.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/week3_evaluation_summary"),
    )
    return parser.parse_args()


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


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_a = json.loads(args.model_a_run.read_text(encoding="utf-8"))
    model_b_comparison = json.loads(args.model_b_comparison.read_text(encoding="utf-8"))

    model_a_rows = [
        {
            "system": "Model A (GenImage test)",
            "dataset_context": "In-domain GenImage",
            "accuracy": model_a["test_metrics"]["accuracy"],
            "precision_ai": model_a["test_metrics"]["precision_ai"],
            "recall_ai": model_a["test_metrics"]["recall_ai"],
            "f1_ai": model_a["test_metrics"]["f1_ai"],
            "notes": "Self-trained image classifier baseline",
        },
        {
            "system": "Model A (Kaggle external)",
            "dataset_context": "External Kaggle",
            "accuracy": model_a["external_metrics"]["accuracy"],
            "precision_ai": model_a["external_metrics"]["precision_ai"],
            "recall_ai": model_a["external_metrics"]["recall_ai"],
            "f1_ai": model_a["external_metrics"]["f1_ai"],
            "notes": "External generalization check",
        },
    ]

    baseline_metrics = model_b_comparison["baseline_metrics"]
    model_b_rows = [
        {
            "system": "Rule-based fusion",
            "dataset_context": "Kaggle-based feature split (same-split baseline)",
            "accuracy": baseline_metrics["rule_based"]["accuracy"],
            "precision_ai": baseline_metrics["rule_based"]["precision_ai"],
            "recall_ai": baseline_metrics["rule_based"]["recall_ai"],
            "f1_ai": baseline_metrics["rule_based"]["f1_ai"],
            "notes": "Current handcrafted final classification on the sampled feature dataset",
        },
        {
            "system": "ML probability >= 0.5",
            "dataset_context": "Kaggle-based feature split (same-split baseline)",
            "accuracy": baseline_metrics["ml_probability_0_5"]["accuracy"],
            "precision_ai": baseline_metrics["ml_probability_0_5"]["precision_ai"],
            "recall_ai": baseline_metrics["ml_probability_0_5"]["recall_ai"],
            "f1_ai": baseline_metrics["ml_probability_0_5"]["f1_ai"],
            "notes": "Model A probability threshold applied inside the sampled feature dataset",
        },
    ]

    for run in model_b_comparison["model_b_runs"]:
        tm = run["test_metrics"]
        model_b_rows.append(
            {
                "system": run["run_name"],
                "dataset_context": "Kaggle-based feature split (same-split Model B)",
                "accuracy": tm["accuracy"],
                "precision_ai": tm["precision_ai"],
                "recall_ai": tm["recall_ai"],
                "f1_ai": tm["f1_ai"],
                "notes": (
                    f"feature_count={run['feature_count']}; "
                    f"excluded_prefixes={','.join(run.get('excluded_prefixes', [])) or '-'}; "
                    f"excluded_features={','.join(run.get('excluded_features', [])) or '-'}"
                ),
            }
        )

    comparison_rows = model_a_rows + model_b_rows
    write_csv(output_dir / "week3_evaluation_table.csv", comparison_rows)

    best_model_b = max(model_b_comparison["model_b_runs"], key=lambda item: item["test_metrics"]["f1_ai"])
    markdown_lines = [
        "# Week 3 Evaluation Summary",
        "",
        "## Frozen Week 3 Findings",
        "",
        "- Model A performs strongly in-domain on GenImage but weakly on external Kaggle.",
        "- The first GenImage-based Model B result was rejected as leaky due to format shortcuts.",
        "- A harder Kaggle-based Model B experiment was built and audited with no perfect single-feature separator.",
        f"- The current best Model B run is `{best_model_b['run_name']}` with accuracy `{format_metric(best_model_b['test_metrics']['accuracy'])}` and AI F1 `{format_metric(best_model_b['test_metrics']['f1_ai'])}`.",
        "",
        "## Consolidated Evaluation Table",
        "",
        "| System | Dataset context | Accuracy | AI Precision | AI Recall | AI F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]

    for row in comparison_rows:
        markdown_lines.append(
            f"| {row['system']} | {row['dataset_context']} | "
            f"{format_metric(row['accuracy'])} | {format_metric(row['precision_ai'])} | "
            f"{format_metric(row['recall_ai'])} | {format_metric(row['f1_ai'])} |"
        )

    markdown_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "These results should not be treated as one direct apples-to-apples benchmark because the Model A external result and the Model B same-split feature result come from different evaluation setups.",
            "",
            "However, the combined Week 3 evidence supports three defensible conclusions:",
            "- Model A alone is not robust enough across datasets.",
            "- Rule-based fusion and a naive ML-probability threshold both underperform on the harder Kaggle-based feature split.",
            "- The learned fusion stage adds measurable value once leakage is controlled.",
            "",
            "## Recommended Week 3 Close",
            "",
            "- Freeze `artifacts/model_a_baseline_gpu/` as the official Model A baseline.",
            "- Freeze `artifacts/model_b_dataset_kaggle_250/`, `artifacts/model_b_audit_kaggle_250/`, and `artifacts/model_b_comparison_kaggle_250/` as the official Model B evidence pack.",
            "- Use this summary as the basis for the evaluation/results subsection in the final report.",
        ]
    )

    (output_dir / "week3_evaluation_summary.md").write_text(
        "\n".join(markdown_lines) + "\n",
        encoding="utf-8",
    )
    (output_dir / "week3_evaluation_summary.json").write_text(
        json.dumps(
            {
                "model_a_run": str(args.model_a_run),
                "model_b_comparison": str(args.model_b_comparison),
                "best_model_b_run": best_model_b["run_name"],
                "rows": comparison_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[OK] Wrote evaluation table -> {output_dir / 'week3_evaluation_table.csv'}")
    print(f"[OK] Wrote evaluation summary -> {output_dir / 'week3_evaluation_summary.md'}")
    print(f"[OK] Wrote evaluation JSON -> {output_dir / 'week3_evaluation_summary.json'}")


if __name__ == "__main__":
    main()
