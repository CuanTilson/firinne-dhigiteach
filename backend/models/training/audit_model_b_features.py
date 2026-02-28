from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


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
        description="Audit Model B feature rows for leakage and trivially separable signals."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model_b_audit"),
    )
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if row.get("target_label") in LABEL_MAP]
        return rows, list(reader.fieldnames or [])


def safe_float(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def infer_extension(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    return suffix or "<none>"


def threshold_metrics(values: np.ndarray, labels: np.ndarray) -> dict[str, float | str]:
    unique_values = np.unique(values)
    if unique_values.size <= 1:
        return {
            "best_accuracy": 0.0,
            "best_f1_ai": 0.0,
            "best_rule": "constant",
            "best_threshold": float(unique_values[0]) if unique_values.size == 1 else 0.0,
        }

    candidates = []
    sorted_values = np.sort(unique_values)
    for left, right in zip(sorted_values[:-1], sorted_values[1:]):
        candidates.append((left + right) / 2.0)
    candidates.extend([sorted_values[0], sorted_values[-1]])

    best = {
        "best_accuracy": -1.0,
        "best_f1_ai": -1.0,
        "best_rule": "",
        "best_threshold": 0.0,
    }
    for threshold in candidates:
        for rule_name, preds in (
            ("value >= threshold", values >= threshold),
            ("value <= threshold", values <= threshold),
        ):
            pred_int = preds.astype(np.int64)
            tp = int(((pred_int == 1) & (labels == 1)).sum())
            fp = int(((pred_int == 1) & (labels == 0)).sum())
            tn = int(((pred_int == 0) & (labels == 0)).sum())
            fn = int(((pred_int == 0) & (labels == 1)).sum())
            total = labels.size
            accuracy = (tp + tn) / total if total else 0.0
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            if accuracy > best["best_accuracy"] or (
                accuracy == best["best_accuracy"] and f1 > best["best_f1_ai"]
            ):
                best = {
                    "best_accuracy": accuracy,
                    "best_f1_ai": f1,
                    "best_rule": rule_name,
                    "best_threshold": float(threshold),
                }
    return best


def build_feature_report(rows: list[dict[str, str]], fieldnames: list[str]) -> list[dict]:
    feature_names = [name for name in fieldnames if name not in NON_FEATURE_COLUMNS]
    labels = np.array([LABEL_MAP[row["target_label"]] for row in rows], dtype=np.int64)
    report_rows: list[dict] = []

    for feature_name in feature_names:
        values = np.array([safe_float(row.get(feature_name)) for row in rows], dtype=np.float64)
        real_values = values[labels == 0]
        ai_values = values[labels == 1]
        threshold_report = threshold_metrics(values, labels)
        real_unique = np.unique(real_values)
        ai_unique = np.unique(ai_values)
        overlap_count = int(np.intersect1d(real_unique, ai_unique).size)
        report_rows.append(
            {
                "feature": feature_name,
                "real_mean": float(real_values.mean()) if real_values.size else 0.0,
                "ai_mean": float(ai_values.mean()) if ai_values.size else 0.0,
                "real_std": float(real_values.std()) if real_values.size else 0.0,
                "ai_std": float(ai_values.std()) if ai_values.size else 0.0,
                "overall_unique_values": int(np.unique(values).size),
                "real_unique_values": int(real_unique.size),
                "ai_unique_values": int(ai_unique.size),
                "shared_unique_values": overlap_count,
                "mean_gap_abs": float(abs(ai_values.mean() - real_values.mean()))
                if real_values.size and ai_values.size
                else 0.0,
                "single_feature_best_accuracy": float(threshold_report["best_accuracy"]),
                "single_feature_best_f1_ai": float(threshold_report["best_f1_ai"]),
                "single_feature_rule": threshold_report["best_rule"],
                "single_feature_threshold": float(threshold_report["best_threshold"]),
                "looks_perfectly_separable": threshold_report["best_accuracy"] >= 0.999,
                "looks_near_perfect": threshold_report["best_accuracy"] >= 0.95,
                "zero_variance": bool(np.allclose(values.std(), 0.0)),
            }
        )

    report_rows.sort(
        key=lambda item: (
            item["single_feature_best_accuracy"],
            item["mean_gap_abs"],
        ),
        reverse=True,
    )
    return report_rows


def build_extension_report(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        counts[row["target_label"]][infer_extension(row.get("filename"))] += 1
    return {label: dict(counter) for label, counter in counts.items()}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict, top_rows: list[dict]) -> None:
    lines = [
        "# Model B Leakage Audit",
        "",
        f"- input: `{summary['input_features_csv']}`",
        f"- total rows: `{summary['total_rows']}`",
        f"- class balance: real=`{summary['class_counts'].get('real', 0)}`, ai=`{summary['class_counts'].get('ai', 0)}`",
        "",
        "## Key Finding",
        "",
    ]

    if summary["perfect_separator_count"] > 0:
        lines.append(
            f"The current Model B dataset contains `{summary['perfect_separator_count']}` "
            "single features that can perfectly separate the classes on this sample."
        )
    else:
        lines.append("No single perfect separator was found in this sample.")

    lines.extend(
        [
            "",
            "## Top Suspect Features",
            "",
            "| Feature | Best Single-Feature Accuracy | Rule | Threshold |",
            "| --- | ---: | --- | ---: |",
        ]
    )
    for row in top_rows:
        lines.append(
            f"| `{row['feature']}` | {row['single_feature_best_accuracy']:.3f} | "
            f"{row['single_feature_rule']} | {row['single_feature_threshold']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Extension Distribution",
            "",
            f"- real: `{summary['extension_counts'].get('real', {})}`",
            f"- ai: `{summary['extension_counts'].get('ai', {})}`",
            "",
            "## Interpretation",
            "",
            "If file extension, JPEG structure, or similar source-format signals separate the "
            "classes almost perfectly, the current Model B result is inflated by dataset shortcuts.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows, fieldnames = load_rows(args.features_csv)
    if len(rows) < 20:
        raise ValueError("Need at least 20 labeled rows to audit Model B features.")

    feature_report = build_feature_report(rows, fieldnames)
    extension_report = build_extension_report(rows)
    perfect_rows = [row for row in feature_report if row["looks_perfectly_separable"]]
    near_perfect_rows = [row for row in feature_report if row["looks_near_perfect"]]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_features_csv": str(args.features_csv),
        "total_rows": len(rows),
        "class_counts": dict(Counter(row["target_label"] for row in rows)),
        "feature_count": len(feature_report),
        "perfect_separator_count": len(perfect_rows),
        "near_perfect_separator_count": len(near_perfect_rows),
        "top_5_features": [row["feature"] for row in feature_report[:5]],
        "extension_counts": extension_report,
    }

    write_csv(output_dir / "feature_leakage_report.csv", feature_report)
    write_markdown(output_dir / "feature_leakage_report.md", summary, feature_report[:10])
    (output_dir / "feature_leakage_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Wrote feature audit -> {output_dir / 'feature_leakage_report.csv'}")
    print(f"[OK] Wrote audit summary -> {output_dir / 'feature_leakage_summary.json'}")
    print(f"[OK] Wrote audit note -> {output_dir / 'feature_leakage_report.md'}")


if __name__ == "__main__":
    main()
