from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from backend.database.db import SessionLocal
from backend.database.models import AnalysisRecord
from backend.models.training.model_b_features import extract_model_b_features_from_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Model B feature rows from analysis records."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label-map",
        type=Path,
        help="Optional CSV with columns record_id,label for supervised fusion training.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for exported records. 0 exports all.",
    )
    return parser.parse_args()


def load_label_map(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    rows: dict[int, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record_id = row.get("record_id")
            label = row.get("label")
            if record_id is None or label is None:
                continue
            rows[int(record_id)] = label.strip().lower()
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No feature rows to export.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    labels = load_label_map(args.label_map)

    db = SessionLocal()
    try:
        query = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.asc())
        if args.limit > 0:
            query = query.limit(args.limit)
        records = query.all()
    finally:
        db.close()

    rows: list[dict] = []
    for record in records:
        row = extract_model_b_features_from_record(record)
        if labels:
            row["target_label"] = labels.get(record.id)
        rows.append(row)

    write_csv(args.output, rows)

    summary = {
        "output": str(args.output),
        "rows": len(rows),
        "label_map_used": str(args.label_map) if args.label_map else None,
        "labeled_rows": sum(1 for row in rows if row.get("target_label") is not None),
        "feature_columns": list(rows[0].keys()) if rows else [],
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] Exported Model B feature rows -> {args.output}")
    print(f"[OK] Export summary -> {summary_path}")


if __name__ == "__main__":
    main()
