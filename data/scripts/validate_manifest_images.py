from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, UnidentifiedImageError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate image files in a manifest and write a cleaned output."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input CSV manifest path.")
    parser.add_argument("--output", type=Path, required=True, help="Output cleaned CSV path.")
    parser.add_argument("--report", type=Path, required=True, help="Output JSON report path.")
    parser.add_argument("--min-width", type=int, default=256)
    parser.add_argument("--min-height", type=int, default=256)
    return parser.parse_args()


def validate_row(row: dict, min_w: int, min_h: int) -> tuple[bool, str]:
    path = Path(row.get("filepath", ""))
    if not path.exists():
        return False, "missing_file"
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w < min_w or h < min_h:
                return False, "too_small"
            img.verify()
        return True, "ok"
    except (UnidentifiedImageError, OSError):
        return False, "corrupt_or_unreadable"
    except Exception:
        return False, "unknown_error"


def main() -> None:
    args = parse_args()
    rows: list[dict] = []
    with args.input.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)

    cleaned: list[dict] = []
    reasons = Counter()
    label_counts = Counter()

    for row in rows:
        ok, reason = validate_row(row, args.min_width, args.min_height)
        reasons[reason] += 1
        if ok:
            cleaned.append(row)
            label_counts[row.get("label", "unknown")] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        if cleaned:
            writer = csv.DictWriter(f, fieldnames=list(cleaned[0].keys()))
        else:
            writer = csv.DictWriter(
                f, fieldnames=["filepath", "label", "source", "generator", "split"]
            )
        writer.writeheader()
        writer.writerows(cleaned)

    report = {
        "input_manifest": str(args.input),
        "output_manifest": str(args.output),
        "min_width": args.min_width,
        "min_height": args.min_height,
        "total_rows": len(rows),
        "kept_rows": len(cleaned),
        "dropped_rows": len(rows) - len(cleaned),
        "drop_reasons": dict(reasons),
        "kept_label_distribution": dict(label_counts),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[OK] Input rows: {len(rows)}")
    print(f"[OK] Kept rows: {len(cleaned)}")
    print(f"[OK] Cleaned manifest: {args.output}")
    print(f"[OK] Report: {args.report}")


if __name__ == "__main__":
    main()
