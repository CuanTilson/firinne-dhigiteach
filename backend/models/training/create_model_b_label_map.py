from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a Model B label-map template from exported feature rows."
    )
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        default=[],
        help="Optional cleaned manifest(s) used to infer labels by filename.",
    )
    return parser.parse_args()


def load_manifest_label_lookup(paths: list[Path]) -> dict[str, str | None]:
    matches: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filepath = row.get("filepath")
                label = row.get("label")
                if not filepath or not label:
                    continue
                matches[Path(filepath).name].add(label.strip().lower())

    lookup: dict[str, str | None] = {}
    for filename, labels in matches.items():
        lookup[filename] = next(iter(labels)) if len(labels) == 1 else None
    return lookup


def main() -> None:
    args = parse_args()
    label_lookup = load_manifest_label_lookup(args.manifest)

    rows: list[dict[str, str]] = []
    with args.features_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            filename = row.get("filename", "")
            inferred = label_lookup.get(filename, "")
            rows.append(
                {
                    "record_id": row.get("record_id", ""),
                    "filename": filename,
                    "inferred_label": inferred or "",
                    "final_label": inferred or "",
                    "label_source": "manifest_match" if inferred else "",
                    "notes": "",
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "record_id",
                "filename",
                "inferred_label",
                "final_label",
                "label_source",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote label map template -> {args.output}")


if __name__ == "__main__":
    main()
