from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cleaned manifests for missing generator/label coverage."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Cleaned manifest to audit. May be provided multiple times.",
    )
    parser.add_argument(
        "--expected-generator",
        action="append",
        default=[],
        help="Expected generator name. May be provided multiple times.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the JSON audit report.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Where to write the markdown audit report.",
    )
    return parser.parse_args()


def audit_manifest(path: Path, expected_generators: list[str]) -> dict:
    counts = Counter()
    generators_seen = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            generator = str(row.get("generator") or "").strip()
            label = str(row.get("label") or "").strip()
            if not generator or not label:
                continue
            counts[(generator, label)] += 1
            generators_seen.add(generator)

    missing_pairs: list[dict] = []
    for generator in expected_generators:
        for label in ("real", "ai"):
            pair_count = counts.get((generator, label), 0)
            if pair_count == 0:
                missing_pairs.append(
                    {
                        "generator": generator,
                        "label": label,
                    }
                )

    return {
        "manifest": str(path),
        "total_rows": sum(counts.values()),
        "generators_seen": sorted(generators_seen),
        "generator_label_counts": {
            f"{generator}:{label}": count
            for (generator, label), count in sorted(counts.items())
        },
        "missing_generator_label_pairs": missing_pairs,
        "ok": len(missing_pairs) == 0,
    }


def main() -> None:
    args = parse_args()
    expected_generators = list(dict.fromkeys(args.expected_generator))
    reports = [audit_manifest(path, expected_generators) for path in args.manifest]

    missing_by_generator: dict[str, list[str]] = defaultdict(list)
    for report in reports:
        for item in report["missing_generator_label_pairs"]:
            missing_by_generator[item["generator"]].append(
                f"{Path(report['manifest']).name}:{item['label']}"
            )

    summary = {
        "expected_generators": expected_generators,
        "manifests": reports,
        "overall_ok": all(report["ok"] for report in reports),
        "missing_by_generator": dict(sorted(missing_by_generator.items())),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Manifest Coverage Audit",
        "",
        f"- Overall ok: `{summary['overall_ok']}`",
        f"- Expected generators: {', '.join(expected_generators) if expected_generators else 'None'}",
        "",
    ]
    for report in reports:
        lines.append(f"## `{Path(report['manifest']).name}`")
        lines.append("")
        lines.append(f"- Total rows: `{report['total_rows']}`")
        lines.append(f"- Generators seen: {', '.join(report['generators_seen']) if report['generators_seen'] else 'None'}")
        if report["missing_generator_label_pairs"]:
            lines.append("- Missing generator/label pairs:")
            for item in report["missing_generator_label_pairs"]:
                lines.append(f"  - `{item['generator']}:{item['label']}`")
        else:
            lines.append("- Missing generator/label pairs: none")
        lines.append("")

    if summary["missing_by_generator"]:
        lines.append("## Problem Summary")
        lines.append("")
        for generator, missing_entries in summary["missing_by_generator"].items():
            lines.append(f"- `{generator}` missing in: {', '.join(missing_entries)}")
        lines.append("")

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Wrote JSON audit -> {args.output_json}")
    print(f"[OK] Wrote markdown audit -> {args.output_md}")


if __name__ == "__main__":
    main()
