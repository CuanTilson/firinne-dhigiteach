from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from backend.database.db import SessionLocal
from backend.database.models import AnalysisRecord
from backend.main import THUMB_DIR, run_full_analysis
from backend.models.training.model_b_features import extract_model_b_features_from_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small labeled Model B dataset by analyzing manifest images."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="One or more cleaned manifests to sample from.",
    )
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--min-unique-extensions-per-class",
        type=int,
        default=1,
        help="Fail if any class has fewer than this many unique file extensions.",
    )
    parser.add_argument(
        "--require-extension-overlap",
        action="store_true",
        help="Fail if real and ai classes do not share at least one extension.",
    )
    parser.add_argument(
        "--analysis-copy-dir",
        type=Path,
        default=Path("backend/storage/uploaded/model_b_batch"),
        help="Where copied analysis inputs should be stored.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model_b_dataset"),
        help="Where feature exports and summaries are written.",
    )
    return parser.parse_args()


def load_manifest_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["_manifest_path"] = str(path)
                rows.append(row)
    return rows


def sample_rows(rows: list[dict], per_class: int, seed: int) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = row.get("label", "").strip().lower()
        if label in {"real", "ai"}:
            grouped[label].append(row)

    rng = random.Random(seed)
    sampled: list[dict] = []
    for label in ("real", "ai"):
        label_rows = grouped[label]
        rng.shuffle(label_rows)
        chosen = label_rows[:per_class]
        sampled.extend(chosen)
        print(f"[INFO] sampled {len(chosen)} rows for label={label}")
    rng.shuffle(sampled)
    return sampled


def summarize_extensions(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        label = row.get("label", "").strip().lower()
        if label in {"real", "ai"}:
            ext = Path(row.get("filepath", "")).suffix.lower() or "<none>"
            counts[label][ext] += 1
    return {label: dict(counter) for label, counter in counts.items()}


def validate_extension_distribution(
    extension_counts: dict[str, dict[str, int]],
    min_unique_extensions_per_class: int,
    require_extension_overlap: bool,
) -> None:
    for label in ("real", "ai"):
        unique_count = len(extension_counts.get(label, {}))
        if unique_count < min_unique_extensions_per_class:
            raise ValueError(
                f"Class '{label}' only has {unique_count} unique extensions in the sampled rows; "
                f"expected at least {min_unique_extensions_per_class}."
            )

    if require_extension_overlap:
        real_exts = set(extension_counts.get("real", {}))
        ai_exts = set(extension_counts.get("ai", {}))
        if not (real_exts & ai_exts):
            raise ValueError(
                "The sampled real and ai rows do not share any file extensions. "
                "This dataset is likely to leak format shortcuts."
            )


def copy_for_analysis(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return dest


def create_thumbnail(image_path: Path) -> Path:
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    thumb_path = THUMB_DIR / f"{uuid.uuid4().hex}_thumb.jpg"
    with Image.open(image_path) as img:
        img.thumbnail((128, 128))
        img.save(thumb_path, "JPEG")
    return thumb_path


def create_record(original_name: str, analysis_path: Path) -> AnalysisRecord:
    analysis = run_full_analysis(analysis_path)
    file_integrity = analysis["file_integrity"]
    ml = analysis["ml"]
    ml_prob = analysis["ml_prob"]
    metadata = analysis["metadata"]
    exif_result = analysis["exif_result"]
    anomaly = analysis["anomaly"]
    qtinfo = analysis["qtinfo"]
    noise_info = analysis["noise_info"]
    watermark_info = analysis["watermark_info"]
    c2pa_info = analysis["c2pa_info"]
    ela_info = analysis["ela_info"]
    heatmap_path = analysis["heatmap_path"]
    fused = analysis["fused"]

    thumb_path = create_thumbnail(analysis_path)

    return AnalysisRecord(
        filename=original_name,
        saved_path=str(analysis_path),
        thumbnail_path=str(thumb_path),
        ml_probability=ml_prob,
        ml_label=ml["label"],
        forensic_score=fused["final_score"],
        classification=fused["classification"],
        gradcam_heatmap=str(heatmap_path),
        ela_heatmap=ela_info["ela_image_path"],
        forensic_score_json=fused,
        ml_prediction={"probability": ml_prob, "label": ml["label"]},
        metadata_anomalies=anomaly,
        file_integrity=file_integrity,
        ai_watermark=watermark_info,
        metadata_json=metadata,
        exif_forensics=exif_result,
        c2pa=c2pa_info,
        jpeg_qtables=qtinfo,
        noise_residual=noise_info,
        ela_analysis=ela_info,
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_rows = load_manifest_rows(args.manifest)
    sampled_rows = sample_rows(source_rows, args.per_class, args.seed)
    extension_counts = summarize_extensions(sampled_rows)
    validate_extension_distribution(
        extension_counts,
        args.min_unique_extensions_per_class,
        args.require_extension_overlap,
    )

    db = SessionLocal()
    created_record_ids: list[int] = []
    feature_rows: list[dict] = []
    label_map_rows: list[dict] = []

    try:
        for index, row in enumerate(sampled_rows, start=1):
            src = Path(row["filepath"])
            if not src.exists():
                print(f"[WARN] missing source file: {src}")
                continue

            copied_path = copy_for_analysis(src, args.analysis_copy_dir)
            record = create_record(src.name, copied_path)
            db.add(record)
            db.commit()
            db.refresh(record)
            created_record_ids.append(record.id)

            feature_row = extract_model_b_features_from_record(record)
            feature_row["target_label"] = row["label"]
            feature_rows.append(feature_row)

            label_map_rows.append(
                {
                    "record_id": record.id,
                    "filename": src.name,
                    "inferred_label": row["label"],
                    "final_label": row["label"],
                    "label_source": "manifest_sample",
                    "notes": row.get("_manifest_path", ""),
                }
            )

            if index % 20 == 0:
                print(f"[INFO] analyzed {index} / {len(sampled_rows)} images")
    finally:
        db.close()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "model_b_features_labeled.csv", feature_rows)
    write_csv(output_dir / "model_b_label_map.csv", label_map_rows)

    summary = {
        "seed": args.seed,
        "per_class_requested": args.per_class,
        "rows_sampled": len(sampled_rows),
        "records_created": len(created_record_ids),
        "output_dir": str(output_dir),
        "analysis_copy_dir": str(args.analysis_copy_dir),
        "manifests": [str(path) for path in args.manifest],
        "extension_counts": extension_counts,
        "min_unique_extensions_per_class": args.min_unique_extensions_per_class,
        "require_extension_overlap": args.require_extension_overlap,
    }
    (output_dir / "build_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"[OK] Labeled Model B features -> {output_dir / 'model_b_features_labeled.csv'}")
    print(f"[OK] Label map -> {output_dir / 'model_b_label_map.csv'}")
    print(f"[OK] Build summary -> {output_dir / 'build_summary.json'}")


if __name__ == "__main__":
    main()
