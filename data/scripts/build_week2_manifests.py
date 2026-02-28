from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Iterable


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
DEFAULT_GENERATORS = ["Midjourney", "stable_diffusion_v_1_5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Week 2 manifests from GenImage and Kaggle datasets."
    )
    parser.add_argument("--genimage-root", type=Path, required=True)
    parser.add_argument("--kaggle-csv-dir", type=Path, required=True)
    parser.add_argument("--kaggle-train-images", type=Path)
    parser.add_argument("--kaggle-test-images", type=Path)
    parser.add_argument(
        "--generators",
        nargs="+",
        default=DEFAULT_GENERATORS,
        help="Generator folders under GenImage root.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="0 means no cap. Otherwise cap rows per class for GenImage.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests"))
    return parser.parse_args()


def iter_images(root: Path) -> Iterable[Path]:
    # Faster than rglob("*") + is_file() checks on very large folders.
    seen: set[str] = set()
    for ext in IMAGE_EXTS:
        for p in root.rglob(f"*{ext}"):
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            yield p
        for p in root.rglob(f"*{ext.upper()}"):
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            yield p


def collect_genimage_rows(
    root: Path, generators: list[str], max_per_class: int = 0
) -> list[dict]:
    rows: list[dict] = []
    class_counts = {"real": 0, "ai": 0}

    def reached_cap() -> bool:
        if max_per_class <= 0:
            return False
        return class_counts["real"] >= max_per_class and class_counts["ai"] >= max_per_class

    def can_take(label: str) -> bool:
        if max_per_class <= 0:
            return True
        return class_counts[label] < max_per_class

    for gen in generators:
        if reached_cap():
            break
        gen_dir = root / gen
        if not gen_dir.exists():
            print(f"[WARN] Missing generator folder: {gen_dir}")
            continue
        print(f"[INFO] Scanning generator: {gen}")
        split_bases = list(gen_dir.glob("*/train"))
        if (gen_dir / "train").exists():
            split_bases.append(gen_dir / "train")
        if not split_bases:
            print(f"[WARN] Could not find train folder under: {gen_dir}")
        # Deduplicate while preserving order
        seen = set()
        ordered_split_bases: list[Path] = []
        for base in split_bases:
            key = str(base.resolve())
            if key not in seen:
                seen.add(key)
                ordered_split_bases.append(base)

        for train_base in ordered_split_bases:
            base_root = train_base.parent
            for split_name in ("train", "val"):
                split_dir = base_root / split_name
                for cls_name, label in (("ai", "ai"), ("nature", "real")):
                    if not can_take(label):
                        continue
                    cls_dir = split_dir / cls_name
                    if not cls_dir.exists():
                        continue
                    for image_path in iter_images(cls_dir):
                        if not can_take(label):
                            break
                        rows.append(
                            {
                                "filepath": os.path.normpath(str(image_path)),
                                "label": label,
                                "source": "genimage",
                                "generator": gen,
                            }
                        )
                        class_counts[label] += 1
                    print(
                        f"[INFO] {gen}/{split_name}/{cls_name}: collected {class_counts[label]} {label}"
                    )
                if reached_cap():
                    break
            if reached_cap():
                break
    return rows


def apply_cap_per_class(rows: list[dict], max_per_class: int, seed: int) -> list[dict]:
    if max_per_class <= 0:
        return rows
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {"real": [], "ai": []}
    for row in rows:
        grouped[row["label"]].append(row)
    capped: list[dict] = []
    for label, label_rows in grouped.items():
        rng.shuffle(label_rows)
        capped.extend(label_rows[:max_per_class])
        print(f"[INFO] {label}: capped {len(label_rows)} -> {min(len(label_rows), max_per_class)}")
    return capped


def stratified_split(
    rows: list[dict], train_ratio: float, val_ratio: float, seed: int
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = {"real": [], "ai": []}
    for row in rows:
        grouped[row["label"]].append(row)

    out = {"train": [], "val": [], "test": []}
    for label, items in grouped.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        out["train"].extend({**x, "split": "train"} for x in items[:n_train])
        out["val"].extend({**x, "split": "val"} for x in items[n_train : n_train + n_val])
        out["test"].extend({**x, "split": "test"} for x in items[n_train + n_val : n_train + n_val + n_test])
        print(f"[INFO] label={label}: train={n_train}, val={n_val}, test={n_test}")
    return out


def detect_col(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def to_binary_label(raw: str) -> str:
    value = str(raw).strip().lower()
    ai_tokens = {"1", "ai", "fake", "generated", "synthetic"}
    return "ai" if value in ai_tokens else "real"


def resolve_image_path(rel: str, image_root: Path | None, csv_parent: Path) -> Path:
    rel_path = Path(rel)
    if rel_path.is_absolute():
        return rel_path

    candidates: list[Path] = []
    if image_root is not None:
        candidates.append(image_root / rel_path)
        candidates.append(image_root / rel_path.name)
        # Handles CSV paths like train_data/foo.jpg when image_root already ends with train_data
        rel_parts = rel_path.parts
        if rel_parts and image_root.name.lower() == rel_parts[0].lower():
            candidates.append(image_root.parent / rel_path)
    candidates.append(csv_parent / rel_path)
    candidates.append(csv_parent / rel_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fall back to first candidate for traceability if nothing exists yet.
    return candidates[0]


def build_kaggle_rows(csv_path: Path, image_root: Path | None, split: str) -> list[dict]:
    if not csv_path.exists():
        print(f"[WARN] Missing CSV: {csv_path}")
        return []

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        columns = list(reader.fieldnames)
        path_col = detect_col(columns, ["path", "filepath", "file_name", "filename", "id", "image_name"])
        label_col = detect_col(columns, ["label", "target", "class", "real_or_fake", "y"])

        if not path_col or not label_col:
            print(f"[WARN] Could not infer columns in {csv_path.name}. Found: {columns}")
            return []

        for idx, row in enumerate(reader, start=1):
            rel = str(row[path_col]).strip()
            if not rel:
                continue
            if "." not in Path(rel).name:
                rel = f"{rel}.jpg"
            full_path = resolve_image_path(rel, image_root, csv_path.parent)
            rows.append(
                {
                    "filepath": os.path.normpath(str(full_path)),
                    "label": to_binary_label(row[label_col]),
                    "source": "kaggle",
                    "generator": "mixed",
                    "split": split,
                }
            )
            if idx % 10000 == 0:
                print(f"[INFO] {csv_path.name}: processed {idx} rows...")
    print(f"[INFO] {csv_path.name}: total processed rows {len(rows)}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "label", "source", "generator", "split"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] Wrote {len(rows)} rows -> {path}")


def summarize(rows: list[dict]) -> dict:
    by_label = Counter(r["label"] for r in rows)
    by_source = Counter(r["source"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    return {
        "count": len(rows),
        "by_label": dict(by_label),
        "by_source": dict(by_source),
        "by_split": dict(by_split),
    }


def main() -> None:
    args = parse_args()
    ratio_total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_total - 1.0) > 0.001:
        raise ValueError("train/val/test ratios must sum to 1.0")

    gen_rows = collect_genimage_rows(
        args.genimage_root, args.generators, args.max_per_class
    )
    # Backward-compatible cap pass in case max_per_class is zero during collection.
    gen_rows = apply_cap_per_class(gen_rows, args.max_per_class, args.seed)
    gen_split = stratified_split(gen_rows, args.train_ratio, args.val_ratio, args.seed)

    kaggle_train = build_kaggle_rows(
        args.kaggle_csv_dir / "train.csv", args.kaggle_train_images, "external_train"
    )
    kaggle_test = build_kaggle_rows(
        args.kaggle_csv_dir / "test.csv", args.kaggle_test_images, "external_test"
    )
    kaggle_rows = kaggle_train + kaggle_test

    out = args.output_dir
    write_csv(out / "genimage_train.csv", gen_split["train"])
    write_csv(out / "genimage_val.csv", gen_split["val"])
    write_csv(out / "genimage_test.csv", gen_split["test"])
    write_csv(out / "kaggle_external_eval.csv", kaggle_rows)

    stats = {
        "genimage_train": summarize(gen_split["train"]),
        "genimage_val": summarize(gen_split["val"]),
        "genimage_test": summarize(gen_split["test"]),
        "kaggle_external_eval": summarize(kaggle_rows),
        "config": {
            "seed": args.seed,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "max_per_class": args.max_per_class,
            "generators": args.generators,
        },
    }
    stats_path = out / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[OK] Wrote stats -> {stats_path}")


if __name__ == "__main__":
    main()
