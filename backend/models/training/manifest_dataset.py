from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


LABEL_MAP = {"real": 0, "ai": 1}


@dataclass(frozen=True)
class ManifestRow:
    filepath: Path
    label_text: str
    source: str
    generator: str
    split: str

    @property
    def label(self) -> int:
        if self.label_text not in LABEL_MAP:
            raise ValueError(f"Unsupported label: {self.label_text}")
        return LABEL_MAP[self.label_text]


def read_manifest(path: str | Path) -> list[ManifestRow]:
    manifest_path = Path(path)
    rows: list[ManifestRow] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                ManifestRow(
                    filepath=Path(row["filepath"]),
                    label_text=row["label"].strip().lower(),
                    source=row.get("source", "").strip(),
                    generator=row.get("generator", "").strip(),
                    split=row.get("split", "").strip(),
                )
            )
    return rows


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: str | Path, transform=None):
        self.rows = read_manifest(manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row.filepath) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, row.label
