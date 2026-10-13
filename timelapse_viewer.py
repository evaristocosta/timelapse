from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps

CATEGORIES = ("AP", "Elevator", "JKU", "Tree")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageRecord:
    category: str
    dt: datetime
    path: Path
    filename: str

    @property
    def date_label(self) -> str:
        return self.dt.date().isoformat()

    @property
    def time_label(self) -> str:
        return self.dt.strftime("%H:%M:%S")


def parse_capture_datetime(filename: str | Path) -> datetime | None:
    match = re.search(r"(\d{8}_\d{6})", str(filename))
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def discover_category_records(
    source_dir: str = "images",
) -> dict[str, list[ImageRecord]]:
    root = Path(__file__).resolve().parent / source_dir
    if not root.exists():
        raise FileNotFoundError(
            f"Image source '{source_dir}' does not exist at {root}."
        )

    records_by_category: dict[str, list[ImageRecord]] = {
        category: [] for category in CATEGORIES
    }

    for category in CATEGORIES:
        category_dir = root / category
        if not category_dir.exists():
            continue

        for item in sorted(category_dir.iterdir()):
            if not item.is_file():
                continue
            if item.suffix.lower() not in IMAGE_SUFFIXES:
                continue

            dt = parse_capture_datetime(item.name)
            if dt is None:
                continue

            records_by_category[category].append(
                ImageRecord(
                    category=category,
                    dt=dt,
                    path=item,
                    filename=item.name,
                )
            )

        records_by_category[category].sort(key=lambda record: record.dt)

    return records_by_category


def get_available_dates(records_by_category: dict[str, list[ImageRecord]]) -> list[str]:
    dates = {
        record.dt.date().isoformat()
        for records in records_by_category.values()
        for record in records
    }
    return sorted(dates)


def pick_record_for_date(
    records: Iterable[ImageRecord], target_date: str
) -> ImageRecord | None:
    matches = [
        record for record in records if record.dt.date().isoformat() == target_date
    ]
    return matches[-1] if matches else None


def get_day_records(
    records_by_category: dict[str, list[ImageRecord]], category: str, target_date: str
) -> list[ImageRecord]:
    return [
        record
        for record in records_by_category.get(category, [])
        if record.dt.date().isoformat() == target_date
    ]


@lru_cache(maxsize=128)
def load_display_image(path: str | Path, max_size: int = 1400) -> Image.Image:
    path = str(path)
    with Image.open(path) as image:
        corrected = ImageOps.exif_transpose(image)
        if max(corrected.size) > max_size:
            corrected.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return corrected.copy()


__all__ = [
    "CATEGORIES",
    "ImageRecord",
    "discover_category_records",
    "get_available_dates",
    "pick_record_for_date",
    "get_day_records",
    "load_display_image",
    "parse_capture_datetime",
]
