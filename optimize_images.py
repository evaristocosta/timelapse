#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def optimize_image(path: Path, output_path: Path, max_size: int, quality: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(path) as image:
            rgb_image = ImageOps.exif_transpose(image)
            if rgb_image.mode not in {"RGB", "L"}:
                rgb_image = rgb_image.convert("RGB")

            if max(rgb_image.size) > max_size:
                rgb_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            rgb_image.save(
                output_path,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
                subsampling=0,
            )
    except UnidentifiedImageError:
        print(f"Skipping unreadable image: {path}")


def process_root(
    source_root: Path,
    output_root: Path,
    max_size: int,
    quality: int,
    dry_run: bool,
    limit: int | None,
) -> tuple[int, int]:
    files = iter_image_files(source_root)
    if limit is not None:
        files = files[:limit]

    total_in = sum(file.stat().st_size for file in files)
    total_out = 0

    for file in files:
        relative = file.relative_to(source_root)
        target = output_root / relative
        target = target.with_suffix(".jpg")

        if dry_run:
            print(f"Would optimize: {file} -> {target}")
            continue

        optimize_image(file, target, max_size=max_size, quality=quality)
        total_out += target.stat().st_size

    return total_in, total_out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create smaller JPEG versions of your timelapse images for GitHub or a web deployment. "
            "This keeps the original source folder intact and writes compressed copies to an output folder."
        )
    )
    parser.add_argument(
        "roots",
        nargs="+",
        help="Image folders to compress, for example: images aligned",
    )
    parser.add_argument(
        "--output-root",
        default="optimized_assets",
        help="Root folder for compressed output. Default: optimized_assets",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=1400,
        help="Maximum side length in pixels before resizing. Default: 1400",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=70,
        help="JPEG quality between 1 and 100. Default: 70",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing: process only the first N files in each folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the planned output paths without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()

    total_in = 0
    total_out = 0

    for root_name in args.roots:
        source_root = Path(root_name).resolve()
        if output_root == source_root:
            raise ValueError(
                "The output folder must be different from the source folder. "
                "Use a separate path like optimized_assets."
            )

        print(f"Processing {source_root} -> {output_root / source_root.name}")
        root_in, root_out = process_root(
            source_root=source_root,
            output_root=output_root / source_root.name,
            max_size=args.max_size,
            quality=args.quality,
            dry_run=args.dry_run,
            limit=args.limit,
        )
        total_in += root_in
        total_out += root_out

    if not args.dry_run:
        print(f"Input size: {total_in / 1024 / 1024:.1f} MB")
        print(f"Output size: {total_out / 1024 / 1024:.1f} MB")
        print(f"Saved: {((total_in - total_out) / total_in * 100):.1f}%")
        print(f"Compressed files written to: {output_root}")
    else:
        print("Dry-run complete. No files were written.")


if __name__ == "__main__":
    main()
