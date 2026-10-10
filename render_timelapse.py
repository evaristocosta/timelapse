import argparse
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def log(message):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a folder of aligned images into a timelapse video."
    )
    parser.add_argument(
        "input_dir",
        help="Folder containing the aligned images to render.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Output video file path. Defaults to <input_dir>.mp4 in the same parent folder.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC codec for the output video, for example mp4v or avc1.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".jpg", ".jpeg", ".png"],
        help="Image file extensions to include.",
    )
    parser.add_argument(
        "--resize-to-first-frame",
        action="store_true",
        help="Resize later frames to match the first readable frame if sizes differ.",
    )
    parser.add_argument(
        "--fill-date-gaps",
        action="store_true",
        help="Insert one synthetic frame per missing calendar day, fading the last photo toward black.",
    )
    parser.add_argument(
        "--save-gap-frames",
        action="store_true",
        help="Save generated gap bridge frames to disk alongside the rendered video.",
    )
    parser.add_argument(
        "--gap-frames-dir",
        default=None,
        help="Directory to save generated gap frames. Defaults to a sibling *_gen folder next to the input image folder.",
    )
    return parser.parse_args()


def collect_image_files(input_dir: Path, extensions: set[str]):
    return sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ]
    )


def parse_capture_datetime(filename: Path):
    match = re.search(r"(\d{8}_\d{6})", filename.name)
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def format_capture_label(capture_datetime):
    if capture_datetime is None:
        return "unknown"
    return capture_datetime.strftime("%Y%m%d_%H%M%S")


def collect_gap_statistics(image_files):
    gap_runs = []
    max_missing_days = 0

    if not image_files:
        return gap_runs, max_missing_days

    previous_datetime = parse_capture_datetime(image_files[0])

    for image_file in image_files[1:]:
        current_datetime = parse_capture_datetime(image_file)
        if previous_datetime is not None and current_datetime is not None:
            missing_days = (current_datetime.date() - previous_datetime.date()).days - 1
            if missing_days > 0:
                gap_runs.append((previous_datetime, current_datetime, missing_days))
                max_missing_days = max(max_missing_days, missing_days)

        previous_datetime = current_datetime

    return gap_runs, max_missing_days


def resize_if_needed(frame, size, allow_resize):
    height, width = size[1], size[0]
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    if not allow_resize:
        raise ValueError(
            f"Frame size mismatch: {frame.shape[1]}x{frame.shape[0]} != {width}x{height}. "
            "Use --resize-to-first-frame to allow resizing."
        )
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def scale_frame(frame, scale):
    return np.clip(frame.astype(np.float32) * scale, 0, 255).astype(np.uint8)


def write_frame(writer, frame, output_size, allow_resize):
    writer.write(resize_if_needed(frame, output_size, allow_resize))


def save_frame(image_dir, filename, frame, output_size, allow_resize):
    image_dir.mkdir(parents=True, exist_ok=True)
    path = image_dir / filename
    cv2.imwrite(str(path), resize_if_needed(frame, output_size, allow_resize))
    return path


def write_hold(writer, frame, output_size, allow_resize, frame_count):
    for _ in range(frame_count):
        write_frame(writer, frame, output_size, allow_resize)


def write_gap_bridge(
    writer,
    last_frame,
    output_size,
    allow_resize,
    missing_days,
    gap_decay_step,
    save_gap_frames=False,
    gap_frames_dir=None,
    gap_frame_prefix=None,
):
    black_frame = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
    for index in range(1, missing_days + 1):
        alpha = min(0.5, index * gap_decay_step)
        blended = cv2.addWeighted(last_frame, 1.0 - alpha, black_frame, alpha, 0.0)
        writer.write(resize_if_needed(blended, output_size, allow_resize))

        if (
            save_gap_frames
            and gap_frames_dir is not None
            and gap_frame_prefix is not None
        ):
            save_frame(
                gap_frames_dir,
                f"{gap_frame_prefix}_gap_{index:04d}.png",
                blended,
                output_size,
                allow_resize,
            )


def estimate_output_frames(
    image_files,
    fill_date_gaps,
):
    if not image_files:
        return 0

    total_frames = 1
    previous_datetime = parse_capture_datetime(image_files[0])

    for image_file in image_files[1:]:
        current_datetime = parse_capture_datetime(image_file)

        if (
            fill_date_gaps
            and previous_datetime is not None
            and current_datetime is not None
        ):
            missing_days = (current_datetime.date() - previous_datetime.date()).days - 1
            if missing_days > 0:
                total_frames += missing_days

        total_frames += 1
        previous_datetime = current_datetime

    return total_frames


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    extensions = {ext.lower() for ext in args.extensions}
    image_files = collect_image_files(input_dir, extensions)

    if not image_files:
        raise ValueError(
            f"No image files found in {input_dir} with extensions: {sorted(extensions)}"
        )

    output_file = (
        Path(args.output_file) if args.output_file else input_dir.with_suffix(".mp4")
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fps = float(args.fps)
    if fps <= 0:
        raise ValueError("FPS must be greater than 0.")
    if (
        args.save_gap_frames
        and args.gap_frames_dir is not None
        and not args.gap_frames_dir.strip()
    ):
        raise ValueError("--gap-frames-dir cannot be empty.")

    first_frame = None
    first_frame_path = None
    for image_file in image_files:
        frame = cv2.imread(str(image_file))
        if frame is not None:
            first_frame = frame
            first_frame_path = image_file
            break

    if first_frame is None:
        raise ValueError(f"Could not read any images from {input_dir}")

    height, width = first_frame.shape[:2]
    output_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, output_size)

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open video writer for {output_file} using codec {args.codec}"
        )

    log(f"📁 Input folder: {input_dir}")
    log(f"🎬 Output video: {output_file}")
    log(f"🔁 Frames found: {len(image_files)}")
    log(f"⚙️ FPS: {fps}")
    log(f"🧩 Codec: {args.codec}")
    log(f"📐 Frame size: {width}x{height}")
    log(f"🖼️ First readable frame: {first_frame_path.name}")
    if args.fill_date_gaps:
        log(
            "🕳️ Date gaps: enabled (one synthetic frame per missing day, fading toward black)"
        )
    else:
        log("🕳️ Date gaps: disabled")

    if args.save_gap_frames:
        gap_frames_dir = (
            Path(args.gap_frames_dir)
            if args.gap_frames_dir
            else input_dir.with_name(f"{input_dir.name}_gen")
        )
        gap_frames_dir.mkdir(parents=True, exist_ok=True)
        log(f"💾 Gap frames: saving to {gap_frames_dir}")
    else:
        gap_frames_dir = None
        log("💾 Gap frames: not saved")

    _gap_runs, max_missing_days = collect_gap_statistics(image_files)
    gap_decay_step = 0.0 if max_missing_days <= 0 else 0.5 / max_missing_days
    if args.fill_date_gaps and max_missing_days > 0:
        log(
            f"🕳️ Gap fade scale: longest gap is {max_missing_days} day(s), step is {gap_decay_step * 100:.1f}% per day"
        )

    estimated_frames = estimate_output_frames(image_files, args.fill_date_gaps)
    log(f"📏 Estimated output frames: {estimated_frames}")

    previous_frame = first_frame
    previous_datetime = parse_capture_datetime(first_frame_path)

    progress = tqdm(total=estimated_frames, desc="Rendering", unit="frame")

    writer.write(previous_frame)
    progress.update(1)

    for image_file in image_files[1:]:
        progress.set_description(f"Rendering {image_file.name}")
        frame = cv2.imread(str(image_file))
        if frame is None:
            log(f"⚠️ Skipping unreadable file: {image_file.name}")
            continue

        current_datetime = parse_capture_datetime(image_file)
        if (
            args.fill_date_gaps
            and previous_datetime is not None
            and current_datetime is not None
        ):
            missing_days = (current_datetime.date() - previous_datetime.date()).days - 1
            if missing_days > 0:
                gap_label = f"{format_capture_label(previous_datetime)}_to_{format_capture_label(current_datetime)}"
                log(
                    f"🕳️ Filling {missing_days} missing day(s) between {previous_datetime.date()} and {current_datetime.date()}"
                )
                write_gap_bridge(
                    writer,
                    previous_frame,
                    output_size,
                    args.resize_to_first_frame,
                    missing_days,
                    gap_decay_step,
                    save_gap_frames=args.save_gap_frames,
                    gap_frames_dir=gap_frames_dir,
                    gap_frame_prefix=gap_label,
                )
                progress.update(missing_days)
                write_frame(writer, frame, output_size, args.resize_to_first_frame)
                progress.update(1)
                previous_frame = frame
                previous_datetime = current_datetime
                continue

        write_frame(writer, frame, output_size, args.resize_to_first_frame)
        progress.update(1)
        previous_frame = frame
        previous_datetime = current_datetime

    progress.close()
    writer.release()

    if estimated_frames == 0:
        raise ValueError("No frames were written to the video.")

    log(f"✅ Video written successfully: {output_file}")
    log(f"📊 Frames written: {estimated_frames}")


if __name__ == "__main__":
    main()
