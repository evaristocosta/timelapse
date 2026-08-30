import argparse
import cv2
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

"""
This is a copy of the notebook contents, organized as a script for easier execution.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align timelapse frames using ORB matching with hybrid keyframe tracking."
    )
    parser.add_argument(
        "--input-dir",
        default="images",
        help="Folder containing one subfolder per scene/location.",
    )
    parser.add_argument(
        "--output-dir",
        default="aligned",
        help="Output folder where aligned images are saved.",
    )
    parser.add_argument(
        "--zoom-factor",
        type=float,
        default=1.15,
        help="Zoom-in crop factor applied after alignment (1.0 = none).",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Maximum ORB features used for alignment.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=100,
        help="Minimum good feature matches required before aligning a frame.",
    )
    parser.add_argument(
        "--keyframe-interval",
        type=int,
        default=10,
        help="Reset to the stable keyframe every N images to reduce drift.",
    )
    parser.add_argument(
        "--time-window-minutes",
        type=int,
        default=45,
        help="Discard images whose capture time is more than N minutes from the folder mean capture time. Use 0 to disable filtering.",
    )
    parser.add_argument(
        "--start-file",
        default=None,
        help="Exact filename to use as the first image to process within each folder.",
    )
    parser.add_argument(
        "--end-file",
        default=None,
        help="Exact filename to use as the last image to process within each folder, inclusive.",
    )
    parser.add_argument(
        "--reference-index",
        type=int,
        default=None,
        help="0-based index inside the selected range to use as the stable reference frame.",
    )
    parser.add_argument(
        "--reference-file",
        default=None,
        help="Exact filename to use as the stable reference frame instead of the first image.",
    )
    brightness_group = parser.add_mutually_exclusive_group()
    brightness_group.add_argument(
        "--brightness-normalization",
        dest="brightness_normalization",
        action="store_true",
        default=True,
        help="Normalize each frame to a standard brightness before alignment and saving (default).",
    )
    brightness_group.add_argument(
        "--no-brightness-normalization",
        dest="brightness_normalization",
        action="store_false",
        help="Disable automatic brightness normalization.",
    )
    parser.add_argument(
        "--brightness-target",
        type=float,
        default=140.0,
        help="Target average brightness (0-255) used when brightness normalization is enabled.",
    )
    return parser.parse_args()


def parse_capture_datetime(filename):
    match = re.search(r"(\d{8}_\d{6})", str(filename))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")


def apply_zoom(image, zoom_factor):
    """
    🔍 Apply zoom-in effect by cropping the center of the image

    This function reduces black borders created during image alignment
    by cropping the center portion and resizing back to original dimensions.

    Args:
        image: Input image (BGR format)
        zoom_factor: Zoom level (1.0 = no zoom, 1.2 = 20% zoom-in)

    Returns:
        Zoomed image with same dimensions as input
    """
    h, w = image.shape[:2]

    # Calculate new dimensions after zoom
    new_h = int(h / zoom_factor)
    new_w = int(w / zoom_factor)

    # Calculate crop coordinates (center crop)
    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2
    end_x = start_x + new_w
    end_y = start_y + new_h

    # Crop the center portion
    cropped = image[start_y:end_y, start_x:end_x]

    # Resize back to original dimensions using high-quality interpolation
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LANCZOS4)

    return zoomed


def normalize_brightness(image, target_brightness):
    """Adjust the image so its average luminance approaches the target value."""
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    current_brightness = float(l_channel.mean())
    if current_brightness <= 0:
        return image

    scale = target_brightness / current_brightness
    scale = float(np.clip(scale, 0.5, 3.0))

    adjusted_l = np.clip(l_channel.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    normalized_lab = cv2.merge((adjusted_l, a_channel, b_channel))
    return cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)


def main():
    args = parse_args()

    if args.keyframe_interval < 1:
        raise ValueError("--keyframe-interval must be at least 1.")
    if args.time_window_minutes < 0:
        raise ValueError("--time-window-minutes must be >= 0.")
    if args.start_file is not None and not args.start_file.strip():
        raise ValueError("--start-file cannot be empty.")
    if args.end_file is not None and not args.end_file.strip():
        raise ValueError("--end-file cannot be empty.")

    input_dir = args.input_dir
    output_dir = args.output_dir
    zoom_factor = args.zoom_factor
    max_features = args.max_features
    min_matches = args.min_matches
    keyframe_interval = args.keyframe_interval
    time_window_minutes = args.time_window_minutes
    start_file = args.start_file
    end_file = args.end_file
    reference_index = args.reference_index
    reference_file = args.reference_file
    brightness_normalization = args.brightness_normalization
    brightness_target = args.brightness_target

    print("✅ Libraries imported and configuration set!")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔍 Zoom factor: {zoom_factor}x")
    print(f"🎯 Max features: {max_features}")
    print(f"🔁 Keyframe interval: every {keyframe_interval} images")
    print(
        f"🕒 Time window: ±{time_window_minutes} minutes from the mean capture time"
        if time_window_minutes > 0
        else "🕒 Time window: disabled"
    )
    if start_file is not None or end_file is not None:
        print(
            f"📏 Processing range: {start_file if start_file is not None else 'start'} to {end_file if end_file is not None else 'end'}"
        )
    if reference_file is not None:
        print(f"📷 Reference file: {reference_file}")
    elif reference_index is not None:
        print(f"📷 Reference index: {reference_index}")
    if brightness_normalization:
        print(f"💡 Brightness normalization: enabled (target {brightness_target:.1f})")
    else:
        print("💡 Brightness normalization: disabled")
    # 📁 Create output directory structure
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir)

    if not output_dir_path.exists():
        output_dir_path.mkdir(parents=True)
        print(f"✅ Created output directory: {output_dir_path}")
    else:
        print(f"📁 Output directory already exists: {output_dir_path}")

    # 🔍 Discover subfolders in images directory
    try:
        subfolders = [p for p in input_dir_path.iterdir() if p.is_dir()]
        root_image_files = [
            p
            for p in input_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        if subfolders:
            print(
                f"📂 Found {len(subfolders)} subfolders: {[p.name for p in subfolders]}"
            )
            folder_specs = [
                (
                    f.name,
                    f,
                    output_dir_path / f.name,
                )
                for f in subfolders
            ]
        elif root_image_files:
            print(
                f"📂 No subfolders found, but {len(root_image_files)} image file(s) exist at the input root; processing as a single folder."
            )
            folder_specs = [
                (
                    input_dir_path.name or "images",
                    input_dir_path,
                    output_dir_path,
                )
            ]
        else:
            print(f"📂 Found 0 subfolders and no root-level images in {input_dir}")
            folder_specs = []
    except FileNotFoundError:
        print(f"❌ Error: '{input_dir}' directory not found!")
        folder_specs = []

    # 🚀 Main Processing Loop
    total_processed = 0
    total_skipped = 0

    for subfolder, input_folder_path, output_folder_path in folder_specs:
        print(f"\n{'='*50}")
        print(f"📂 Processing folder: {subfolder}")
        print(f"{'='*50}")

        # 📁 Create output subfolder if needed
        if not output_folder_path.exists():
            output_folder_path.mkdir(parents=True)
            print(f"✅ Created output subfolder: {output_folder_path}")
        else:
            print(f"📁 Output subfolder exists: {output_folder_path}")

        # 🖼️ Discover image files
        image_files = [
            p
            for p in input_folder_path.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        if not image_files:
            print(f"⚠️ No image files found in {subfolder}")
            continue

        # 📋 Sort files for consistent processing order
        image_files.sort()

        start_index = 0
        end_index = len(image_files) - 1

        if start_file is not None:
            start_matches = [p for p in image_files if p.name == start_file]
            if not start_matches:
                print(f"⚠️ Start file '{start_file}' was not found in {subfolder}.")
                continue
            start_index = image_files.index(start_matches[0])

        if end_file is not None:
            end_matches = [p for p in image_files if p.name == end_file]
            if not end_matches:
                print(f"⚠️ End file '{end_file}' was not found in {subfolder}.")
                continue
            end_index = image_files.index(end_matches[0])

        if start_index > end_index:
            print(f"⚠️ Start file comes after end file in {subfolder}.")
            continue

        selected_files = image_files[start_index : end_index + 1]
        print(f"🔢 Found {len(selected_files)} images in the selected range")

        if time_window_minutes > 0:
            timestamps = []
            in_window_files = []
            for filename in selected_files:
                capture_dt = parse_capture_datetime(filename)
                if capture_dt is None:
                    print(
                        f"⚠️ Could not parse timestamp from '{filename}', skipping it."
                    )
                    continue
                timestamps.append(capture_dt)
                in_window_files.append(filename)

            if not timestamps:
                print(f"⚠️ No files with parseable timestamps in {subfolder}.")
                continue

            minutes_from_midnight = np.array(
                [dt.hour * 60 + dt.minute + dt.second / 60 for dt in timestamps]
            )
            mean_minutes = float(minutes_from_midnight.mean())
            lower_bound = mean_minutes - time_window_minutes
            upper_bound = mean_minutes + time_window_minutes

            valid_files = [
                filename
                for filename, capture_dt in zip(
                    in_window_files,
                    timestamps,
                )
                if lower_bound
                <= (capture_dt.hour * 60 + capture_dt.minute + capture_dt.second / 60)
                <= upper_bound
            ]
            dropped_files = len(in_window_files) - len(valid_files)

            if dropped_files > 0:
                print(
                    f"🕒 Removed {dropped_files} outlier(s) outside ±{time_window_minutes} minutes around the mean capture time ({mean_minutes/60:.2f}h)."
                )

            if not valid_files:
                print(f"⚠️ No files left after time filtering in {subfolder}.")
                continue

            selected_files = valid_files
            print(f"🔢 Kept {len(selected_files)} images after time filtering")

        if reference_file is not None:
            reference_matches = [p for p in selected_files if p.name == reference_file]
            if not reference_matches:
                print(
                    f"⚠️ Reference file '{reference_file}' is not within the selected range for {subfolder}."
                )
                continue
            keyframe_file = reference_matches[0]
            keyframe_index = selected_files.index(keyframe_file)
        elif reference_index is not None:
            if reference_index < 0 or reference_index >= len(selected_files):
                print(
                    f"⚠️ Reference index {reference_index} is out of range for {subfolder} (0..{len(selected_files)-1})."
                )
                continue
            keyframe_file = selected_files[reference_index]
            keyframe_index = reference_index
        else:
            keyframe_file = selected_files[0]
            keyframe_index = 0

        # 📷 Load stable keyframe image
        keyframe_image_path = keyframe_file
        keyframe_img_color = cv2.imread(str(keyframe_image_path))

        if keyframe_img_color is None:
            print(f"❌ Could not read keyframe image: {keyframe_image_path}")
            continue

        if brightness_normalization:
            keyframe_img_color = normalize_brightness(
                keyframe_img_color, brightness_target
            )

        keyframe_img_gray = cv2.cvtColor(keyframe_img_color, cv2.COLOR_BGR2GRAY)
        print(f"📸 Keyframe image: {keyframe_file.name}")
        print(
            f"📐 Image dimensions: {keyframe_img_color.shape[1]}x{keyframe_img_color.shape[0]}"
        )

        # 🎯 Initialize ORB feature detector
        orb = cv2.ORB_create(nfeatures=max_features)
        keyframe_kp, keyframe_des = orb.detectAndCompute(keyframe_img_gray, None)

        if keyframe_des is None or len(keyframe_kp) < min_matches:
            print(
                f"⚠️ Not enough features in keyframe image ({len(keyframe_kp) if keyframe_kp else 0} < {min_matches})"
            )
            continue

        print(f"🔍 Features detected in keyframe: {len(keyframe_kp)}")

        # 🔄 Start from a stable keyframe, then track locally to reduce drift
        current_ref_color = keyframe_img_color
        current_ref_gray = keyframe_img_gray
        current_ref_kp = keyframe_kp
        current_ref_des = keyframe_des

        # 🔄 Process each image in the folder
        folder_processed = 0
        folder_skipped = 0

        for index, filename in enumerate(
            tqdm(selected_files, desc=f"Aligning {subfolder}", unit="img")
        ):
            input_image_path = filename
            output_image_path = output_folder_path / filename.name

            # 📖 Read current image
            img_color = cv2.imread(str(input_image_path))

            if img_color is None:
                print(f"⚠️ Could not read {filename.name}, skipping...")
                folder_skipped += 1
                continue

            if brightness_normalization:
                img_color = normalize_brightness(img_color, brightness_target)

            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # 📷 Handle keyframe image (no alignment needed)
            if filename == keyframe_file:
                zoomed_aligned = apply_zoom(img_color, zoom_factor)
                cv2.imwrite(str(output_image_path), zoomed_aligned)
                folder_processed += 1
                continue

            # 🔁 Reset to the original keyframe every N images to limit drift
            if index % keyframe_interval == 0 or index == keyframe_index:
                current_ref_color = keyframe_img_color
                current_ref_gray = keyframe_img_gray
                current_ref_kp = keyframe_kp
                current_ref_des = keyframe_des
                print(
                    f"🔁 Resetting reference to keyframe before processing {filename.name}"
                )

            # 🔍 Detect features in current image
            kp2, des2 = orb.detectAndCompute(img_gray, None)

            if des2 is None or len(kp2) < min_matches:
                print(
                    f"⚠️ Not enough features in {filename.name} ({len(kp2) if kp2 else 0} < {min_matches})"
                )
                folder_skipped += 1
                continue

            # 🎯 Match features between the active reference and current image
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(current_ref_des, des2)

            if len(matches) < min_matches:
                print(
                    f"⚠️ Not enough matches for {filename.name} ({len(matches)} < {min_matches})"
                )
                folder_skipped += 1
                continue

            # 📊 Select best matches for transformation
            matches = sorted(matches, key=lambda x: x.distance)
            num_matches = max(20, min(len(matches), int(len(matches) * 0.3)))
            good_matches = matches[:num_matches]

            # 📐 Extract matched point coordinates
            src_pts = np.float32(
                [current_ref_kp[m.queryIdx].pt for m in good_matches]
            ).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )

            # 🎯 Calculate robust transformation using RANSAC
            try:
                M, mask = cv2.estimateAffinePartial2D(
                    dst_pts,
                    src_pts,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=3.0,
                    maxIters=2000,
                    confidence=0.99,
                )

                if M is None:
                    print(f"⚠️ Could not estimate transformation for {filename.name}")
                    folder_skipped += 1
                    continue

                # 🎨 Apply transformation to preserve colors
                aligned = cv2.warpAffine(
                    img_color,
                    M,
                    (current_ref_color.shape[1], current_ref_color.shape[0]),
                )

                # ✂️ Apply zoom to reduce black borders
                zoomed_aligned = apply_zoom(aligned, zoom_factor)

                # 💾 Save aligned image
                cv2.imwrite(str(output_image_path), zoomed_aligned)
                folder_processed += 1

                # 🔄 Use the aligned result as the local reference for the next frame
                current_ref_color = aligned
                current_ref_gray = cv2.cvtColor(current_ref_color, cv2.COLOR_BGR2GRAY)
                current_ref_kp, current_ref_des = orb.detectAndCompute(
                    current_ref_gray, None
                )

                if current_ref_des is None or len(current_ref_kp) < min_matches:
                    current_ref_color = keyframe_img_color
                    current_ref_gray = keyframe_img_gray
                    current_ref_kp = keyframe_kp
                    current_ref_des = keyframe_des
                    print(
                        f"⚠️ Local reference lost detail after {filename.name}; reverting to keyframe"
                    )

            except Exception as e:
                print(f"❌ Error processing {filename.name}: {e}")
                folder_skipped += 1
                continue

        # 📊 Folder summary
        print(f"\n📊 Folder '{subfolder}' Summary:")
        print(f"   ✅ Successfully processed: {folder_processed}")
        print(f"   ⚠️ Skipped: {folder_skipped}")
        print(
            f"   📈 Success rate: {folder_processed/(folder_processed+folder_skipped)*100:.1f}%"
        )

        total_processed += folder_processed
        total_skipped += folder_skipped

    # 🎉 Final summary
    print(f"\n{'='*60}")
    print(f"🎉 PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"✅ Total images processed: {total_processed}")
    print(f"⚠️ Total images skipped: {total_skipped}")
    print(
        f"📈 Overall success rate: {total_processed/(total_processed+total_skipped)*100:.1f}%"
    )
    print(f"🔍 Zoom factor applied: {zoom_factor}x")
    print(f"🔁 Keyframe interval: every {keyframe_interval} images")
    print(f"📁 Results saved to: {output_dir}/")
    print(f"\n🎬 Your aligned images are ready for timelapse creation!")


if __name__ == "__main__":
    main()
