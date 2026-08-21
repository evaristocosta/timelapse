import argparse
import cv2
import numpy as np
import os
import re
from datetime import datetime
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
        default=20,
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
        "--start-index",
        type=int,
        default=None,
        help="0-based first image index to process within each subfolder.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="0-based last image index to process within each subfolder, inclusive.",
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
    return parser.parse_args()


def parse_capture_datetime(filename):
    match = re.search(r"(\d{8}_\d{6})", filename)
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


def main():
    args = parse_args()

    if args.keyframe_interval < 1:
        raise ValueError("--keyframe-interval must be at least 1.")
    if args.time_window_minutes < 0:
        raise ValueError("--time-window-minutes must be >= 0.")
    if args.start_index is not None and args.start_index < 0:
        raise ValueError("--start-index must be >= 0.")
    if args.end_index is not None and args.end_index < 0:
        raise ValueError("--end-index must be >= 0.")
    if args.start_index is not None and args.end_index is not None:
        if args.start_index > args.end_index:
            raise ValueError("--start-index must be less than or equal to --end-index.")

    input_dir = args.input_dir
    output_dir = args.output_dir
    zoom_factor = args.zoom_factor
    max_features = args.max_features
    min_matches = args.min_matches
    keyframe_interval = args.keyframe_interval
    time_window_minutes = args.time_window_minutes
    start_index = args.start_index
    end_index = args.end_index
    reference_index = args.reference_index
    reference_file = args.reference_file

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
    if start_index is not None or end_index is not None:
        print(
            f"📏 Processing range: {start_index if start_index is not None else 0} to {end_index if end_index is not None else 'end'}"
        )
    if reference_file is not None:
        print(f"📷 Reference file: {reference_file}")
    elif reference_index is not None:
        print(f"📷 Reference index: {reference_index}")

    # 📁 Create output directory structure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ Created output directory: {output_dir}")
    else:
        print(f"📁 Output directory already exists: {output_dir}")

    # 🔍 Discover subfolders in images directory
    try:
        subfolders = [
            f
            for f in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, f))
        ]
        print(f"📂 Found {len(subfolders)} subfolders: {subfolders}")
    except FileNotFoundError:
        print(f"❌ Error: '{input_dir}' directory not found!")
        subfolders = []

    # 🚀 Main Processing Loop
    total_processed = 0
    total_skipped = 0

    for subfolder in subfolders:
        print(f"\n{'='*50}")
        print(f"📂 Processing folder: {subfolder}")
        print(f"{'='*50}")

        # 📁 Setup folder paths
        input_folder_path = os.path.join(input_dir, subfolder)
        output_folder_path = os.path.join(output_dir, subfolder)

        # 📁 Create output subfolder if needed
        if not os.path.exists(output_folder_path):
            os.makedirs(output_folder_path)
            print(f"✅ Created output subfolder: {output_folder_path}")
        else:
            print(f"📁 Output subfolder exists: {output_folder_path}")

        # 🖼️ Discover image files
        image_files = [
            f
            for f in os.listdir(input_folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not image_files:
            print(f"⚠️ No image files found in {subfolder}")
            continue

        # 📋 Sort files for consistent processing order
        image_files.sort()

        if start_index is not None:
            start_index_clamped = max(0, min(start_index, len(image_files) - 1))
        else:
            start_index_clamped = 0

        if end_index is not None:
            end_index_clamped = max(
                start_index_clamped, min(end_index, len(image_files) - 1)
            )
        else:
            end_index_clamped = len(image_files) - 1

        if start_index_clamped > end_index_clamped:
            print(f"⚠️ No images to process in {subfolder} with the selected range.")
            continue

        selected_files = image_files[start_index_clamped : end_index_clamped + 1]
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
            if reference_file not in selected_files:
                print(
                    f"⚠️ Reference file '{reference_file}' is not within the selected range for {subfolder}."
                )
                continue
            keyframe_filename = reference_file
            keyframe_index = selected_files.index(keyframe_filename)
        elif reference_index is not None:
            if reference_index < 0 or reference_index >= len(selected_files):
                print(
                    f"⚠️ Reference index {reference_index} is out of range for {subfolder} (0..{len(selected_files)-1})."
                )
                continue
            keyframe_filename = selected_files[reference_index]
            keyframe_index = reference_index
        else:
            keyframe_filename = selected_files[0]
            keyframe_index = 0

        # 📷 Load stable keyframe image
        keyframe_image_path = os.path.join(input_folder_path, keyframe_filename)
        keyframe_img_color = cv2.imread(keyframe_image_path)

        if keyframe_img_color is None:
            print(f"❌ Could not read keyframe image: {keyframe_image_path}")
            continue

        keyframe_img_gray = cv2.cvtColor(keyframe_img_color, cv2.COLOR_BGR2GRAY)
        print(f"📸 Keyframe image: {keyframe_filename}")
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
            input_image_path = os.path.join(input_folder_path, filename)
            output_image_path = os.path.join(output_folder_path, filename)

            # 📖 Read current image
            img_color = cv2.imread(input_image_path)

            if img_color is None:
                print(f"⚠️ Could not read {filename}, skipping...")
                folder_skipped += 1
                continue

            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # 📷 Handle keyframe image (no alignment needed)
            if filename == keyframe_filename:
                zoomed_aligned = apply_zoom(img_color, zoom_factor)
                cv2.imwrite(output_image_path, zoomed_aligned)
                folder_processed += 1
                continue

            # 🔁 Reset to the original keyframe every N images to limit drift
            if index % keyframe_interval == 0 or index == keyframe_index:
                current_ref_color = keyframe_img_color
                current_ref_gray = keyframe_img_gray
                current_ref_kp = keyframe_kp
                current_ref_des = keyframe_des
                print(
                    f"🔁 Resetting reference to keyframe before processing {filename}"
                )

            # 🔍 Detect features in current image
            kp2, des2 = orb.detectAndCompute(img_gray, None)

            if des2 is None or len(kp2) < min_matches:
                print(
                    f"⚠️ Not enough features in {filename} ({len(kp2) if kp2 else 0} < {min_matches})"
                )
                folder_skipped += 1
                continue

            # 🎯 Match features between the active reference and current image
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(current_ref_des, des2)

            if len(matches) < min_matches:
                print(
                    f"⚠️ Not enough matches for {filename} ({len(matches)} < {min_matches})"
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
                    print(f"⚠️ Could not estimate transformation for {filename}")
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
                cv2.imwrite(output_image_path, zoomed_aligned)
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
                        f"⚠️ Local reference lost detail after {filename}; reverting to keyframe"
                    )

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                folder_skipped += 1
                continue

            # 🔁 Reset to the original keyframe every N images to limit drift
            if index % keyframe_interval == 0:
                current_ref_color = keyframe_img_color
                current_ref_gray = keyframe_img_gray
                current_ref_kp = keyframe_kp
                current_ref_des = keyframe_des
                print(
                    f"🔁 Resetting reference to keyframe before processing {filename}"
                )

            # 🔍 Detect features in current image
            kp2, des2 = orb.detectAndCompute(img_gray, None)

            if des2 is None or len(kp2) < min_matches:
                print(
                    f"⚠️ Not enough features in {filename} ({len(kp2) if kp2 else 0} < {min_matches})"
                )
                folder_skipped += 1
                continue

            # 🎯 Match features between the active reference and current image
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(current_ref_des, des2)

            if len(matches) < min_matches:
                print(
                    f"⚠️ Not enough matches for {filename} ({len(matches)} < {min_matches})"
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
                    print(f"⚠️ Could not estimate transformation for {filename}")
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
                cv2.imwrite(output_image_path, zoomed_aligned)
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
                        f"⚠️ Local reference lost detail after {filename}; reverting to keyframe"
                    )

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
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
