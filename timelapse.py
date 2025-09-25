import cv2
import numpy as np
import os
from tqdm import tqdm

"""
This is a copy of the notebook contents, organized as a script for easier execution.
"""


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
    # 📂 Directory Configuration
    input_dir = "images"  # Source folder containing subfolders with images
    output_dir = "aligned"  # Output folder for aligned images

    # ⚙️ Processing Parameters
    zoom_factor = 1.15  # Zoom level (1.0 = no zoom, 1.2 = 20% zoom-in)
    max_features = 5000  # Number of ORB features to detect
    min_matches = 10  # Minimum matches required for alignment

    print("✅ Libraries imported and configuration set!")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"🔍 Zoom factor: {zoom_factor}x")
    print(f"🎯 Max features: {max_features}")

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
        print(f"🔢 Found {len(image_files)} images to process")

        # 📷 Load reference image (first image in folder)
        ref_image_path = os.path.join(input_folder_path, image_files[0])
        ref_img_color = cv2.imread(ref_image_path)

        if ref_img_color is None:
            print(f"❌ Could not read reference image: {ref_image_path}")
            continue

        ref_img_gray = cv2.cvtColor(ref_img_color, cv2.COLOR_BGR2GRAY)
        print(f"📸 Reference image: {image_files[0]}")
        print(f"📐 Image dimensions: {ref_img_color.shape[1]}x{ref_img_color.shape[0]}")

        # 🎯 Initialize ORB feature detector
        orb = cv2.ORB_create(nfeatures=max_features)
        kp1, des1 = orb.detectAndCompute(ref_img_gray, None)
        print(f"🔍 Features detected in reference: {len(kp1)}")

        # 🔄 Process each image in the folder
        folder_processed = 0
        folder_skipped = 0

        for filename in tqdm(image_files, desc=f"Aligning {subfolder}", unit="img"):
            input_image_path = os.path.join(input_folder_path, filename)
            output_image_path = os.path.join(output_folder_path, filename)

            # 📖 Read current image
            img_color = cv2.imread(input_image_path)

            if img_color is None:
                print(f"⚠️ Could not read {filename}, skipping...")
                folder_skipped += 1
                continue

            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            # 📷 Handle reference image (no alignment needed)
            if filename == image_files[0]:
                zoomed_aligned = apply_zoom(img_color, zoom_factor)
                cv2.imwrite(output_image_path, zoomed_aligned)
                folder_processed += 1
                continue

            # 🔍 Detect features in current image
            kp2, des2 = orb.detectAndCompute(img_gray, None)

            if des2 is None or len(kp2) < min_matches:
                print(
                    f"⚠️ Not enough features in {filename} ({len(kp2) if kp2 else 0} < {min_matches})"
                )
                folder_skipped += 1
                continue

            # 🎯 Match features between reference and current image
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)

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
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
                -1, 1, 2
            )
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
                    img_color, M, (ref_img_color.shape[1], ref_img_color.shape[0])
                )

                # ✂️ Apply zoom to reduce black borders
                zoomed_aligned = apply_zoom(aligned, zoom_factor)

                # 💾 Save aligned image
                cv2.imwrite(output_image_path, zoomed_aligned)
                folder_processed += 1

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
    print(f"📁 Results saved to: {output_dir}/")
    print(f"\n🎬 Your aligned images are ready for timelapse creation!")


if __name__ == "__main__":
    main()
