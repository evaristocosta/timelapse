# 📸 Advanced Image Alignment for Timelapse Creation

A robust Python solution for aligning images taken from the same location over time to create smooth, professional-quality timelapse videos.

## ✨ Features

- Color preservation with full-color alignment
- ORB feature detection and RANSAC-based transform estimation
- Hybrid reference strategy with periodic resets to reduce drift
- Optional processing ranges and better-reference selection
- Timestamp-based filtering to discard outlier captures
- Batch processing for multiple folders under a shared input directory

## 🚀 Quick Start

### Prerequisites

This project uses uv to manage the Python environment and dependencies.

```bash
# Install uv if you do not already have it
pip install uv

# Create the environment and install the project dependencies
uv sync
```

Run the alignment script from the project root:

```powershell
uv run python .\timelapse.py --input-dir .\images --output-dir .\aligned --zoom-factor 1.15 --max-features 5000 --min-matches 100 --keyframe-interval 10 --time-window-minutes 45
```

### Folder Structure

```text
images/
├── location1/
│   ├── IMG_0001.jpg
│   ├── IMG_0002.jpg
│   └── ...
├── location2/
│   └── ...
└── ...
```

### Script Usage

```powershell
uv run python .\timelapse.py --input-dir .\images --output-dir .\aligned
```

Useful examples:

```powershell
uv run python .\timelapse.py --input-dir .\images --output-dir .\aligned --start-index 120 --end-index 220 --reference-index 10 --time-window-minutes 45
uv run python .\timelapse.py --input-dir .\images --output-dir .\aligned --start-index 120 --end-index 220 --reference-file IMG_0542.jpg --time-window-minutes 45
```

## 📋 Configuration Options

| Argument                | Default   | Description                                                    |
| ----------------------- | --------- | -------------------------------------------------------------- |
| `--input-dir`           | `images`  | Source folder containing subfolders with images                |
| `--output-dir`          | `aligned` | Folder where aligned images are saved                          |
| `--zoom-factor`         | `1.15`    | Zoom factor applied after alignment                            |
| `--max-features`        | `5000`    | ORB feature count                                              |
| `--min-matches`         | `100`     | Minimum good matches required                                  |
| `--keyframe-interval`   | `10`      | Reset to the stable keyframe every N images                    |
| `--time-window-minutes` | `45`      | Filter out captures farther than ±N minutes from the mean time |
| `--start-index`         | `None`    | First index to process within each folder                      |
| `--end-index`           | `None`    | Last index to process within each folder                       |
| `--reference-index`     | `None`    | Index of the selected reference image                          |
| `--reference-file`      | `None`    | Exact filename to use as the reference image                   |

## 🧠 Alignment Strategy

The current workflow is a hybrid approach:

1. Start from a stable keyframe, usually the first valid image in the folder.
2. Align the next frames against the last successful frame for local continuity.
3. Reset to the stable keyframe every `--keyframe-interval` images to reduce drift.
4. Restrict processing to selected ranges when needed.
5. Remove timestamp outliers using the capture time embedded in the filename.

This works much better than keeping a single static reference throughout a very long sequence.

## 🔧 Algorithm Details

- Feature detection: ORB
- Matching: BFMatcher with cross-checking
- Transformation: affine estimation with RANSAC
- Color handling: preserve the original BGR image while warping
- Border cleanup: crop-and-resize zoom to reduce black borders

## 🐛 Troubleshooting

## 🔬 ORB and Matching — Quick Explanation

- **ORB (Oriented FAST and Rotated BRIEF):** a fast, efficient feature detector + descriptor. ORB finds keypoints with FAST and describes them using a binary BRIEF-like descriptor that is rotation-aware. It is computationally inexpensive and works well for many timelapse scenes where features are reasonably stable.

- **Matching:** the script uses a `BFMatcher` with Hamming distance (suitable for ORB's binary descriptors) and `crossCheck=True` so matches are mutual best matches. After matching, the code keeps the best subset of matches and uses RANSAC to robustly estimate an affine transform.

- **What `--min-matches` means:** this parameter sets the minimum number of "good" matched keypoints required before the script will attempt to compute and accept an alignment for a frame. If either the detector or the matcher returns fewer than `--min-matches`, the frame is skipped to avoid producing unreliable warps.

- **Tuning guidance:**
  - The script default is conservative to avoid bad alignments (see the `timelapse.py` defaults).
  - Typical ranges:
    - High-quality, textured scenes: `min_matches = 20..50` — faster, more permissive.
    - Challenging or low-feature scenes: `min_matches = 5..15` — accept weaker matches.
    - Strict alignment (few false positives): `min_matches = 80..150` — safer but may skip many frames.

Adjust `--max-features` and `--zoom-factor` together with `--min-matches` for best results: increasing `--max-features` gives the matcher more keypoints to work with; lowering `--min-matches` allows alignment in sparse scenes but increases the risk of bad transforms.

## ⏱️ Keyframe Interval — When to Change

- **What it does:** `--keyframe-interval` controls how often (in frames) the algorithm resets the active reference back to the stable keyframe. The pipeline tracks locally using the last successful aligned frame for continuity, but periodically reverting to the stable keyframe prevents slow drift accumulating across many frames.

- **Trade-offs:**
  - Lower values (e.g., `1..5`): the reference is reset frequently. This reduces drift and is safer when the scene changes quickly or when matches are noisy, but it can break local continuity and increase the chance of visible jumps.
  - Higher values (e.g., `10..50`): the algorithm relies longer on local tracking, producing smoother motion when matches are reliable, but drift can accumulate over long sequences.

- **When to lower the interval:**
  - The scene has strong lighting changes or moving foreground objects.
  - You see gradual misalignment over time (drift) in longer runs.
  - Feature matching is marginal and the local reference often gets replaced due to insufficient keypoints.

- **When to increase the interval:**
  - Images are captured frequently and look nearly identical between frames (stable tripod, small motion).
  - You want very smooth local motion and have plenty of reliable matches.

- **Practical example:** start with the default (`10`) and inspect a short rendered timelapse. If drift appears after many frames, reduce the interval; if you observe small jittery jumps after resets, increase it.

### Common issues

- Not enough features detected: increase `--max-features` or choose a sharper reference image.
- Not enough matches found: lower `--min-matches` or pick a more stable `--reference-index` / `--reference-file`.
- Alignment only works for a short range: narrow the selected range and use a better reference frame.
- Too much border cropping: reduce `--zoom-factor` to `1.0` or `1.1`.

## 🎬 Creating a Video

Once the images are aligned, you can create a timelapse from the output folder with FFmpeg:

```bash
ffmpeg -framerate 30 -pattern_type glob -i "aligned/your_folder/*.jpg" -c:v libx264 -pix_fmt yuv420p timelapse.mp4
```

If you want a Python-only option, use the bundled renderer through uv:

```powershell
uv run python .\render_timelapse.py .\aligned\AP --output-file .\timelapse_AP.mp4 --fps 15 --fill-date-gaps --save-gap-frames
```

Useful options:

- `--fps 15`: slower playback that works well for long timelapses.
- `--fill-date-gaps`: inserts exactly one synthetic frame for each missing calendar day, fading the last photo toward black with the fade strength normalized by the longest gap in the sequence.
- `--save-gap-frames`: writes the generated frames to a sibling `*_gen` folder next to the source image sequence so they can be reused in another editor.
- `--gap-frames-dir <path>`: choose a custom folder for the saved bridge frames.
- `--output-file <path>`: choose the final video file name and location.

If your aligned images live in a different folder, replace `./aligned/AP` with that folder path.

## ✅ Notebook

The notebook in the project root is kept in sync with the script logic for interactive experimentation and parameter tuning. The script is the recommended option for repeatable batch processing.

### Parameter Tuning

| Scenario                  | Recommended Settings                  |
| ------------------------- | ------------------------------------- |
| **High-quality photos**   | `max_features=7000`, `min_matches=20` |
| **Challenging lighting**  | `max_features=5000`, `min_matches=8`  |
| **More border cropping**  | `zoom_factor=1.2-1.3`                 |
| **Preserve more content** | `zoom_factor=1.0-1.1`                 |

## 🐛 Troubleshooting

### Common Issues

**"Not enough features detected"**

- Increase `max_features` to 7000-10000
- Ensure images have sufficient detail/texture
- Check image quality and focus

**"Not enough matches found"**

- Decrease `min_matches` to 5-8
- Verify images are from same viewpoint
- Check for consistent lighting conditions

**"Too much border cropping"**

- Reduce `zoom_factor` to 1.0-1.1
- Consider manual cropping before processing

## 🙏 Acknowledgments

- OpenCV team for excellent computer vision library
- ORB algorithm developers for robust feature detection
- Community contributors for feedback and improvements
