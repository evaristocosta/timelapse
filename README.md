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

Use the project environment or install the required dependencies:

```bash
pip install opencv-python numpy tqdm
```

If you are using Conda, this is the recommended workflow:

```powershell
conda activate timelapse
python .\timelapse.py --input-dir .\images --output-dir .\aligned --zoom-factor 1.15 --max-features 5000 --min-matches 20 --keyframe-interval 10 --time-window-minutes 45
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
conda activate timelapse
python .\timelapse.py --input-dir .\images --output-dir .\aligned
```

Useful examples:

```powershell
python .\timelapse.py --input-dir .\images --output-dir .\aligned --start-index 120 --end-index 220 --reference-index 10 --time-window-minutes 45
python .\timelapse.py --input-dir .\images --output-dir .\aligned --start-index 120 --end-index 220 --reference-file IMG_0542.jpg --time-window-minutes 45
```

## 📋 Configuration Options

| Argument                | Default   | Description                                                    |
| ----------------------- | --------- | -------------------------------------------------------------- |
| `--input-dir`           | `images`  | Source folder containing subfolders with images                |
| `--output-dir`          | `aligned` | Folder where aligned images are saved                          |
| `--zoom-factor`         | `1.15`    | Zoom factor applied after alignment                            |
| `--max-features`        | `5000`    | ORB feature count                                              |
| `--min-matches`         | `20`      | Minimum good matches required                                  |
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
