# 📸 Advanced Image Alignment for Timelapse Creation

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.0+-green.svg)
[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow?logo=buy-me-a-coffee)](https://buymeacoffee.com/lucascosta74)

A robust Python solution for aligning images taken from the same location over time to create smooth, professional-quality timelapse videos. This tool handles camera shake, slight position changes, and lighting variations while preserving original colors.

## ✨ Features

- **🎨 Color Preservation**: Maintains original image colors (no grayscale output)
- **🎯 Robust Alignment**: Uses ORB feature detection with RANSAC for reliable matching
- **✂️ Smart Cropping**: Applies zoom-in to reduce black borders from alignment
- **📁 Batch Processing**: Handles multiple folders automatically
- **🛡️ Error Handling**: Gracefully manages problematic images
- **📊 Detailed Reporting**: Progress tracking and success statistics
- **⚙️ Configurable**: Easy parameter tuning for different scenarios

## 🚀 Quick Start

### Prerequisites

```bash
pip install opencv-python numpy tqdm
```

### Folder Structure

Organize your images in the following structure:

```
images/
├── location1/          # e.g., "apartment_view"
│   ├── IMG_001.jpg
│   ├── IMG_002.jpg
│   └── ...
├── location2/          # e.g., "construction_site"
│   ├── IMG_001.jpg
│   └── ...
└── location3/          # e.g., "garden_growth"
    └── ...
```

### Usage

1. **Open the Jupyter Notebook**: `timelapse.ipynb`
2. **Configure parameters** in the second cell if needed
3. **Run all cells** to process your images
4. **Find aligned images** in the `aligned/` directory

The tool will automatically:

- Create output directories
- Process each location folder independently
- Use the first image in each folder as reference
- Apply robust alignment and smart cropping
- Generate detailed progress reports

## 📋 Configuration Options

| Parameter      | Default     | Description                                   |
| -------------- | ----------- | --------------------------------------------- |
| `input_dir`    | `"images"`  | Source folder containing image subfolders     |
| `output_dir`   | `"aligned"` | Output folder for aligned images              |
| `zoom_factor`  | `1.15`      | Zoom level (1.0 = no zoom, 1.2 = 20% zoom-in) |
| `max_features` | `5000`      | Number of ORB features to detect              |
| `min_matches`  | `10`        | Minimum matches required for alignment        |

## 🎬 Creating Timelapse Videos

After alignment, some options to create videos are:

### FFmpeg (Recommended)

```bash
# Create MP4 timelapse at 30 FPS
ffmpeg -framerate 30 -pattern_type glob -i "aligned/your_folder/*.jpg" \
       -c:v libx264 -pix_fmt yuv420p timelapse.mp4

# Create GIF timelapse
ffmpeg -framerate 10 -pattern_type glob -i "aligned/your_folder/*.jpg" \
       -vf "scale=800:-1" timelapse.gif
```

### DaVinci Resolve (Free)

1. Import image sequence
2. Set duration per frame
3. Add transitions and effects
4. Export in various formats

### Adobe Premiere/After Effects

1. Import as image sequence
2. Adjust frame rate
3. Add motion blur for smoother playback

## 🔧 Algorithm Details

### Image Alignment Process

1. **📷 Reference Selection**: First image in each folder serves as alignment target
2. **🔍 Feature Detection**: ORB (Oriented FAST and Rotated BRIEF) extracts keypoints
3. **🎯 Feature Matching**: Brute-force matcher with cross-checking finds correspondences
4. **📐 Transformation**: RANSAC estimates robust affine transformation
5. **🎨 Color Application**: Transformation applied to full-color images
6. **✂️ Smart Cropping**: Zoom applied to reduce alignment artifacts

### Key Technologies

- **OpenCV**: Computer vision operations
- **ORB Features**: Fast, rotation-invariant feature detection
- **RANSAC**: Robust transformation estimation
- **Affine Transformation**: Handles translation, rotation, and scaling
- **LANCZOS4 Interpolation**: High-quality image resizing

## 📊 Performance Tips

### For Better Results

- **📅 Consistent Timing**: Take photos at the same time of day
- **📷 Stable Position**: Use tripod or consistent hand position
- **⏰ Regular Intervals**: Maintain consistent time gaps
- **🌤️ Weather Awareness**: Avoid windy conditions for outdoor subjects

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

