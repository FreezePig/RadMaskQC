# RadMaskQC: Medical Image Segmentation Quality Checking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python toolkit for automated quality assessment of medical image segmentations. RadMaskQC detects common segmentation errors including noise, connectivity issues, elongated projections, sharp concavities, CT value anomalies, and internal holes.

## Features

- **2D Noise Detection**: Identifies small noisy components in mask slices
- **3D Connectivity Analysis**: Detects multiple disconnected ROIs in 3D
- **Elongated Projection Detection**: Finds leakage regions with abnormal aspect ratios
- **Sharp Concavity Detection**: Identifies sharp concave/convex defects
- **CT Value Anomaly Detection**: Detects tissue leakage using HU value analysis
- **Internal Hole Detection**: Identifies accidentally erased tissue using relative histogram analysis
- **Batch Processing**: Process multiple patient cases in one command
- **Visualization**: Automatic generation of detection visualizations
- **JSON Reports**: Export detailed detection reports for further analysis

## Installation

### Requirements

- Python 3.8 or higher
- NumPy
- OpenCV (cv2)
- NiBabel
- SciPy
- PyDicom (optional, for DICOM support)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/yourusername/RadMaskQC.git
cd RadMaskQC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Single File Processing

Run comprehensive quality checks on a single case:

```bash
# Run all available checks
python mask_check.py check-all mask.nii.gz ct.nii.gz -o ./output

# Run individual checks
python mask_check.py detect-2d-noise mask.nii.gz
python mask_check.py check-connectivity mask.nii.gz
python mask_check.py detect-elongated mask.nii.gz -o ./output
python mask_check.py detect-concavities mask.nii.gz
python mask_check.py detect-ct-anomalies mask.nii.gz ct.nii.gz
python mask_check.py detect-holes mask.nii.gz ct.nii.gz
```

### Batch Processing

Process multiple patients with a single command:

```bash
python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz
```

This will:
- Scan all patient directories in `./data`
- Process each patient's CT and mask files
- Save results to individual patient folders
- Generate a comprehensive batch summary report

## Directory Structure for Batch Processing

```
data/
├── patient_001/
│   ├── ct/
│   │   └── ct.nii.gz
│   ├── mask/
│   │   └── mask.nii.gz
│   └── qc_results/          # Auto-generated output
│       ├── connectivity_report.json
│       ├── 2d_noise_report.json
│       ├── elongated_report.json
│       ├── concavities_report.json
│       ├── ct_anomalies_report.json
│       ├── holes_report.json
│       ├── comprehensive_report.json
│       └── slice_XXXX_*.png  # Visualization images
├── patient_002/
│   └── ...
└── batch_summary.json       # Overall batch summary
```

## Command Reference

### mask_check.py (Single Case)

```bash
# General syntax
python mask_check.py <command> [options]

# Available commands:
#   detect-2d-noise          Detect 2D noise in mask slices
#   check-connectivity      Check 3D connectivity
#   detect-elongated        Detect elongated projections
#   detect-concavities      Detect sharp concavities
#   detect-ct-anomalies     Detect CT value anomalies
#   detect-holes            Detect internal holes
#   check-all               Run all checks

# Common options:
#   -o, --output            Output directory for results
#   --detailed              Print detailed report
#   --no-viz                Disable visualization
#   --help                  Show help message
```

#### Examples

```bash
# Detect 2D noise with custom threshold
python mask_check.py detect-2d-noise mask.nii.gz --min-area 20

# Check connectivity allowing up to 2 components
python mask_check.py check-connectivity mask.nii.gz --max-components 2

# Detect elongated projections with custom thresholds
python mask_check.py detect-elongated mask.nii.gz --aspect-ratio 6.0 --convexity 0.9

# Detect internal holes with custom percentiles
python mask_check.py detect-holes mask.nii.gz ct.nii.gz -t_air 20 -t_soft 50

# Run all checks with custom parameters
python mask_check.py check-all mask.nii.gz ct.nii.gz -o ./results \
    --aspect-ratio 6.0 \
    --angle-threshold 25.0 \
    --min-volume 100
```

### mask_check_batch.py (Batch Processing)

```bash
# General syntax
python mask_check_batch.py <root_folder> --ct-path <ct_path> --mask-path <mask_path> [options]

# Required arguments:
#   root_folder             Root folder containing patient directories
#   --ct-path               Relative path to CT file (e.g., 'ct/ct.nii.gz')
#   --mask-path             Relative path to mask file (e.g., 'mask/mask.nii.gz')

# Common options:
#   -o, --output            Output subdirectory name (default: qc_results)
#   --continue-on-error     Continue processing even if checks fail
#   --skip-on-error         Skip to next patient on error
#   --summary               Print detailed summary
#   --no-viz                Disable visualization
```

#### Examples

```bash
# Basic batch processing
python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz

# Custom output folder
python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz -o results

# Continue on errors (don't stop subsequent checks)
python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz --continue-on-error

# Skip to next patient on error
python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz --skip-on-error

# With detailed summary
python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz --summary

# Custom detection parameters
python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz \
    --aspect-ratio 6.0 \
    --angle-threshold 25.0 \
    --threshold-soft 60
```

## Detection Parameters Reference

### Common Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-components` | 1 | Maximum allowed 3D connected components |
| `--connectivity` | 3 | Connectivity type (1=6-conn, 2=18-conn, 3=26-conn) |
| `--min-area` | 10 | Minimum area threshold for 2D noise (pixels) |
| `--visualize` | True | Generate visualization images |

### Shape Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--aspect-ratio` | 5.0 | Aspect ratio threshold for elongated detection |
| `--convexity` | 0.85 | Convexity threshold for elongated detection |
| `--angle-threshold` | 30.0 | Sharp angle threshold for concavities (degrees) |
| `--distance-threshold` | 5.0 | Defect depth threshold for concavities (pixels) |

### CT-Based Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-volume` | 50 | Minimum anomaly region size (voxels) |
| `--z-score` | 2.0 | Z-score threshold for CT anomalies |
| `--max-area` | 20 | Maximum hole area for noise classification (pixels) |
| `--threshold-air` | 20 | Percentile threshold for air (20th percentile) |
| `--threshold-soft` | 50 | Percentile threshold for soft tissue (50th percentile) |

## Output Format

### Report Structure

Each detection generates a JSON report with the following structure:

```json
{
  "is_valid": true,
  "summary": {
    "affected_slices": [10, 15, 20]
  },
  "output_dir": "./output",
  "visualizations": 3
}
```

### Status Flags

- `is_valid: true` - No issues detected
- `is_valid: false` - Issues detected (see report for details)

### Visualization Files

- `slice_XXXX_elongated.png` - Elongated projections marked in red
- `slice_XXXX_concavities.png` - Sharp concavities marked in purple
- `slice_XXXX_ct_anomalies.png` - CT anomalies marked in cyan
- `slice_XXXX_holes.png` - Internal holes marked in yellow

## Use Cases

### Quality Control Pipeline

Integrate RadMaskQC into your medical image processing pipeline:

```bash
# After segmentation generation
python mask_check_batch.py ./segmentations \
    --ct-path images/ct.nii.gz \
    --mask-path segmentations/mask.nii.gz \
    --summary

# Review batch summary
cat batch_summary.json
```

### Research and Validation

Use RadMaskQC for systematic validation of segmentation algorithms:

```python
from check_utils import check_3d_connectivity, detect_2d_noise

# Evaluate segmentation quality
report = check_3d_connectivity("segmentation.nii.gz")
if not report["is_valid"]:
    print(f"Found {report['num_3d_components']} components")

# Compare different segmentation methods
for method in ["method_a", "method_b", "method_c"]:
    report = detect_2d_noise(f"{method}/mask.nii.gz")
    print(f"{method}: {report['summary']['noise_count']} noise regions")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use RadMaskQC in your research, please cite:

```bibtex
@software{radmaskqc2026,
  title = {RadMaskQC: Medical Image Segmentation Quality Checking},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/RadMaskQC}
}
```

## Acknowledgments

- Built with [NumPy](https://numpy.org/), [OpenCV](https://opencv.org/), and [SciPy](https://scipy.org/)
- Medical image I/O powered by [NiBabel](https://nipy.org/nibabel/)

## Contact

For questions, issues, or suggestions, please open an issue on GitHub.

---

**Note**: This tool is designed for research and quality assurance purposes. Clinical use requires additional validation and regulatory approval.
