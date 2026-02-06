# Changelog

All notable changes to RadMaskQC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of RadMaskQC
- 2D noise detection using connected component analysis
- 3D connectivity checking with configurable connectivity types
- Elongated projection detection with aspect ratio and convexity analysis
- Sharp concavity detection using convexity defects
- CT value anomaly detection using IQR and z-score methods
- Internal hole detection with relative histogram analysis
- Single file processing interface (mask_check.py)
- Batch processing interface (mask_check_batch.py)
- Automatic visualization generation
- JSON report export
- Comprehensive documentation

## [1.0.0] - 2024-02-06

### Added
- Initial public release
- Complete quality checking toolkit for medical image segmentations
- Support for multiple medical image formats (.nii, .nii.gz, .npy, .dcm)
- Batch processing capabilities
- Detailed documentation and usage examples
