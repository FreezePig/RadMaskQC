# Contributing to RadMaskQC

Thank you for your interest in contributing to RadMaskQC! This document provides guidelines and instructions for contributing to the project.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description** of the problem
- **Minimal reproducible example** if applicable
- **System information** (Python version, OS, package versions)
- **Steps to reproduce** the issue
- **Expected behavior** vs **actual behavior**
- **Screenshots/logs** if helpful

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- **Clear description** of the proposed enhancement
- **Use cases** and benefits
- **Potential implementation approach** (if known)
- **Examples** or mockups if applicable

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Make your changes** with clear, descriptive commit messages
4. **Add tests** if applicable (see Testing section below)
5. **Update documentation** as needed
6. **Ensure all tests pass**: `pytest`
7. **Submit a pull request** with a clear description of changes

## Development Setup

### Cloning the Repository

```bash
git clone https://github.com/yourusername/RadMaskQC.git
cd RadMaskQC
```

### Creating a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=radtools --cov-report=html

# Run specific test file
pytest tests/test_check_utils.py
```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions, classes, and modules
- Include type hints for function parameters and return values

### Example Code Style

```python
def detect_2d_noise(
    mask_data: Union[str, np.ndarray],
    output_dir: Optional[str] = None,
    min_area_threshold: int = 10
) -> Dict[str, Any]:
    """
    Detect 2D noise in mask slices using connected component analysis.

    Args:
        mask_data: Path to mask file or numpy array
        output_dir: Output directory for visualizations (optional)
        min_area_threshold: Minimum area threshold for noise detection

    Returns:
        Dictionary containing detection results
    """
    # Implementation here
    pass
```

### Documentation

- Update README.md if adding new features or changing usage
- Update docstrings when modifying function behavior
- Add examples for new use cases
- Keep CHANGELOG.md updated with notable changes

## Project Structure

```
RadMaskQC/
├── check_utils.py          # Core detection functions
├── io_utils.py             # Medical image I/O utilities
├── mask_check.py           # Single file processing CLI
├── mask_check_batch.py     # Batch processing CLI
├── tests/                  # Test files (to be created)
│   ├── test_check_utils.py
│   ├── test_io_utils.py
│   └── test_cli.py
├── data/                   # Test data (gitignored)
├── figure/                 # Generated figures (gitignored)
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── setup.py                # Package setup
└── LICENSE                 # MIT License
```

## Testing Guidelines

### Writing Tests

- Write tests for new features and bug fixes
- Use descriptive test names
- Follow the Arrange-Act-Assert pattern
- Test both success and failure cases
- Mock external dependencies when appropriate

### Example Test

```python
import pytest
import numpy as np
from check_utils import detect_2d_noise

def test_detect_2d_noise_with_clean_mask():
    """Test that clean masks return is_valid=True"""
    clean_mask = np.ones((10, 64, 64), dtype=np.uint8)
    report = detect_2d_noise(clean_mask, min_area_threshold=10)
    assert report["is_valid"] == True

def test_detect_2d_noise_with_noisy_mask():
    """Test that noisy masks detect noise correctly"""
    noisy_mask = np.ones((10, 64, 64), dtype=np.uint8)
    noisy_mask[5, 10:15, 10:15] = 1  # Small noise component
    report = detect_2d_noise(noisy_mask, min_area_threshold=50)
    assert report["is_valid"] == False
    assert report["summary"]["noise_count"] > 0
```

## Code Review Process

1. **Automated checks**: All PRs must pass CI tests
2. **Manual review**: Maintainers will review code for:
   - Adherence to coding standards
   - Documentation completeness
   - Test coverage
   - Performance considerations
   - Backward compatibility
3. **Feedback**: Address review comments promptly
4. **Approval**: PR will be merged after approval

## Questions or Need Help?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues and discussions first

## License

By contributing to RadMaskQC, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to RadMaskQC! 🎉
