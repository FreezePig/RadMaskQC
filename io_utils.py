"""
Medical Imaging I/O Utilities

This module provides functions to read common medical imaging file formats
(NIfTI, DICOM) and convert them to numpy arrays or save as .npy files.
"""

import numpy as np
import os
from typing import Union, Optional, Tuple
from pathlib import Path

# Optional imports for medical imaging formats
try:
    import nibabel as nib

    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("Warning: nibabel not installed. NIfTI files will not be supported.")
    print("Install with: pip install nibabel")

try:
    import pydicom
    from pydicom import dcmread

    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("Warning: pydicom not installed. DICOM files will not be supported.")
    print("Install with: pip install pydicom")


def load_medical_image(
    input_path: Union[str, Path], return_meta: bool = False, reorient: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """
    Load medical image files (NIfTI, DICOM, or NPY) into numpy arrays.

    Supports:
    - NIfTI files: .nii, .nii.gz
    - DICOM files: .dcm, .dicom (single file or folder with DICOM series)
    - NumPy files: .npy

    Args:
        input_path (str or Path): Path to the file or folder.
                                  - For NIfTI/NPY: path to the file
                                  - For DICOM: path to single .dcm file or folder containing DICOM series
        return_meta (bool): If True, returns (array, metadata) tuple.
                           If False, returns only the array. Default is False.
        reorient (bool): If True, reorient NIfTI arrays from (x,y,z) to (z,y,x) order.
                        Only affects NIfTI files. Default is True.

    Returns:
        np.ndarray or tuple: If return_meta=False, returns numpy array.
                            If return_meta=True, returns (array, metadata_dict).

    Raises:
        FileNotFoundError: If the input path doesn't exist.
        ValueError: If the file format is not supported or required libraries are missing.
        TypeError: If the input type is invalid.

    Examples:
        >>> # Load NIfTI file with reorientation (default)
        >>> ct_array = load_medical_image("patient_ct.nii.gz")
        >>> print(ct_array.shape)  # (depth, height, width)
        >>>
        >>> # Load without reorientation
        >>> ct_array = load_medical_image("patient_ct.nii.gz", reorient=False)
        >>> print(ct_array.shape)  # (x, y, z) - original NIfTI order
        >>>
        >>> # Load DICOM series from folder
        >>> ct_array = load_medical_image("./dicom_series/")
        >>>
        >>> # Load with metadata
        >>> ct_array, meta = load_medical_image("ct.nii.gz", return_meta=True)
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    # Determine file type
    suffix = input_path.suffix.lower()

    # Handle .nii.gz (double extension)
    if suffix == ".gz" and input_path.stem.endswith(".nii"):
        suffix = ".nii.gz"

    metadata = {}

    # Load based on file type
    if suffix in [".npy"]:
        # NumPy array file
        array = np.load(input_path)
        metadata["format"] = "npy"
        metadata["shape"] = array.shape
        metadata["shape_original"] = array.shape

    elif suffix in [".nii", ".nii.gz"]:
        # NIfTI file
        if not NIBABEL_AVAILABLE:
            raise ValueError(
                "nibabel is required to read NIfTI files. Install with: pip install nibabel"
            )

        nii_img = nib.load(input_path)
        array = nii_img.get_fdata()
        original_shape = array.shape

        # Reorient from (x, y, z) to (z, y, x) if requested
        if reorient:
            # Check if array is 3D
            if array.ndim == 3:
                # Transpose from (x, y, z) to (z, y, x)
                # This gives us (depth, height, width) order
                array = np.transpose(array, (2, 1, 0))
                metadata["reoriented"] = True
                metadata["original_shape"] = original_shape
                metadata["original_orientation"] = "(x, y, z)"
            else:
                metadata["reoriented"] = False
        else:
            metadata["reoriented"] = False

        # Extract metadata
        metadata["format"] = "nifti"
        metadata["shape"] = array.shape
        metadata["affine"] = nii_img.affine
        metadata["voxel_size"] = nii_img.header.get_zooms()[:3]
        metadata["header"] = nii_img.header

    elif suffix in [".dcm", ".dicom", ""]:
        # DICOM file (single) or DICOM series (folder)
        if not PYDICOM_AVAILABLE:
            raise ValueError(
                "pydicom is required to read DICOM files. Install with: pip install pydicom"
            )

        if input_path.is_file():
            # Single DICOM file
            array, metadata = _load_single_dicom(input_path)
        elif input_path.is_dir():
            # DICOM series (folder with multiple DICOM files)
            array, metadata = _load_dicom_series(input_path)
        else:
            raise FileNotFoundError(f"Invalid DICOM path: {input_path}")

    else:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .npy, .nii, .nii.gz, .dcm, .dicom folders"
        )

    if return_meta:
        return array, metadata
    else:
        return array


def _load_single_dicom(dicom_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load a single DICOM file.

    Args:
        dicom_path (Path): Path to the DICOM file.

    Returns:
        tuple: (numpy_array, metadata_dict)
    """
    dcm = dcmread(dicom_path)

    # Extract pixel array
    array = dcm.pixel_array

    # Apply rescale slope and intercept if available (convert to Hounsfield Units for CT)
    if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
        slope = getattr(dcm, "RescaleSlope", 1)
        intercept = getattr(dcm, "RescaleIntercept", 0)
        array = array.astype(np.float32) * slope + intercept

    metadata = {
        "format": "dicom",
        "shape": array.shape,
        "patient_id": getattr(dcm, "PatientID", "unknown"),
        "study_date": getattr(dcm, "StudyDate", "unknown"),
        "modality": getattr(dcm, "Modality", "unknown"),
        "series_description": getattr(dcm, "SeriesDescription", "unknown"),
    }

    return array, metadata


def _load_dicom_series(folder_path: Path) -> Tuple[np.ndarray, dict]:
    """
    Load a DICOM series from a folder.

    Reads all DICOM files in the folder, sorts them by slice position,
    and stacks them into a 3D volume.

    Args:
        folder_path (Path): Path to the folder containing DICOM files.

    Returns:
        tuple: (3D_numpy_array, metadata_dict)
    """
    # Find all DICOM files in the folder
    dicom_files = []

    for file_path in folder_path.iterdir():
        if file_path.is_file():
            try:
                # Try to read as DICOM
                dcm = dcmread(file_path, stop_before_pixels=True)
                # Check if it has the required DICOM attributes
                if hasattr(dcm, "SliceLocation") or hasattr(
                    dcm, "ImagePositionPatient"
                ):
                    dicom_files.append((file_path, dcm))
            except Exception:
                # Not a valid DICOM file, skip
                continue

    if not dicom_files:
        raise ValueError(f"No valid DICOM files found in: {folder_path}")

    # Sort by slice location
    def get_slice_position(item):
        _, dcm = item
        # Try different DICOM attributes for slice position
        if hasattr(dcm, "SliceLocation"):
            return dcm.SliceLocation
        elif hasattr(dcm, "ImagePositionPatient"):
            return float(dcm.ImagePositionPatient[2])  # Z-coordinate
        elif hasattr(dcm, "InstanceNumber"):
            return dcm.InstanceNumber
        else:
            return 0

    dicom_files.sort(key=get_slice_position)

    # Read all slices and stack them
    slices = []
    metadata = None

    for file_path, first_dcm in dicom_files:
        dcm = dcmread(file_path)
        slice_array = dcm.pixel_array

        # Apply rescale slope and intercept if available
        if hasattr(dcm, "RescaleSlope") and hasattr(dcm, "RescaleIntercept"):
            slope = getattr(dcm, "RescaleSlope", 1)
            intercept = getattr(dcm, "RescaleIntercept", 0)
            slice_array = slice_array.astype(np.float32) * slope + intercept

        slices.append(slice_array)

        # Store metadata from first file
        if metadata is None:
            metadata = {
                "format": "dicom_series",
                "num_slices": len(dicom_files),
                "shape": (len(dicom_files),) + slice_array.shape,
                "patient_id": getattr(dcm, "PatientID", "unknown"),
                "study_date": getattr(dcm, "StudyDate", "unknown"),
                "modality": getattr(dcm, "Modality", "unknown"),
                "series_description": getattr(dcm, "SeriesDescription", "unknown"),
            }

    # Stack into 3D volume
    volume = np.stack(slices, axis=0)

    return volume, metadata


def save_as_npy(
    array: np.ndarray, output_path: Union[str, Path], compress: bool = True
) -> str:
    """
    Save a numpy array as .npy file.

    Args:
        array (np.ndarray): The array to save.
        output_path (str or Path): Output file path (should end with .npy).
        compress (bool): If True, use compression. Default is True.

    Returns:
        str: The path where the file was saved.

    Raises:
        ValueError: If the output path doesn't end with .npy.

    Examples:
        >>> ct_array = load_medical_image("ct.nii.gz")
        >>> save_as_npy(ct_array, "output/ct_array.npy")
    """
    output_path = Path(output_path)

    # Ensure .npy extension
    if output_path.suffix != ".npy":
        output_path = output_path.with_suffix(".npy")

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with or without compression
    if compress:
        np.savez_compressed(output_path, data=array)
    else:
        np.save(output_path, array)

    return str(output_path)


def convert_medical_to_npy(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    compress: bool = True,
    reorient: bool = True,
) -> Union[str, np.ndarray]:
    """
    Convert medical image file (NIfTI, DICOM) to .npy format.

    This is a convenience function that combines load_medical_image()
    and save_as_npy().

    Args:
        input_path (str or Path): Path to the input file (NIfTI, DICOM, or folder).
        output_path (str or Path, optional): Output .npy path. If not provided,
                                            will use input_path with .npy extension.
        compress (bool): If True, use compression. Default is True.
        reorient (bool): If True, reorient NIfTI arrays from (x,y,z) to (z,y,x) order.
                        Only affects NIfTI files. Default is True.

    Returns:
        Union[str, np.ndarray]: The path where the .npy file was saved, or the array if no output path was provided.
    Examples:
        >>> # Convert NIfTI to NPY with reorientation (default)
        >>> convert_medical_to_npy("ct.nii.gz", "output/ct.npy")
        >>>
        >>> # Convert without reorientation
        >>> convert_medical_to_npy("ct.nii.gz", "output/ct.npy", reorient=False)
        >>>
        >>> # Convert DICOM series to NPY (auto-generates output name)
        >>> convert_medical_to_npy("./dicom_series/")
    """
    input_path = Path(input_path)

    # Load the medical image
    array = load_medical_image(input_path, return_meta=False, reorient=reorient)

    # Save as .npy
    if output_path is None:
        return array
    else:
        output_path = Path(output_path)

    saved_path = save_as_npy(array, output_path, compress=compress)

    return saved_path


def get_image_info(input_path: Union[str, Path]) -> dict:
    """
    Get information about a medical image file without loading the full array.

    Returns metadata like shape, format, voxel size, etc.

    Args:
        input_path (str or Path): Path to the file or folder.

    Returns:
        dict: Dictionary containing image information.

    Examples:
        >>> info = get_image_info("ct.nii.gz")
        >>> print(f"Shape: {info['shape']}")
        >>> print(f"Voxel size: {info['voxel_size']}")
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    suffix = input_path.suffix.lower()

    # Handle .nii.gz (double extension)
    if suffix == ".gz" and input_path.stem.endswith(".nii"):
        suffix = ".nii.gz"

    info = {"path": str(input_path)}

    if suffix in [".npy"]:
        # For NPY, we need to load to get shape
        array = np.load(input_path, mmap_mode="r")
        info["format"] = "npy"
        info["shape"] = array.shape
        info["dtype"] = str(array.dtype)
        info["size_mb"] = array.nbytes / (1024 * 1024)

    elif suffix in [".nii", ".nii.gz"]:
        if not NIBABEL_AVAILABLE:
            raise ValueError("nibabel is required to read NIfTI files.")

        nii_img = nib.load(input_path)
        info["format"] = "nifti"
        info["shape"] = nii_img.shape
        info["dtype"] = str(nii_img.get_data_dtype())
        info["voxel_size"] = nii_img.header.get_zooms()[:3]
        info["affine"] = nii_img.affine.tolist()

    elif suffix in [".dcm", ".dicom", ""]:
        if not PYDICOM_AVAILABLE:
            raise ValueError("pydicom is required to read DICOM files.")

        if input_path.is_file():
            dcm = dcmread(input_path, stop_before_pixels=True)
            info["format"] = "dicom"
            info["shape"] = (dcm.Rows, dcm.Columns)
            info["dtype"] = str(
                dcm.pixel_array.dtype if hasattr(dcm, "pixel_array") else "unknown"
            )
            info["patient_id"] = getattr(dcm, "PatientID", "unknown")
            info["modality"] = getattr(dcm, "Modality", "unknown")
        elif input_path.is_dir():
            # Count DICOM files in folder
            dicom_count = 0
            for file_path in input_path.iterdir():
                if file_path.is_file():
                    try:
                        dcmread(file_path, stop_before_pixels=True)
                        dicom_count += 1
                    except Exception:
                        continue
            info["format"] = "dicom_series"
            info["num_files"] = dicom_count

    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    return info
