import cv2
import numpy as np
import os
from typing import Union, Dict, Any, List, Tuple, Optional
from scipy import ndimage as ndi
from io_utils import load_medical_image


def detect_2d_noise(
    input_data: Union[str, np.ndarray],
    output_dir: Optional[str] = None,
    min_area_threshold: int = 10,
    box_color: Tuple[int, int, int] = (0, 0, 255),
    box_thickness: int = 2,
    box_padding: int = 5,
    file_prefix: str = "",
) -> Dict[str, Any]:
    """
    Detects 2D noise areas in mask slices and optionally visualizes them with bounding boxes.

    This function focuses ONLY on noise detection on 2D slices. For 3D connectivity checking
    (detecting multiple disconnected organs), use check_3d_connectivity() instead.

    Args:
        input_data (str or np.ndarray): Path to the medical image file (.npy, .nii, .nii.gz, .dcm)
                                        or the numpy array itself.
                                        Expected shape: (Depth, Height, Width) or (H, W).
        output_dir (str, optional): Directory to save visualization PNGs. If None, only analyzes
                                   without saving images. Default is None.
        min_area_threshold (int): Connected components smaller than this area (pixels)
                                  will be flagged as noise. Default is 10.
        box_color (Tuple[int, int, int]): BGR color for bounding boxes. Default is red (0, 0, 255).
        box_thickness (int): Thickness of bounding box lines. Default is 2.
        box_padding (int): Extra padding around bounding boxes. Default is 5 pixels.
        file_prefix (str): Prefix for output PNG filenames. Default is "".

    Returns:
        dict: Report with 2D noise detection results:
            {
                "is_valid": bool,              # True if no noise found
                "total_slices": int,           # Total number of slices processed
                "noise_slices": list,          # List of slice indices with noise
                "total_noise_regions": int,    # Total number of noise regions found
                "saved_images": int,           # Number of PNG images saved (0 if no output_dir)
                "output_dir": str or None,     # Directory where images were saved
                "details": dict                # Key: slice_index, Value: noise region info
            }
            details[slice_index] contains:
            {
                "noise_regions": list,         # List of noise region info with bbox and area
                "num_noise_regions": int       # Number of noise regions in this slice
            }

    Raises:
        FileNotFoundError: If input file path doesn't exist.
        ValueError: If file loading fails.
        TypeError: If input_data is not str or np.ndarray.

    Examples:
        >>> # Detect noise only (no visualization)
        >>> report = detect_2d_noise("mask.npy")
        >>>
        >>> # Detect noise and visualize with bounding boxes
        >>> report = detect_2d_noise("mask.npy", output_dir="./viz", file_prefix="patient_001")
        >>>
        >>> # Check results
        >>> if not report["is_valid"]:
        ...     print(f"Found noise in {len(report['noise_slices'])} slices")
        ...     print(f"Total noise regions: {report['total_noise_regions']}")
    """
    # ========== 1. Load Data ==========
    # Load mask (supports .npy, .nii, .nii.gz, .dcm, or numpy array)
    if isinstance(input_data, np.ndarray):
        mask = input_data.copy() if output_dir else input_data
    elif isinstance(input_data, str):
        try:
            mask = load_medical_image(input_data, return_meta=False, reorient=True)
            print(mask.shape)
        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")
    else:
        raise TypeError("input_data must be a file path (str) or a numpy array.")

    # ========== 2. Standardize to 3D (Depth, Height, Width) ==========
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]

    # ========== 3. Create output directory if needed ==========
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ========== 4. Initialize report ==========
    report = {
        "is_valid": True,
        "total_slices": mask.shape[0],
        "noise_slices": [],
        "total_noise_regions": 0,
        "saved_images": 0,
        "output_dir": output_dir,
        "details": {},
    }

    # ========== 5. Process each slice ==========
    for z_index in range(mask.shape[0]):
        # Extract slice
        slice_img = mask[z_index, :, :]

        # Binarize
        slice_uint8 = np.zeros_like(slice_img, dtype=np.uint8)
        slice_uint8[slice_img > 0] = 255

        # Skip completely empty slices
        if np.sum(slice_uint8) == 0:
            continue

        # Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            slice_uint8, connectivity=8
        )

        # Create color image for visualization if needed
        slice_color = None
        if output_dir is not None:
            slice_color = cv2.cvtColor(slice_uint8, cv2.COLOR_GRAY2BGR)

        # ========== Detect Noise Regions ==========
        noise_regions = []

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]

            if area < min_area_threshold:
                # Extract bounding box info
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]

                # Add padding and ensure bounds
                img_h, img_w = slice_img.shape[:2]
                x1 = max(0, x - box_padding)
                y1 = max(0, y - box_padding)
                x2 = min(img_w, x + width + box_padding)
                y2 = min(img_h, y + height + box_padding)

                noise_region_info = {
                    "region_id": i,
                    "area": area,
                    "bbox": (x1, y1, x2, y2),
                    "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                }
                noise_regions.append(noise_region_info)

                # Draw bounding box if visualization is enabled
                if slice_color is not None:
                    cv2.rectangle(
                        slice_color, (x1, y1), (x2, y2), box_color, box_thickness
                    )
                    label_text = f"A:{area}"
                    cv2.putText(
                        slice_color,
                        label_text,
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        box_color,
                        1,
                    )

        # ========== Update report if noise found ==========
        if noise_regions:
            report["is_valid"] = False
            report["noise_slices"].append(z_index)
            report["total_noise_regions"] += len(noise_regions)
            report["details"][z_index] = {
                "noise_regions": noise_regions,
                "num_noise_regions": len(noise_regions),
            }

            # Save visualization if output_dir is provided
            if output_dir is not None and slice_color is not None:
                filename = f"{file_prefix}_slice_{z_index:04d}_noise.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, slice_color)
                report["saved_images"] += 1

    return report


def check_3d_connectivity(
    input_data: Union[str, np.ndarray], max_components: int = 1, connectivity: int = 3
) -> Dict[str, Any]:
    """
    Check if a mask contains multiple independent 3D ROIs (Regions of Interest).

    Uses 3D connected component analysis to detect if there are multiple disconnected
    foreground regions in the mask. This helps identify cases where a segmentation
    mask accidentally contains multiple separate organs/structures instead of one.

    Common scenarios:
    - Single ROI (expected): num_components = 1
    - Multiple organs (issue): num_components > 1
    - Organs + noise debris (issue): num_components > 1

    Args:
        input_data (str or np.ndarray): Path to medical image file (.npy, .nii, .nii.gz, .dcm)
                                        or numpy array.
                                        Expected shape: (Depth, Height, Width).
        max_components (int): Maximum allowed number of 3D connected components.
                             Default is 1 (single ROI assumption).
        connectivity (int): Type of connectivity for 3D analysis.
                           1 = faces only (6-connectivity)
                           2 = faces + edges (18-connectivity)
                           3 = faces + edges + corners (26-connectivity)
                           Default is 3 (most permissive).

    Returns:
        dict: Report with 3D connectivity analysis:
            {
                "is_valid": bool,              # True if number of components <= max_components
                "total_voxels": int,           # Total foreground voxels in mask
                "num_3d_components": int,      # Number of 3D connected components found
                "component_volumes": list,     # List of all component volumes (sorted descending)
                "volume_percentages": list,    # Percentage of total volume for each component
                "largest_component_volume": int,  # Volume of largest component
                "largest_component_percentage": float,  # Percentage of largest component
                "max_components_allowed": int,  # Maximum allowed components
                "connectivity_type": int,      # Connectivity type used
                "details": dict                # Detailed info for each component
            }

    Examples:
        >>> # Check if mask contains only one ROI
        >>> report = check_3d_connectivity("mask.npy")
        >>>
        >>> # Allow up to 2 ROIs (e.g., bilateral organs)
        >>> report = check_3d_connectivity("mask.npy", max_components=2)
        >>>
        >>> # Check results
        >>> if not report["is_valid"]:
        ...     print(f"Found {report['num_3d_components']} ROIs (expected <= {report['max_components_allowed']})")
        ...     for i, vol in enumerate(report['component_volumes'], 1):
        ...         print(f"  ROI {i}: {vol} voxels ({report['volume_percentages'][i-1]:.1f}%)")
    """
    from scipy import ndimage as ndi

    # ========== 1. Load Data ==========
    if isinstance(input_data, np.ndarray):
        mask = input_data.copy()
    elif isinstance(input_data, str):
        try:
            mask = load_medical_image(input_data, return_meta=False, reorient=True)
        except Exception as e:
            raise ValueError(f"Failed to load file: {str(e)}")
    else:
        raise TypeError("input_data must be a file path (str) or a numpy array.")

    # ========== 2. Standardize to 3D ==========
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]

    # ========== 3. Binarize ==========
    mask_binary = (mask > 0).astype(np.int32)

    total_voxels = np.sum(mask_binary)

    if total_voxels == 0:
        return {
            "is_valid": False,
            "error": "Empty mask - no foreground voxels found",
            "total_voxels": 0,
            "num_3d_components": 0,
        }

    # ========== 4. 3D Connected Component Analysis ==========
    # structure defines connectivity:
    # - connectivity=1: faces only (6-connectivity)
    # - connectivity=2: faces + edges (18-connectivity)
    # - connectivity=3: faces + edges + corners (26-connectivity)

    if connectivity == 1:
        structure = ndi.generate_binary_structure(rank=3, connectivity=1)
    elif connectivity == 2:
        structure = ndi.generate_binary_structure(rank=3, connectivity=2)
    else:  # connectivity == 3 or default
        structure = ndi.generate_binary_structure(rank=3, connectivity=3)

    # Label connected components in 3D
    labeled_mask, num_components = ndi.label(mask_binary, structure=structure)

    # ========== 5. Analyze each component ==========
    component_volumes = []
    component_details = {}

    for component_id in range(1, num_components + 1):
        # Get volume of this component
        component_volume = np.sum(labeled_mask == component_id)
        component_volumes.append(component_volume)

        # Get bounding box
        indices = np.where(labeled_mask == component_id)
        min_z, max_z = indices[0].min(), indices[0].max()
        min_y, max_y = indices[1].min(), indices[1].max()
        min_x, max_x = indices[2].min(), indices[2].max()

        # Get centroid
        centroid_z = np.mean(indices[0])
        centroid_y = np.mean(indices[1])
        centroid_x = np.mean(indices[2])

        component_details[component_id] = {
            "volume": component_volume,
            "percentage": float(component_volume / total_voxels * 100),
            "bbox": {
                "z_range": (int(min_z), int(max_z)),
                "y_range": (int(min_y), int(max_y)),
                "x_range": (int(min_x), int(max_x)),
            },
            "centroid": {
                "z": float(centroid_z),
                "y": float(centroid_y),
                "x": float(centroid_x),
            },
        }

    # Sort by volume (descending)
    component_volumes_sorted = sorted(component_volumes, reverse=True)
    volume_percentages = [vol / total_voxels * 100 for vol in component_volumes_sorted]
    largest_component_volume = component_volumes_sorted[0] if component_volumes else 0
    largest_component_percentage = volume_percentages[0] if volume_percentages else 0

    # ========== 6. Build report ==========
    report = {
        "is_valid": num_components <= max_components,
        "total_voxels": total_voxels,
        "num_3d_components": num_components,
        "component_volumes": component_volumes_sorted,
        "volume_percentages": volume_percentages,
        "largest_component_volume": largest_component_volume,
        "largest_component_percentage": largest_component_percentage,
        "max_components_allowed": max_components,
        "connectivity_type": connectivity,
        "details": component_details,
    }

    # Add issues if invalid
    if not report["is_valid"]:
        report["issues"] = [
            f"Found {num_components} 3D connected components (expected <= {max_components})",
            f"Component volumes: {component_volumes_sorted}",
            f"Volume percentages: {[f'{p:.1f}%' for p in volume_percentages]}",
        ]

    return report


def detect_elongated_projections(
    mask_data: Union[str, np.ndarray],
    output_dir: Optional[str] = None,
    file_prefix: str = "",
    aspect_ratio_threshold: float = 5.0,
    convexity_threshold: float = 0.85,
    visualize: bool = True,
    leak_color: Tuple[int, int, int] = (0, 0, 255),
) -> Dict[str, Any]:
    """
    Detect elongated projection leakage regions in medical segmentation masks.

    Elongated projections occur when the mask body is rounded but has elongated
    regions extending to surrounding tissues, indicating potential segmentation leakage.

    Args:
        mask_data (str or np.ndarray): Path to mask file (.npy, .nii, .nii.gz, .dcm) or numpy array.
        output_dir (str, optional): Output directory for saving visualization results. If None, no saving.
        file_prefix (str): Output file prefix. Default is "".
        aspect_ratio_threshold (float): Aspect ratio threshold above which shape is considered elongated.
        convexity_threshold (float): Convexity threshold below which suggests shape anomaly.
        visualize (bool): Whether to generate visualizations. Default True.
        leak_color (Tuple[int, int, int]): Leakage marking color (BGR). Default red (0,0,255).

    Returns:
        dict: Detection report containing:
            {
                "is_valid": bool,              # True if no leakage detected
                "total_leak_regions": int,     # Total number of leakage regions
                "leak_regions": list,          # Details of leakage regions
                "summary": dict,               # Summary statistics
                "output_dir": str or None,     # Output directory
                "visualizations": int          # Number of saved visualizations
            }

    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If data loading fails.
        TypeError: If input data type is incorrect.
    """
    # ========== 1. Load Data ==========
    if isinstance(mask_data, np.ndarray):
        mask = mask_data.copy() if output_dir else mask_data
    elif isinstance(mask_data, str):
        try:
            mask = load_medical_image(mask_data, return_meta=False, reorient=True)
        except Exception as e:
            raise ValueError(f"Failed to load mask file: {str(e)}")
    else:
        raise TypeError("mask_data must be a file path (str) or numpy array.")

    # Standardize to 3D
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]

    # ========== 2. Create output directory ==========
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ========== 3. Initialize report ==========
    report = {
        "is_valid": True,
        "total_leak_regions": 0,
        "leak_regions": [],
        "summary": {"total_leak_voxels": 0, "affected_slices": []},
        "output_dir": output_dir,
        "visualizations": 0,
    }

    # ========== 4. Process slice by slice ==========
    for z_index in range(mask.shape[0]):
        slice_mask = mask[z_index, :, :]

        # Binarize
        mask_binary = (slice_mask > 0).astype(np.uint8)

        # Skip empty slices
        if np.sum(mask_binary) == 0:
            continue

        # Create visualization image
        viz_img = None
        if output_dir is not None and visualize:
            viz_img = cv2.cvtColor(mask_binary * 255, cv2.COLOR_GRAY2BGR)

        # Detect leak regions in this slice
        slice_leak_regions = []

        # Extract contours
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        for contour_idx, contour in enumerate(contours):
            # Calculate contour area
            area = cv2.contourArea(contour)
            if area == 0:
                continue

            # Get rotated bounding box (minimum area rectangle)
            rotated_rect = cv2.minAreaRect(contour)
            w, h = rotated_rect[1]

            # Calculate aspect ratio using true dimensions (not axis-aligned)
            aspect_ratio = float(max(w, h)) / (min(w, h) + 1e-6)

            # Detect elongated projection
            if aspect_ratio > aspect_ratio_threshold:
                # Further verification: calculate convex hull and convexity
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)

                if hull_area > 0:
                    convexity = area / hull_area

                    # If convexity is also low, more likely to be leakage
                    if convexity < convexity_threshold:
                        box_points = cv2.boxPoints(rotated_rect).astype(int)
                        leak_info = {
                            "slice": z_index,
                            "contour_id": contour_idx,
                            "bbox": box_points.tolist(),
                            "area": area,
                            "aspect_ratio": aspect_ratio,
                            "convexity": convexity,
                            "type": "elongated_projection",
                        }
                        slice_leak_regions.append(leak_info)

                        # Visualization
                        if viz_img is not None:
                            cv2.drawContours(viz_img, [box_points], 0, leak_color, 1)
                            label = f"AR:{aspect_ratio:.1f}"
                            cv2.putText(
                                viz_img,
                                label,
                                (box_points[0][0], max(0, box_points[0][1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                leak_color,
                                1,
                            )

        # ========== 5. Update report ==========
        if slice_leak_regions:
            report["is_valid"] = False
            report["total_leak_regions"] += len(slice_leak_regions)
            report["leak_regions"].extend(slice_leak_regions)
            report["summary"]["total_leak_voxels"] += sum(
                r["area"] for r in slice_leak_regions
            )
            report["summary"]["affected_slices"].append(z_index)

            # Save visualization
            if output_dir is not None and visualize and viz_img is not None:
                filename = f"{file_prefix}_slice_{z_index:04d}_elongated.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, viz_img)
                report["visualizations"] += 1

    return report


def detect_sharp_concavities(
    mask_data: Union[str, np.ndarray],
    output_dir: Optional[str] = None,
    file_prefix: str = "",
    sharp_angle_threshold: float = 30.0,
    distance_threshold: float = 5.0,
    visualize: bool = True,
    defect_color: Tuple[int, int, int] = (255, 0, 255),
) -> Dict[str, Any]:
    """
    Detect sharp concave/convex defects in medical segmentation masks.

    Identifies abnormal sharp shapes in contours by analyzing convexity defects.
    Sharp concavities may indicate segmentation issues or irregular anatomical structures.

    Args:
        mask_data (str or np.ndarray): Path to mask file (.npy, .nii, .nii.gz, .dcm) or numpy array.
        output_dir (str, optional): Output directory for saving visualization results. If None, no saving.
        file_prefix (str): Output file prefix. Default is "".
        sharp_angle_threshold (float): Angle threshold in degrees below which defect is considered sharp.
        visualize (bool): Whether to generate visualizations. Default True.
        defect_color (Tuple[int, int, int]): Defect marking color (BGR). Default purple (255,0,255).

    Returns:
        dict: Detection report containing:
            {
                "is_valid": bool,              # True if no defects detected
                "total_defect_points": int,    # Total number of defect points
                "defect_points": list,         # Details of defect points
                "summary": dict,               # Summary statistics
                "output_dir": str or None,     # Output directory
                "visualizations": int          # Number of saved visualizations
            }

    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If data loading fails.
        TypeError: If input data type is incorrect.
    """
    # ========== 1. Load Data ==========
    if isinstance(mask_data, np.ndarray):
        mask = mask_data.copy() if output_dir else mask_data
    elif isinstance(mask_data, str):
        try:
            mask = load_medical_image(mask_data, return_meta=False, reorient=True)
        except Exception as e:
            raise ValueError(f"Failed to load mask file: {str(e)}")
    else:
        raise TypeError("mask_data must be a file path (str) or numpy array.")

    # Standardize to 3D
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]

    # ========== 2. Create output directory ==========
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ========== 3. Initialize report ==========
    report = {
        "is_valid": True,
        "total_defect_points": 0,
        "defect_points": [],
        "shape_defects": {},
        "summary": {"affected_slices": []},
        "output_dir": output_dir,
        "visualizations": 0,
    }

    # ========== 4. Process slice by slice ==========
    for z_index in range(mask.shape[0]):
        slice_mask = mask[z_index, :, :]

        # Binarize
        mask_binary = (slice_mask > 0).astype(np.uint8)

        # Skip empty slices
        if np.sum(mask_binary) == 0:
            continue

        # Create visualization image
        viz_img = None
        if output_dir is not None and visualize:
            viz_img = cv2.cvtColor(mask_binary * 255, cv2.COLOR_GRAY2BGR)

        # Detect defect points in this slice
        slice_defect_points = []

        # Extract contours
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        for contour_idx, contour in enumerate(contours):
            # Calculate convexity defects
            hull = cv2.convexHull(contour, returnPoints=False)

            if len(hull) >= 3:  # Need at least 3 points to calculate defects
                defects = cv2.convexityDefects(contour, hull)

                if defects is not None:
                    for defect_idx, defect in enumerate(defects):
                        s, e, f, d = defect[0]

                        # Get defect's start, end, and farthest points
                        start = tuple(contour[s][0])
                        end = tuple(contour[e][0])
                        far = tuple(contour[f][0])

                        # Calculate distance (OpenCV stores distance * 256)
                        distance = d / 256.0

                        # Only focus on larger defects (possibly problematic concavities)
                        if distance > distance_threshold:  # Adjustable threshold
                            # Calculate angle to detect sharpness
                            vec1 = np.array([start[0] - far[0], start[1] - far[1]])
                            vec2 = np.array([end[0] - far[0], end[1] - far[1]])

                            # Normalize
                            norm1 = np.linalg.norm(vec1)
                            norm2 = np.linalg.norm(vec2)

                            if norm1 > 0 and norm2 > 0:
                                vec1_norm = vec1 / norm1
                                vec2_norm = vec2 / norm2

                                # Calculate angle (degrees)
                                dot_product = np.clip(
                                    np.dot(vec1_norm, vec2_norm), -1.0, 1.0
                                )
                                angle = np.arccos(dot_product) * 180.0 / np.pi

                                # Detect sharp angle
                                if angle < sharp_angle_threshold:
                                    defect_info = {
                                        "slice": z_index,
                                        "contour_id": contour_idx,
                                        "defect_id": defect_idx,
                                        "position": far,
                                        "depth": distance,
                                        "angle": angle,
                                        "type": "sharp_concavity",
                                    }
                                    slice_defect_points.append(defect_info)

                                    # Visualization
                                    if viz_img is not None:
                                        cv2.circle(viz_img, far, 3, defect_color, 1)

        # ========== 5. Update report ==========
        if slice_defect_points:
            report["is_valid"] = False
            report["total_defect_points"] += len(slice_defect_points)
            report["defect_points"].extend(slice_defect_points)
            report["shape_defects"][z_index] = slice_defect_points
            report["summary"]["affected_slices"].append(z_index)

            # Save visualization
            if output_dir is not None and visualize and viz_img is not None:
                filename = f"{file_prefix}_slice_{z_index:04d}_concavities.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, viz_img)
                report["visualizations"] += 1

    return report


def detect_ct_value_anomalies(
    mask_data: Union[str, np.ndarray],
    ct_data: Optional[Union[str, np.ndarray]] = None,
    output_dir: Optional[str] = None,
    file_prefix: str = "",
    enable_ct_check: bool = True,
    min_leak_volume: int = 50,
    z_score_threshold: float = 2.0,
    visualize: bool = True,
    anomaly_color: Tuple[int, int, int] = (255, 255, 0),
) -> Dict[str, Any]:
    """
    Detect CT value jumps/anomalies within mask regions in medical segmentation.

    Identifies HU value anomaly regions appearing inside mask (e.g., sudden jump to air/bone),
    which may indicate segmentation leakage to other tissue types.

    Args:
        mask_data (str or np.ndarray): Path to mask file (.npy, .nii, .nii.gz, .dcm) or numpy array.
        ct_data (str or np.ndarray, optional): Path to CT image file or array for HU value analysis.
                                              Supports .npy, .nii, .nii.gz, .dcm formats.
        output_dir (str, optional): Output directory for saving visualization results. If None, no saving.
        file_prefix (str): Output file prefix. Default is "".
        enable_ct_check (bool): Whether to enable CT value checking. Default True.
        min_leak_volume (int): Minimum anomaly region size (voxels) to be considered leakage.
        z_score_threshold (float): Z-score threshold above which anomaly is considered significant.
        visualize (bool): Whether to generate visualizations. Default True.
        anomaly_color (Tuple[int, int, int]): Anomaly marking color (BGR). Default cyan (255,255,0).

    Returns:
        dict: Detection report containing:
            {
                "is_valid": bool,              # True if no anomalies detected
                "total_anomaly_voxels": int,   # Total number of anomaly voxels
                "anomaly_regions": list,       # Details of anomaly regions
                "ct_anomalies": dict,          # CT value anomalies for each slice
                "summary": dict,               # Summary statistics
                "output_dir": str or None,     # Output directory
                "visualizations": int          # Number of saved visualizations
            }

    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If data loading fails or dimensions do not match.
        TypeError: If input data type is incorrect.
    """
    # ========== 1. Load Data ==========
    if isinstance(mask_data, np.ndarray):
        mask = mask_data.copy() if output_dir else mask_data
    elif isinstance(mask_data, str):
        try:
            mask = load_medical_image(mask_data, return_meta=False, reorient=True)
        except Exception as e:
            raise ValueError(f"Failed to load mask file: {str(e)}")
    else:
        raise TypeError("mask_data must be a file path (str) or numpy array.")

    # Load CT image (optional)
    ct = None
    if ct_data is not None:
        if isinstance(ct_data, np.ndarray):
            ct = ct_data.copy()
        elif isinstance(ct_data, str):
            try:
                ct = load_medical_image(ct_data, return_meta=False, reorient=True)
            except Exception as e:
                raise ValueError(f"Failed to load CT file: {str(e)}")
        else:
            raise TypeError("ct_data must be a file path (str) or numpy array.")

        # Check dimension matching
        if ct.shape != mask.shape:
            raise ValueError(
                f"CT image shape {ct.shape} does not match mask shape {mask.shape}."
            )

    # Standardize to 3D
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]
    if ct is not None and ct.ndim == 2:
        ct = ct[np.newaxis, :, :]

    # ========== 2. Create output directory ==========
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ========== 3. Initialize report ==========
    report = {
        "is_valid": True,
        "total_anomaly_voxels": 0,
        "anomaly_regions": [],
        "ct_anomalies": {},
        "summary": {"affected_slices": []},
        "output_dir": output_dir,
        "visualizations": 0,
    }

    # Skip if CT data not available or check is disabled
    if ct is None or not enable_ct_check:
        return report

    # ========== 4. Process slice by slice ==========
    for z_index in range(mask.shape[0]):
        slice_mask = mask[z_index, :, :]
        slice_ct = ct[z_index, :, :]

        # Binarize
        mask_binary = (slice_mask > 0).astype(np.uint8)

        # Skip empty slices
        if np.sum(mask_binary) == 0:
            continue

        # Create visualization image
        viz_img = None
        if output_dir is not None and visualize:
            viz_img = cv2.cvtColor(mask_binary * 255, cv2.COLOR_GRAY2BGR)

        # Detect anomaly regions in this slice
        slice_anomaly_voxels = []

        # Get CT values inside mask
        mask_hu_values = slice_ct[mask_binary > 0]

        if len(mask_hu_values) == 0:
            continue

        # Calculate statistical features
        mean_hu = np.mean(mask_hu_values)
        std_hu = np.std(mask_hu_values)
        q25_hu = np.percentile(mask_hu_values, 25)
        q75_hu = np.percentile(mask_hu_values, 75)
        iqr_hu = q75_hu - q25_hu

        # Use IQR method to detect outliers (more robust than mean ± 3*std)
        lower_bound = q25_hu - 3.0 * iqr_hu
        upper_bound = q75_hu + 3.0 * iqr_hu

        # Detect CT value anomaly regions inside mask
        anomaly_mask = (mask_binary > 0) & (
            (slice_ct < lower_bound) | (slice_ct > upper_bound)
        )
        anomaly_count = np.sum(anomaly_mask)

        if anomaly_count == 0:
            continue

        # Further analysis: whether outliers form connected regions (not discrete noise)
        anomaly_uint8 = anomaly_mask.astype(np.uint8)
        num_anomaly_labels, anomaly_labels, stats, _ = cv2.connectedComponentsWithStats(
            anomaly_uint8, connectivity=8
        )

        # Check each anomaly connected region (skip label 0 which is background)
        for label_id in range(1, num_anomaly_labels):
            region_mask = anomaly_labels == label_id
            region_size = np.sum(region_mask)

            # Only focus on sufficiently large anomaly regions (possibly leakage)
            if region_size >= min_leak_volume:
                region_hu_values = slice_ct[region_mask]
                region_mean_hu = np.mean(region_hu_values)
                region_std_hu = np.std(region_hu_values)

                # Calculate deviation from overall mask mean (in standard deviations)
                z_score = abs(region_mean_hu - mean_hu) / (std_hu + 1e-6)

                # Only report if deviation is very significant
                if z_score > z_score_threshold:
                    anomaly_info = {
                        "slice": z_index,
                        "type": "hu_intensity_jump",
                        "voxel_count": int(region_size),
                        "region_mean_hu": float(region_mean_hu),
                        "region_std_hu": float(region_std_hu),
                        "global_mean_hu": float(mean_hu),
                        "global_std_hu": float(std_hu),
                        "z_score": float(z_score),
                        "hu_range": (
                            float(np.min(region_hu_values)),
                            float(np.max(region_hu_values)),
                        ),
                    }
                    slice_anomaly_voxels.append(anomaly_info)

                    # Visualization (mark CT jump regions with cyan)
                    if viz_img is not None:
                        # Create overlay for anomaly region
                        region_overlay = np.zeros_like(viz_img)
                        region_overlay[region_mask] = anomaly_color
                        viz_img = cv2.addWeighted(viz_img, 0.5, region_overlay, 0.5, 0)

                        # Add label
                        indices = np.where(region_mask)
                        if len(indices[0]) > 0:
                            center_y = int(np.mean(indices[0]))
                            center_x = int(np.mean(indices[1]))
                            label = f"HU:{int(region_mean_hu)}"
                            cv2.putText(
                                viz_img,
                                label,
                                (center_x, center_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                anomaly_color,
                                1,
                            )

        # ========== 5. Update report ==========
        if slice_anomaly_voxels:
            report["is_valid"] = False
            report["total_anomaly_voxels"] += sum(
                a["voxel_count"] for a in slice_anomaly_voxels
            )
            report["anomaly_regions"].extend(slice_anomaly_voxels)
            report["ct_anomalies"][z_index] = slice_anomaly_voxels
            report["summary"]["affected_slices"].append(z_index)

            # Save visualization
            if output_dir is not None and visualize and viz_img is not None:
                filename = f"{file_prefix}_slice_{z_index:04d}_ct_anomalies.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, viz_img)
                report["visualizations"] += 1

    return report


def detect_internal_holes(
    mask_data: Union[str, np.ndarray],
    ct_data: Union[str, np.ndarray],
    output_dir: Optional[str] = None,
    file_prefix: str = "",
    max_hole_area: int = 20,
    visualize: bool = True,
    threshold_air: float = 20,
    threshold_soft: float = 50,
    hole_color: Tuple[int, int, int] = (0, 255, 255),
) -> Dict[str, Any]:
    """
    Detect internal holes/noise (erased tissue) within foreground regions in medical segmentation masks.

    This function identifies holes inside the mask that are likely accidentally erased tumor tissue,
    using relative histogram analysis to distinguish them from normal anatomical cavities.

    Detection criteria (using relative histogram analysis):
    - Holes with mean CT value >= threshold_soft percentile → noise (erased tissue, regardless of size)
    - Holes with mean CT value <= threshold_air percentile → normal cavity (ignored)
    - Holes with mean CT value between percentiles → classified by size (small = noise, large = cavity)

    Args:
        mask_data (str or np.ndarray): Path to mask file (.npy, .nii, .nii.gz, .dcm) or numpy array.
        ct_data (str or np.ndarray): Path to CT image file or array for HU value analysis.
                                        Supports .npy, .nii, .nii.gz, .dcm formats. REQUIRED.
        output_dir (str, optional): Output directory for saving visualization results.
        file_prefix (str): Output file prefix. Default is "".
        max_hole_area (int): Maximum hole area (pixels) to be considered as noise for middle-range CT values.
                             Default 20.
        threshold_air (float): Percentile threshold for air/normal cavity. Default 20 (20th percentile).
        threshold_soft (float): Percentile threshold for soft tissue/erased tissue. Default 50 (50th percentile).
        visualize (bool): Whether to generate visualizations. Default True.
        hole_color (Tuple[int, int, int]): Hole marking color (BGR). Default yellow (0,255,255).

    Returns:
        dict: Detection report containing:
            {
                "is_valid": bool,              # True if no noise holes detected
                "total_hole_count": int,        # Total number of noise holes detected
                "noise_holes": list,           # Details of noise holes (erased tissue)
                "summary": dict,               # Summary statistics
                "output_dir": str or None,     # Output directory
                "visualizations": int          # Number of saved visualizations
            }

    Raises:
        FileNotFoundError: If file path does not exist.
        ValueError: If data loading fails or dimensions do not match.
        TypeError: If input data type is incorrect.
    """
    # ========== 1. Load Data ==========
    if isinstance(mask_data, np.ndarray):
        mask = mask_data.copy() if output_dir else mask_data
    elif isinstance(mask_data, str):
        try:
            mask = load_medical_image(mask_data, return_meta=False, reorient=True)
        except Exception as e:
            raise ValueError(f"Failed to load mask file: {str(e)}")
    else:
        raise TypeError("mask_data must be a file path (str) or numpy array.")

    # Load CT image (required)
    if isinstance(ct_data, np.ndarray):
        ct = ct_data.copy()
    elif isinstance(ct_data, str):
        try:
            ct = load_medical_image(ct_data, return_meta=False, reorient=True)
        except Exception as e:
            raise ValueError(f"Failed to load CT file: {str(e)}")
    else:
        raise TypeError("ct_data must be a file path (str) or numpy array.")

    # Check dimension matching
    if ct.shape != mask.shape:
        raise ValueError(
            f"CT image shape {ct.shape} does not match mask shape {mask.shape}."
        )

    # Standardize to 3D
    if mask.ndim == 2:
        mask = mask[np.newaxis, :, :]
    if ct.ndim == 2:
        ct = ct[np.newaxis, :, :]

    # ========== 2. Create output directory ==========
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    # ========== 3. Initialize report ==========
    report = {
        "is_valid": True,
        "total_hole_count": 0,
        "noise_holes": [],
        "summary": {"total_noise_voxels": 0, "affected_slices": []},
        "output_dir": output_dir,
        "visualizations": 0,
    }

    # ========== 4. Process slice by slice ==========
    for z_index in range(mask.shape[0]):
        slice_mask = mask[z_index, :, :]
        slice_ct = ct[z_index, :, :] if ct is not None else None

        # Binarize
        mask_binary = (slice_mask > 0).astype(np.uint8)

        # Skip empty slices
        if np.sum(mask_binary) == 0:
            continue

        # Create visualization image
        viz_img = None
        if output_dir is not None and visualize:
            viz_img = cv2.cvtColor(mask_binary * 255, cv2.COLOR_GRAY2BGR)

        # Detect holes in this slice
        slice_noise_holes = []

        # Find contours of the mask
        contours, _ = cv2.findContours(
            mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        # Create a mask filled with mask content, then find holes by subtraction
        mask_filled = cv2.drawContours(
            np.zeros_like(mask_binary), contours, 0, 1, thickness=cv2.FILLED
        )
        holes_mask = mask_binary ^ mask_filled  # XOR to find holes (0 or 1)

        # Calculate histogram statistics of the mask (excluding holes) for relative classification
        mask_no_holes = mask_binary & (~holes_mask)

        if np.sum(mask_no_holes) > 0:
            mask_hu_values = slice_ct[mask_no_holes > 0]
            percentile_air = np.percentile(mask_hu_values, threshold_air)
            percentile_soft = np.percentile(mask_hu_values, threshold_soft)
        else:
            percentile_air = 0
            percentile_soft = 0

        # Find connected components in holes
        num_holes, holes_labels = cv2.connectedComponents(holes_mask, connectivity=8)

        for hole_id in range(1, num_holes):  # Skip background (label 0)
            hole_mask = holes_labels == hole_id
            hole_area = np.sum(hole_mask)

            # Get hole's center for labeling
            indices = np.where(hole_mask)
            if len(indices[0]) == 0:
                continue
            center_y = int(np.mean(indices[0]))
            center_x = int(np.mean(indices[1]))

            # Get CT values in hole
            hole_hu_values = slice_ct[hole_mask]
            hole_mean_hu = float(np.mean(hole_hu_values))
            hole_hu_range = (
                float(np.min(hole_hu_values)),
                float(np.max(hole_hu_values)),
            )

            # Determine if this is a noise hole (erased tissue) or normal cavity
            reason = ""

            # Use relative histogram-based classification
            if hole_mean_hu <= percentile_air:
                # Hole in lower percentile → likely air/normal cavity, skip
                continue
            elif hole_mean_hu >= percentile_soft:
                # Hole in upper percentile → likely erased tissue (noise, regardless of size)
                reason = "high_percentile_noise"
            else:
                # Middle percentiles → classify by size primarily
                if hole_area < max_hole_area:
                    reason = "mid_percentile_small"
                else:
                    # Large area in middle range → likely normal cavity, skip
                    continue

            # Create hole info (only for noise holes)
            hole_info = {
                "slice": z_index,
                "hole_id": hole_id,
                "center": (center_x, center_y),
                "area": int(hole_area),
                "mean_hu": hole_mean_hu,
                "hu_range": hole_hu_range,
                "reason": reason,
                "type": "noise_hole",
            }

            # Add to noise holes list
            slice_noise_holes.append(hole_info)

            # Visualization for noise holes
            if viz_img is not None:
                cv2.circle(
                    viz_img,
                    (center_x, center_y),
                    max(3, int(np.sqrt(hole_area))),
                    hole_color,
                    1,
                )
                label = f"N:{hole_area}"
                cv2.putText(
                    viz_img,
                    label,
                    (center_x + 5, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    hole_color,
                    1,
                )

        # ========== 5. Update report ==========
        if slice_noise_holes:
            report["is_valid"] = False
            report["total_hole_count"] += len(slice_noise_holes)
            report["noise_holes"].extend(slice_noise_holes)
            report["summary"]["total_noise_voxels"] += sum(
                h["area"] for h in slice_noise_holes
            )
            report["summary"]["affected_slices"].append(z_index)

            # Save visualization
            if output_dir is not None and visualize and viz_img is not None:
                filename = f"{file_prefix}_slice_{z_index:04d}_holes.png"
                output_path = os.path.join(output_dir, filename)
                cv2.imwrite(output_path, viz_img)
                report["visualizations"] += 1

    return report
