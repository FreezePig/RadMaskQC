#!/usr/bin/env python3
"""
Batch Processing Script for Medical Image Segmentation Quality Checking.

This script processes multiple patient directories in batch, running comprehensive
quality checks on segmentation masks.

Directory Structure:
    root_folder/
        patient_001/
            ct/ct.nii.gz           (or specified CT path)
            mask/mask.nii.gz       (or specified mask path)
        patient_002/
            ct/ct.nii.gz
            mask/mask.nii.gz
        ...

Usage:
    python mask_check_batch.py <root_folder> --ct-path <ct_path> --mask-path <mask_path> [options]

Examples:
    # Batch process all patients
    python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz

    # With custom output subfolder and parameters
    python mask_check_batch.py ./data --ct-path ct/ct.nii.gz --mask-path mask/mask.nii.gz -o qc_results --aspect-ratio 6.0
"""

import argparse
import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import traceback

# Import all detection functions from mask_clean
from check_utils import (
    check_3d_connectivity,
    detect_2d_noise,
    detect_elongated_projections,
    detect_sharp_concavities,
    detect_ct_value_anomalies,
    detect_internal_holes
)


def process_single_case(
    patient_dir: str,
    ct_path: str,
    mask_path: str,
    output_subdir: str = "qc_results",
    max_components: int = 1,
    connectivity: int = 3,
    min_area: int = 10,
    aspect_ratio: float = 5.0,
    convexity: float = 0.85,
    angle_threshold: float = 30.0,
    distance_threshold: float = 5.0,
    max_area: int = 20,
    threshold_air: float = 20,
    threshold_soft: float = 50,
    min_volume: int = 50,
    z_score: float = 2.0,
    visualize: bool = True,
    stop_on_error: bool = True
) -> Dict[str, Any]:
    """
    Process a single patient case with all quality checks.

    Args:
        patient_dir (str): Path to patient directory
        ct_path (str): Relative path to CT file from patient directory
        mask_path (str): Relative path to mask file from patient directory
        output_subdir (str): Subdirectory name for output (within patient directory)
        ... (other detection parameters)

    Returns:
        dict: Processing result with status and reports
    """
    patient_name = os.path.basename(patient_dir)

    # Construct full paths
    ct_file = os.path.join(patient_dir, ct_path)
    mask_file = os.path.join(patient_dir, mask_path)
    output_dir = os.path.join(patient_dir, output_subdir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    result = {
        "patient": patient_name,
        "patient_dir": patient_dir,
        "mask_file": mask_file,
        "ct_file": ct_file,
        "status": "unknown",
        "checks_completed": [],
        "checks_skipped": [],
        "error": None,
        "reports": {}
    }

    # Check if required files exist
    if not os.path.exists(mask_file):
        result["status"] = "error"
        result["error"] = f"Mask file not found: {mask_file}"
        return result

    has_ct = os.path.exists(ct_file)
    if not has_ct:
        print(f"  ⚠ CT file not found: {ct_file} - skipping CT-based checks")

    # Helper function to save individual report
    def save_individual_report(report_name, report):
        output_file = os.path.join(output_dir, f"{report_name}_report.json")
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

    try:
        # 1. Check 3D connectivity
        print(f"  [1/6] Checking 3D connectivity...")
        connectivity_report = check_3d_connectivity(
            mask_file,
            max_components=max_components,
            connectivity=connectivity
        )
        result["reports"]["connectivity"] = connectivity_report
        result["checks_completed"].append("connectivity")
        save_individual_report("connectivity", connectivity_report)

        if not connectivity_report["is_valid"]:
            print(f"  ⚠ Error in 3D connectivity check")
            if stop_on_error:
                result["status"] = "failed"
                return result
        else:
            # 2. Detect 2D noise
            print(f"  [2/6] Detecting 2D noise...")
            noise_2d_report = detect_2d_noise(
                mask_file,
                output_dir=output_dir,
                min_area_threshold=min_area,
                file_prefix=""
            )
            result["reports"]["2d_noise"] = noise_2d_report
            result["checks_completed"].append("2d_noise")
            save_individual_report("2d_noise", noise_2d_report)

            if not noise_2d_report["is_valid"]:
                print(f"  ⚠ Error in 2D noise detection")
                if stop_on_error:
                    result["status"] = "failed"
                    return result
            else:
                # 3. Detect elongated projections
                print(f"  [3/6] Detecting elongated projections...")
                elongated_report = detect_elongated_projections(
                    mask_file,
                    output_dir=output_dir,
                    file_prefix="",
                    aspect_ratio_threshold=aspect_ratio,
                    convexity_threshold=convexity,
                    visualize=visualize
                )
                result["reports"]["elongated"] = elongated_report
                result["checks_completed"].append("elongated")
                save_individual_report("elongated", elongated_report)

                if not elongated_report["is_valid"]:
                    print(f"  ⚠ Error in elongated projections detection")
                    if stop_on_error:
                        result["status"] = "failed"
                        return result
                else:
                    # 4. Detect sharp concavities
                    print(f"  [4/6] Detecting sharp concavities...")
                    concavities_report = detect_sharp_concavities(
                        mask_file,
                        output_dir=output_dir,
                        file_prefix="",
                        sharp_angle_threshold=angle_threshold,
                        distance_threshold=distance_threshold,
                        visualize=visualize
                    )
                    result["reports"]["concavities"] = concavities_report
                    result["checks_completed"].append("concavities")
                    save_individual_report("concavities", concavities_report)

                    if not concavities_report["is_valid"]:
                        print(f"  ⚠ Error in sharp concavities detection")
                        if stop_on_error:
                            result["status"] = "failed"
                            return result
                    else:
                        # 5. Detect internal holes (if CT available)
                        if has_ct:
                            print(f"  [5/6] Detecting internal holes...")
                            holes_report = detect_internal_holes(
                                mask_file,
                                ct_data=ct_file,
                                output_dir=output_dir,
                                file_prefix="",
                                max_hole_area=max_area,
                                threshold_air=threshold_air,
                                threshold_soft=threshold_soft,
                                visualize=visualize
                            )
                            result["reports"]["holes"] = holes_report
                            result["checks_completed"].append("holes")
                            save_individual_report("holes", holes_report)

                            if not holes_report["is_valid"]:
                                print(f"  ⚠ Error in internal holes detection")
                                if stop_on_error:
                                    result["status"] = "failed"
                                    return result
                        else:
                            result["checks_skipped"].append("holes")
                            print(f"  [5/6] Skipping internal holes (no CT)")

                        # 6. Detect CT anomalies (if CT available)
                        if has_ct:
                            print(f"  [6/6] Detecting CT anomalies...")
                            ct_report = detect_ct_value_anomalies(
                                mask_file,
                                ct_data=ct_file,
                                output_dir=output_dir,
                                file_prefix="",
                                min_leak_volume=min_volume,
                                z_score_threshold=z_score,
                                visualize=visualize
                            )
                            result["reports"]["ct_anomalies"] = ct_report
                            result["checks_completed"].append("ct_anomalies")
                            save_individual_report("ct_anomalies", ct_report)

                            if not ct_report["is_valid"]:
                                print(f"  ⚠ Error in CT anomalies detection")
                                if stop_on_error:
                                    result["status"] = "failed"
                                    return result
                        else:
                            result["checks_skipped"].append("ct_anomalies")
                            print(f"  [6/6] Skipping CT anomalies (no CT)")

        # Determine overall status
        all_valid = all(report.get("is_valid", True) for report in result["reports"].values())
        result["status"] = "passed" if all_valid else "warning"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()

    return result


def main():
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch processing for medical image segmentation quality checking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "root_folder",
        help="Root folder containing all patient directories"
    )
    parser.add_argument(
        "--ct-path",
        required=True,
        help="Relative path to CT file from patient directory (e.g., 'ct/ct.nii.gz')"
    )
    parser.add_argument(
        "--mask-path",
        required=True,
        help="Relative path to mask file from patient directory (e.g., 'mask/mask.nii.gz')"
    )
    parser.add_argument(
        "-o", "--output",
        default="qc_results",
        help="Output subdirectory name (within each patient folder, default: qc_results)"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining checks even if one check fails (default: stop on error)"
    )
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Skip to next patient if an error occurs (default: continue with next check)"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary report at the end"
    )

    # Detection parameters
    parser.add_argument("--max-components", type=int, default=1,
                       help="Maximum allowed 3D components (default: 1)")
    parser.add_argument("--connectivity", type=int, default=3, choices=[1, 2, 3],
                       help="Connectivity type: 1=6-conn, 2=18-conn, 3=26-conn (default: 3)")
    parser.add_argument("--min-area", type=int, default=10,
                       help="Minimum area threshold for 2D noise detection (default: 10)")
    parser.add_argument("--aspect-ratio", type=float, default=5.0,
                       help="Aspect ratio threshold (default: 5.0)")
    parser.add_argument("--convexity", type=float, default=0.85,
                       help="Convexity threshold (default: 0.85)")
    parser.add_argument("--angle-threshold", type=float, default=30.0,
                       help="Sharp angle threshold in degrees (default: 30.0)")
    parser.add_argument("--distance-threshold", type=float, default=5.0,
                       help="Defect depth threshold in pixels (default: 5.0)")
    parser.add_argument("--max-area", type=int, default=20,
                       help="Maximum hole area for noise classification (default: 20)")
    parser.add_argument("--threshold-air", type=float, default=20,
                       help="Percentile threshold for air (default: 20)")
    parser.add_argument("--threshold-soft", type=float, default=50,
                       help="Percentile threshold for soft tissue (default: 50)")
    parser.add_argument("--min-volume", type=int, default=50,
                       help="Minimum anomaly region size in voxels (default: 50)")
    parser.add_argument("--z-score", type=float, default=2.0,
                       help="Z-score threshold (default: 2.0)")
    parser.add_argument("--no-viz", dest="visualize", action="store_false", default=True,
                       help="Disable visualization")

    args = parser.parse_args()

    # Validate root folder
    root_folder = args.root_folder
    if not os.path.isdir(root_folder):
        print(f"Error: Root folder not found: {root_folder}")
        sys.exit(1)

    # Find all patient directories (subdirectories in root folder)
    patient_dirs = [
        os.path.join(root_folder, d)
        for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ]

    if not patient_dirs:
        print(f"Error: No patient directories found in {root_folder}")
        sys.exit(1)

    patient_dirs.sort()  # Process in alphabetical order

    print("=" * 70)
    print("BATCH QUALITY CHECKING")
    print("=" * 70)
    print(f"Root folder: {root_folder}")
    print(f"CT path: {args.ct_path}")
    print(f"Mask path: {args.mask_path}")
    print(f"Output subdirectory: {args.output}")
    print(f"Total patients: {len(patient_dirs)}")
    print("=" * 70)

    # Process each patient
    all_results = []
    summary = {
        "total": len(patient_dirs),
        "passed": 0,
        "warning": 0,
        "failed": 0,
        "error": 0,
        "details": []
    }

    for i, patient_dir in enumerate(patient_dirs, 1):
        patient_name = os.path.basename(patient_dir)
        print(f"\n[{i}/{len(patient_dirs)}] Processing: {patient_name}")
        print("-" * 70)

        result = process_single_case(
            patient_dir=patient_dir,
            ct_path=args.ct_path,
            mask_path=args.mask_path,
            output_subdir=args.output,
            max_components=args.max_components,
            connectivity=args.connectivity,
            min_area=args.min_area,
            aspect_ratio=args.aspect_ratio,
            convexity=args.convexity,
            angle_threshold=args.angle_threshold,
            distance_threshold=args.distance_threshold,
            max_area=args.max_area,
            threshold_air=args.threshold_air,
            threshold_soft=args.threshold_soft,
            min_volume=args.min_volume,
            z_score=args.z_score,
            visualize=args.visualize,
            stop_on_error=not args.continue_on_error
        )

        all_results.append(result)

        # Update summary
        status = result["status"]
        if status == "passed":
            summary["passed"] += 1
            print(f"  ✓ Status: PASSED")
        elif status == "warning":
            summary["warning"] += 1
            print(f"  ⚠ Status: WARNING (some checks failed)")
        elif status == "failed":
            summary["failed"] += 1
            print(f"  ✗ Status: FAILED")
        elif status == "error":
            summary["error"] += 1
            print(f"  ✗ Status: ERROR")
            if result["error"]:
                print(f"  Error: {result['error']}")

        summary["details"].append({
            "patient": patient_name,
            "status": status,
            "checks_completed": result["checks_completed"],
            "checks_skipped": result["checks_skipped"]
        })

        # Skip to next patient if requested
        if args.skip_on_error and status in ["failed", "error"]:
            print(f"  → Skipping to next patient due to error")
            continue

    # Print summary
    print("\n" + "=" * 70)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total patients: {summary['total']}")
    print(f"✓ Passed: {summary['passed']}")
    print(f"⚠ Warning: {summary['warning']}")
    print(f"✗ Failed: {summary['failed']}")
    print(f"✗ Error: {summary['error']}")
    print("=" * 70)

    # Save summary to JSON
    summary_file = os.path.join(root_folder, "batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump({
            "summary": summary,
            "results": all_results
        }, f, indent=2)
    print(f"\nBatch summary saved to: {summary_file}")

    # Detailed summary if requested
    if args.summary:
        print("\n" + "=" * 70)
        print("DETAILED RESULTS")
        print("=" * 70)
        for detail in summary["details"]:
            status_symbol = {
                "passed": "✓",
                "warning": "⚠",
                "failed": "✗",
                "error": "✗"
            }.get(detail["status"], "?")
            print(f"{status_symbol} {detail['patient']}: {detail['status'].upper()}")
            print(f"    Completed: {', '.join(detail['checks_completed']) if detail['checks_completed'] else 'None'}")
            if detail['checks_skipped']:
                print(f"    Skipped: {', '.join(detail['checks_skipped'])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
