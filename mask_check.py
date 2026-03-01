#!/usr/bin/env python3
"""
Command Line Interface for Medical Image Segmentation Quality Checking.

This script provides convenient command-line access to various segmentation
quality checking functions including:
- 2D noise detection
- 3D connectivity checking
- Elongated projection detection
- Sharp concavity detection
- CT value anomaly detection
- Internal hole detection

Usage:
    python mask_check_cli.py <command> [options]

Commands:
    detect-2d-noise          Detect 2D noise in mask slices
    check-connectivity      Check if mask contains multiple 3D ROIs
    detect-elongated        Detect elongated projection leakage
    detect-concavities      Detect sharp concave/convex defects
    detect-ct-anomalies     Detect CT value jumps/anomalies
    detect-holes            Detect internal holes (erased tissue)
    check-all               Run all available checks

Examples:
    # Detect 2D noise
    python mask_check_cli.py detect-2d-noise mask.nii.gz

    # Check 3D connectivity
    python mask_check_cli.py check-connectivity mask.nii.gz

    # Detect elongated projections
    python mask_check_cli.py detect-elongated mask.nii.gz -o ./output

    # Detect internal holes with CT data
    python mask_check_cli.py detect-holes mask.nii.gz ct.nii.gz -t_air 20 -t_soft 50

    # Run all checks
    python mask_check_cli.py check-all mask.nii.gz ct.nii.gz -o ./results
"""

import argparse
import sys
import json
import os
from typing import Optional
from pathlib import Path

import numpy as np

# Import all detection functions from mask_clean
from check_utils import (
    check_3d_connectivity,
    detect_2d_noise,
    detect_elongated_projections,
    detect_sharp_concavities,
    detect_ct_value_anomalies,
    detect_internal_holes,
)


def print_report(report: dict, detailed: bool = False):
    """Print detection report in a formatted way."""
    print("\n" + "=" * 60)
    print("DETECTION REPORT")
    print("=" * 60)

    # Print validation status
    status = "✓ VALID" if report.get("is_valid", True) else "✗ INVALID"
    print(f"Status: {status}")

    # Print summary statistics
    if "summary" in report:
        print("\nSummary:")
        for key, value in report["summary"].items():
            if isinstance(value, list) and value:
                print(f"  {key}: {value}")
            elif isinstance(value, (int, float)) and value > 0:
                print(f"  {key}: {value}")

    # Print detailed results if requested
    if detailed:
        print("\nDetailed Results:")
        for key, value in report.items():
            if (
                key not in ["is_valid", "summary", "output_dir", "visualizations"]
                and value
            ):
                if isinstance(value, list) and len(value) > 0:
                    print(f"\n{key} ({len(value)} items):")
                    for i, item in enumerate(value[:5], 1):  # Show first 5 items
                        print(f"  [{i}] {item}")
                    if len(value) > 5:
                        print(f"  ... and {len(value) - 5} more items")
                elif isinstance(value, dict):
                    print(f"\n{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")

    # Print visualization info
    if report.get("visualizations", 0) > 0:
        print(f"\nVisualizations saved: {report['visualizations']}")
        if report.get("output_dir"):
            print(f"Output directory: {report['output_dir']}")

    print("=" * 60 + "\n")


def cmd_detect_2d_noise(args):
    """Handle detect-2d-noise command."""
    print(f"\nDetecting 2D noise for: {args.mask}")

    report = detect_2d_noise(
        args.mask,
        output_dir=args.output,
        min_area_threshold=args.min_area,
        file_prefix=args.prefix,
    )

    print_report(report, args.detailed)

    # Save report to JSON if requested
    if args.output:
        output_file = os.path.join(args.output, "2d_noise_report.json")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")


def cmd_check_connectivity(args):
    """Handle check-connectivity command."""
    print(f"\nChecking 3D connectivity for: {args.mask}")

    report = check_3d_connectivity(
        args.mask, max_components=args.max_components, connectivity=args.connectivity
    )

    print_report(report, args.detailed)

    # Save report to JSON if requested
    if args.output:
        output_file = os.path.join(args.output, "connectivity_report.json")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")


def cmd_detect_elongated(args):
    """Handle detect-elongated command."""
    print(f"\nDetecting elongated projections for: {args.mask}")

    report = detect_elongated_projections(
        args.mask,
        output_dir=args.output,
        file_prefix=args.prefix,
        aspect_ratio_threshold=args.aspect_ratio,
        convexity_threshold=args.convexity,
        visualize=args.visualize,
    )

    print_report(report, args.detailed)

    # Save report to JSON if requested
    if args.output:
        output_file = os.path.join(args.output, "elongated_report.json")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")


def cmd_detect_concavities(args):
    """Handle detect-concavities command."""
    print(f"\nDetecting sharp concavities for: {args.mask}")

    report = detect_sharp_concavities(
        args.mask,
        output_dir=args.output,
        file_prefix=args.prefix,
        sharp_angle_threshold=args.angle_threshold,
        distance_threshold=args.distance_threshold,
        visualize=args.visualize,
    )

    print_report(report, args.detailed)

    # Save report to JSON if requested
    if args.output:
        output_file = os.path.join(args.output, "concavities_report.json")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")


def cmd_detect_ct_anomalies(args):
    """Handle detect-ct-anomalies command."""
    print(f"\nDetecting CT value anomalies for: {args.mask}")

    if args.ct is None:
        print("Error: CT image file is required for CT anomaly detection")
        sys.exit(1)

    report = detect_ct_value_anomalies(
        args.mask,
        ct_data=args.ct,
        output_dir=args.output,
        file_prefix=args.prefix,
        enable_ct_check=True,
        min_leak_volume=args.min_volume,
        z_score_threshold=args.z_score,
        visualize=args.visualize,
    )

    print_report(report, args.detailed)

    # Save report to JSON if requested
    if args.output:
        output_file = os.path.join(args.output, "ct_anomalies_report.json")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")


def cmd_detect_holes(args):
    """Handle detect-holes command."""
    print(f"\nDetecting internal holes for: {args.mask}")

    if args.ct is None:
        print("Error: CT image file is required for internal hole detection")
        sys.exit(1)

    report = detect_internal_holes(
        args.mask,
        ct_data=args.ct,
        output_dir=args.output,
        file_prefix=args.prefix,
        max_hole_area=args.max_area,
        threshold_air=args.threshold_air,
        threshold_soft=args.threshold_soft,
        visualize=args.visualize,
    )

    print_report(report, args.detailed)

    # Save report to JSON if requested
    if args.output:
        output_file = os.path.join(args.output, "holes_report.json")
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {output_file}")


def cmd_check_all(args):
    """Handle check-all command - run all available checks."""
    print(f"\nRunning comprehensive quality checks for: {args.mask}")
    print("=" * 60)

    # Prepare output directory
    if args.output:
        os.makedirs(args.output, exist_ok=True)

    all_reports = {}
    overall_valid = True

    # Helper function to save individual report
    def save_individual_report(report_name, report):
        if args.output:
            output_file = os.path.join(args.output, f"{report_name}_report.json")
            with open(output_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {output_file}")

    # 1. Check 3D connectivity
    print("\n[1/6] Checking 3D connectivity...")
    connectivity_report = check_3d_connectivity(
        args.mask, max_components=args.max_components, connectivity=args.connectivity
    )
    all_reports["connectivity"] = connectivity_report
    if not connectivity_report["is_valid"]:
        overall_valid = False
    print_report(connectivity_report, detailed=False)
    save_individual_report("connectivity", connectivity_report)
    if not connectivity_report["is_valid"]:
        print("\n⚠ Error detected in 3D connectivity check. Stopping further checks.")
        overall_valid = False
    else:
        # 2. Detect 2D noise
        print("[2/6] Detecting 2D noise...")
        noise_2d_report = detect_2d_noise(
            args.mask,
            output_dir=args.output,
            min_area_threshold=args.min_area,
            file_prefix="",
        )
        all_reports["2d_noise"] = noise_2d_report
        if not noise_2d_report["is_valid"]:
            overall_valid = False
        print_report(noise_2d_report, detailed=False)
        save_individual_report("2d_noise", noise_2d_report)
        if not noise_2d_report["is_valid"]:
            print("\n⚠ Error detected in 2D noise detection. Stopping further checks.")
        else:
            # 3. Detect elongated projections
            print("[3/6] Detecting elongated projections...")
            elongated_report = detect_elongated_projections(
                args.mask,
                output_dir=args.output,
                file_prefix="",
                aspect_ratio_threshold=args.aspect_ratio,
                convexity_threshold=args.convexity,
                visualize=args.visualize,
            )
            all_reports["elongated"] = elongated_report
            if not elongated_report["is_valid"]:
                overall_valid = False
            print_report(elongated_report, detailed=False)
            save_individual_report("elongated", elongated_report)
            if not elongated_report["is_valid"]:
                print(
                    "\n⚠ Error detected in elongated projections detection. Stopping further checks."
                )
            else:
                # 4. Detect sharp concavities
                print("[4/6] Detecting sharp concavities...")
                concavities_report = detect_sharp_concavities(
                    args.mask,
                    output_dir=args.output,
                    file_prefix="",
                    sharp_angle_threshold=args.angle_threshold,
                    distance_threshold=args.distance_threshold,
                    visualize=args.visualize,
                )
                all_reports["concavities"] = concavities_report
                if not concavities_report["is_valid"]:
                    overall_valid = False
                print_report(concavities_report, detailed=False)
                save_individual_report("concavities", concavities_report)
                if not concavities_report["is_valid"]:
                    print(
                        "\n⚠ Error detected in sharp concavities detection. Stopping further checks."
                    )
                else:
                    # 5. Detect internal holes (if CT provided)
                    if args.ct:
                        print("[5/6] Detecting internal holes...")
                        holes_report = detect_internal_holes(
                            args.mask,
                            ct_data=args.ct,
                            output_dir=args.output,
                            file_prefix="",
                            max_hole_area=args.max_area,
                            threshold_air=args.threshold_air,
                            threshold_soft=args.threshold_soft,
                            visualize=args.visualize,
                        )
                        all_reports["holes"] = holes_report
                        if not holes_report["is_valid"]:
                            overall_valid = False
                        print_report(holes_report, detailed=False)
                        save_individual_report("holes", holes_report)
                        if not holes_report["is_valid"]:
                            print(
                                "\n⚠ Error detected in internal holes detection. Stopping further checks."
                            )
                        else:
                            # 6. Detect CT anomalies (if CT provided)
                            if args.ct:
                                print("[6/6] Detecting CT value anomalies...")
                                ct_report = detect_ct_value_anomalies(
                                    args.mask,
                                    ct_data=args.ct,
                                    output_dir=args.output,
                                    file_prefix="",
                                    min_leak_volume=args.min_volume,
                                    z_score_threshold=args.z_score,
                                    visualize=args.visualize,
                                )
                                all_reports["ct_anomalies"] = ct_report
                                if not ct_report["is_valid"]:
                                    overall_valid = False
                                print_report(ct_report, detailed=False)
                                save_individual_report("ct_anomalies", ct_report)
                                if not ct_report["is_valid"]:
                                    print(
                                        "\n⚠ Error detected in CT anomalies detection. Stopping further checks."
                                    )
                            else:
                                print(
                                    "[6/6] Skipping CT anomaly detection (no CT image provided)"
                                )
                    else:
                        print(
                            "[5/6] Skipping internal hole detection (no CT image provided)"
                        )
                        # 6. Detect CT anomalies (if CT provided)
                        if args.ct:
                            print("[6/6] Detecting CT value anomalies...")
                            ct_report = detect_ct_value_anomalies(
                                args.mask,
                                ct_data=args.ct,
                                output_dir=args.output,
                                file_prefix="",
                                min_leak_volume=args.min_volume,
                                z_score_threshold=args.z_score,
                                visualize=args.visualize,
                            )
                            all_reports["ct_anomalies"] = ct_report
                            if not ct_report["is_valid"]:
                                overall_valid = False
                            print_report(ct_report, detailed=False)
                            save_individual_report("ct_anomalies", ct_report)
                            if not ct_report["is_valid"]:
                                print(
                                    "\n⚠ Error detected in CT anomalies detection. Stopping further checks."
                                )
                        else:
                            print(
                                "[6/6] Skipping CT anomaly detection (no CT image provided)"
                            )

    # Overall summary
    print("\n" + "=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    status = "✓ ALL CHECKS PASSED" if overall_valid else "✗ ISSUES DETECTED"
    print(f"Overall Status: {status}")
    print("\nDetailed Reports:")
    for check_name, report in all_reports.items():
        check_status = "✓" if report["is_valid"] else "✗"
        print(f"  {check_status} {check_name}")
    print("=" * 60 + "\n")

    # Save comprehensive report
    if args.output:
        output_file = os.path.join(args.output, "comprehensive_report.json")
        comprehensive_report = {"overall_valid": overall_valid, "checks": all_reports}
        with open(output_file, "w") as f:
            json.dump(comprehensive_report, f, indent=2)
        print(f"Comprehensive report saved to: {output_file}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Medical Image Segmentation Quality Checking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "-v", "--version", action="version", version="mask_check_cli.py 1.0"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Common arguments for commands that need mask
    mask_args = argparse.ArgumentParser(add_help=False)
    mask_args.add_argument("mask", help="Path to mask file (.nii, .nii.gz, .npy, .dcm)")
    mask_args.add_argument(
        "-o", "--output", help="Output directory for results and visualizations"
    )
    mask_args.add_argument("--prefix", default="", help="File prefix for output files")
    mask_args.add_argument(
        "--detailed", action="store_true", help="Print detailed report"
    )
    mask_args.add_argument(
        "--no-viz", dest="visualize", action="store_false", help="Disable visualization"
    )

    # Common arguments for commands that need both mask and CT
    mask_ct_args = argparse.ArgumentParser(add_help=False)
    mask_ct_args.add_argument(
        "mask", help="Path to mask file (.nii, .nii.gz, .npy, .dcm)"
    )
    mask_ct_args.add_argument("ct", help="Path to CT image file")
    mask_ct_args.add_argument(
        "-o", "--output", help="Output directory for results and visualizations"
    )
    mask_ct_args.add_argument(
        "--prefix", default="", help="File prefix for output files"
    )
    mask_ct_args.add_argument(
        "--detailed", action="store_true", help="Print detailed report"
    )
    mask_ct_args.add_argument(
        "--no-viz",
        dest="visualize",
        action="store_true",
        default=True,
        help="Disable visualization",
    )

    # detect-2d-noise command
    parser_2d_noise = subparsers.add_parser(
        "detect-2d-noise",
        parents=[mask_args],
        help="Detect 2D noise in mask slices",
        description="Detect small noise areas in 2D mask slices using connected component analysis",
    )
    parser_2d_noise.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Minimum area threshold for noise detection (default: 10 pixels)",
    )

    # check-connectivity command
    parser_connectivity = subparsers.add_parser(
        "check-connectivity",
        parents=[mask_args],
        help="Check if mask contains multiple 3D ROIs",
        description="Check if mask contains multiple independent 3D ROIs using connected component analysis",
    )
    parser_connectivity.add_argument(
        "--max-components",
        type=int,
        default=1,
        help="Maximum allowed number of 3D components (default: 1)",
    )
    parser_connectivity.add_argument(
        "--connectivity",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Connectivity type: 1=6-conn, 2=18-conn, 3=26-conn (default: 3)",
    )

    # detect-elongated command
    parser_elongated = subparsers.add_parser(
        "detect-elongated",
        parents=[mask_args],
        help="Detect elongated projection leakage",
        description="Detect elongated projections that may indicate segmentation leakage",
    )
    parser_elongated.add_argument(
        "--aspect-ratio",
        type=float,
        default=5.0,
        help="Aspect ratio threshold (default: 5.0)",
    )
    parser_elongated.add_argument(
        "--convexity",
        type=float,
        default=0.85,
        help="Convexity threshold (default: 0.85)",
    )

    # detect-concavities command
    parser_concavities = subparsers.add_parser(
        "detect-concavities",
        parents=[mask_args],
        help="Detect sharp concave/convex defects",
        description="Detect sharp concavities that may indicate segmentation issues",
    )
    parser_concavities.add_argument(
        "--angle-threshold",
        type=float,
        default=30.0,
        help="Sharp angle threshold in degrees (default: 30.0)",
    )
    parser_concavities.add_argument(
        "--distance-threshold",
        type=float,
        default=5.0,
        help="Defect depth threshold in pixels (default: 5.0)",
    )

    # detect-ct-anomalies command
    parser_ct = subparsers.add_parser(
        "detect-ct-anomalies",
        parents=[mask_ct_args],
        help="Detect CT value anomalies",
        description="Detect CT value jumps that may indicate segmentation leakage to other tissues",
    )
    parser_ct.add_argument(
        "--min-volume",
        type=int,
        default=50,
        help="Minimum anomaly region size in voxels (default: 50)",
    )
    parser_ct.add_argument(
        "--z-score",
        type=float,
        default=2.0,
        help="Z-score threshold for anomaly detection (default: 2.0)",
    )

    # detect-holes command
    parser_holes = subparsers.add_parser(
        "detect-holes",
        parents=[mask_ct_args],
        help="Detect internal holes (erased tissue)",
        description="Detect internal holes that are likely accidentally erased tumor tissue",
    )
    parser_holes.add_argument(
        "--max-area",
        type=int,
        default=20,
        help="Maximum hole area (pixels) for noise classification (default: 20)",
    )
    parser_holes.add_argument(
        "--threshold-air",
        type=float,
        default=20,
        help="Percentile threshold for air/normal cavity (default: 20)",
    )
    parser_holes.add_argument(
        "--threshold-soft",
        type=float,
        default=50,
        help="Percentile threshold for soft tissue/erased tissue (default: 50)",
    )

    # check-all command
    parser_all = subparsers.add_parser(
        "check-all",
        help="Run all available quality checks",
        description="Run comprehensive quality checking with all available detection methods",
    )
    parser_all.add_argument(
        "mask", help="Path to mask file (.nii, .nii.gz, .npy, .dcm)"
    )
    parser_all.add_argument(
        "ct",
        nargs="?",
        help="Path to CT image file (optional, required for CT-based checks)",
    )
    parser_all.add_argument(
        "-o", "--output", help="Output directory for results and visualizations"
    )
    parser_all.add_argument(
        "--detailed", action="store_true", help="Print detailed report"
    )
    parser_all.add_argument(
        "--no-viz",
        dest="visualize",
        action="store_false",
        default=True,
        help="Disable visualization",
    )
    # Connectivity options
    parser_all.add_argument(
        "--max-components",
        type=int,
        default=1,
        help="Maximum allowed 3D components (default: 1)",
    )
    parser_all.add_argument(
        "--connectivity",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Connectivity type: 1=6-conn, 2=18-conn, 3=26-conn (default: 3)",
    )
    # 2D noise options
    parser_all.add_argument(
        "--min-area",
        type=int,
        default=10,
        help="Minimum area threshold for 2D noise detection (default: 10)",
    )
    # Elongated projection options
    parser_all.add_argument(
        "--aspect-ratio",
        type=float,
        default=5.0,
        help="Aspect ratio threshold (default: 5.0)",
    )
    parser_all.add_argument(
        "--convexity",
        type=float,
        default=0.85,
        help="Convexity threshold (default: 0.85)",
    )
    # Concavities options
    parser_all.add_argument(
        "--angle-threshold",
        type=float,
        default=30.0,
        help="Sharp angle threshold in degrees (default: 30.0)",
    )
    parser_all.add_argument(
        "--distance-threshold",
        type=float,
        default=5.0,
        help="Defect depth threshold in pixels (default: 5.0)",
    )
    # CT anomalies options
    parser_all.add_argument(
        "--min-volume",
        type=int,
        default=50,
        help="Minimum anomaly region size in voxels (default: 50)",
    )
    parser_all.add_argument(
        "--z-score", type=float, default=2.0, help="Z-score threshold (default: 2.0)"
    )
    # Holes options
    parser_all.add_argument(
        "--max-area",
        type=int,
        default=20,
        help="Maximum hole area for noise classification (default: 20)",
    )
    parser_all.add_argument(
        "--threshold-air",
        type=float,
        default=20,
        help="Percentile threshold for air (default: 20)",
    )
    parser_all.add_argument(
        "--threshold-soft",
        type=float,
        default=50,
        help="Percentile threshold for soft tissue (default: 50)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Route to appropriate command handler
    if args.command == "detect-2d-noise":
        cmd_detect_2d_noise(args)
    elif args.command == "check-connectivity":
        cmd_check_connectivity(args)
    elif args.command == "detect-elongated":
        cmd_detect_elongated(args)
    elif args.command == "detect-concavities":
        cmd_detect_concavities(args)
    elif args.command == "detect-ct-anomalies":
        cmd_detect_ct_anomalies(args)
    elif args.command == "detect-holes":
        cmd_detect_holes(args)
    elif args.command == "check-all":
        cmd_check_all(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
