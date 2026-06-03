#!/usr/bin/env python3
"""
Thesis Graph Generator v2 - Accurate Metrics

Key improvements:
- Groups attempts by GT target (retries don't count as separate failures)
- Focuses on UNIQUE GT targets, not raw publish counts
- Shows final outcome per target (success/failure after all retries)
- Accurate timing: time to FIRST success, not per-attempt

Usage:
    python3 plot_evaluation_results_v2.py eval_detailed.csv -o plots/
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import List, Dict, Optional
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class TargetRecord:
    """Single published target record"""
    target_num: int
    publish_time_s: float
    world_x: float
    world_y: float
    world_z: float
    robot_x: float
    robot_y: float
    robot_z: float
    matched_gt_name: Optional[str]
    gt_x: Optional[float]
    gt_y: Optional[float]
    gt_z: Optional[float]
    swincar_gt_y: Optional[float]
    swincar_gt_z: Optional[float]
    position_error_mm: Optional[float]
    is_true_positive: bool
    beam_status: str
    pointing_duration_s: Optional[float]


@dataclass
class GTTargetResult:
    """Aggregated result for a single GT target"""
    gt_name: str
    gt_pos: tuple  # (x, y, z)
    swincar_gt_y: Optional[float]
    swincar_gt_z: Optional[float]
    total_attempts: int
    successful: bool  # Did it eventually succeed?
    first_success_time: Optional[float]  # Time when first succeeded
    first_attempt_time: float  # When first attempted
    total_time_spent: float  # Sum of all pointing durations
    time_to_success: Optional[float]  # Time from first attempt to success
    best_error_mm: float  # Best (lowest) position error across attempts
    final_error_mm: float  # Error on successful attempt (or best if failed)
    attempts: List[TargetRecord] = field(default_factory=list)
    retries_before_success: int = 0


@dataclass 
class GTData:
    """Ground truth from status CSV"""
    name: str
    x: float
    y: float
    z: float
    detected: bool
    detection_time_s: Optional[float]


def load_detailed_csv(filepath: str) -> List[TargetRecord]:
    """Load detailed CSV data"""
    records = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # Compute position error from components (dx, dy, dz)
                error_dx = float(row['error_dx_mm']) if row['error_dx_mm'] else 0
                error_dy = float(row['error_dy_mm']) if row['error_dy_mm'] else 0
                error_dz = float(row['error_dz_mm']) if row['error_dz_mm'] else 0
                position_error = math.sqrt(error_dx**2 + error_dy**2 + error_dz**2)
                
                records.append(TargetRecord(
                    target_num=int(row['target_num']),
                    publish_time_s=float(row['publish_time_s']),
                    world_x=float(row['world_x']),
                    world_y=float(row['world_y']),
                    world_z=float(row['world_z']),
                    robot_x=float(row['robot_x']),
                    robot_y=float(row['robot_y']),
                    robot_z=float(row['robot_z']),
                    matched_gt_name=row['matched_gt_name'] if row['matched_gt_name'] else None,
                    gt_x=float(row['gt_x']) if row['gt_x'] else None,
                    gt_y=float(row['gt_y']) if row['gt_y'] else None,
                    gt_z=float(row['gt_z']) if row['gt_z'] else None,
                    swincar_gt_y=float(row['swincar_gt_y']) if row.get('swincar_gt_y') else None,
                    swincar_gt_z=float(row['swincar_gt_z']) if row.get('swincar_gt_z') else None,
                    position_error_mm=position_error if position_error > 0 else None,
                    is_true_positive=row['is_true_positive'] == '1',
                    beam_status=row['beam_status'],
                    pointing_duration_s=float(row['pointing_duration_s']) if row['pointing_duration_s'] else None
                ))
            except Exception as e:
                print(f"Warning: Could not parse row: {e}")
    return records


def load_gt_status_csv(filepath: str) -> List[GTData]:
    """Load GT status CSV"""
    gt_data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both 'gt_name' and 'name' column names
                gt_name = row.get('gt_name') or row.get('name')
                gt_data.append(GTData(
                    name=gt_name,
                    x=float(row['x']),
                    y=float(row['y']),
                    z=float(row['z']),
                    detected=row.get('detected', '1') == '1',  # Default to True if missing
                    detection_time_s=float(row['detection_time_s']) if row.get('detection_time_s') else None
                ))
    except FileNotFoundError:
        print(f"Warning: GT status file not found: {filepath}")
    except KeyError as e:
        print(f"Warning: GT status CSV missing expected column: {e}")
    return gt_data


def aggregate_by_gt_target(records: List[TargetRecord]) -> Dict[str, GTTargetResult]:
    """
    Group all attempts by GT target and compute final outcome.
    This is the KEY function - it determines if a target was ultimately successful.
    """
    # Group by GT name
    by_gt: Dict[str, List[TargetRecord]] = defaultdict(list)
    fp_records: List[TargetRecord] = []
    
    for r in records:
        if r.matched_gt_name:
            by_gt[r.matched_gt_name].append(r)
        else:
            fp_records.append(r)
    
    results = {}
    
    for gt_name, attempts in by_gt.items():
        # Sort by time
        attempts.sort(key=lambda x: x.publish_time_s)
        
        first_attempt = attempts[0]
        gt_pos = (first_attempt.gt_x, first_attempt.gt_y, first_attempt.gt_z)
        
        # Check if any attempt succeeded
        success_attempts = [a for a in attempts if a.beam_status == 'success']
        successful = len(success_attempts) > 0
        
        # Find first success
        first_success_time = None
        time_to_success = None
        retries_before_success = 0
        
        if successful:
            first_success = success_attempts[0]
            first_success_time = first_success.publish_time_s
            time_to_success = first_success_time - first_attempt.publish_time_s
            # Count failures before first success
            for a in attempts:
                if a.beam_status == 'success':
                    break
                retries_before_success += 1
        
        # Best error (minimum across all attempts)
        errors = [a.position_error_mm for a in attempts if a.position_error_mm is not None]
        best_error = min(errors) if errors else 0.0
        
        # Final error (from successful attempt, or best if failed)
        if successful:
            final_error = success_attempts[0].position_error_mm or best_error
        else:
            final_error = best_error
        
        # Total time spent
        total_time = sum(a.pointing_duration_s or 0 for a in attempts)
        
        results[gt_name] = GTTargetResult(
            gt_name=gt_name,
            gt_pos=gt_pos,
            swincar_gt_y=first_attempt.swincar_gt_y,
            swincar_gt_z=first_attempt.swincar_gt_z,
            total_attempts=len(attempts),
            successful=successful,
            first_success_time=first_success_time,
            first_attempt_time=first_attempt.publish_time_s,
            total_time_spent=total_time,
            time_to_success=time_to_success,
            best_error_mm=best_error,
            final_error_mm=final_error,
            attempts=attempts,
            retries_before_success=retries_before_success
        )
    
    return results


def plot_position_error_histogram(gt_results: Dict[str, GTTargetResult], output_dir: str):
    """
    Histogram of position errors - ONE value per GT target (final/best error).
    Excludes outliers: position errors > 5mm
    """
    # Only successful targets for "achieved accuracy"
    successful = [r for r in gt_results.values() if r.successful]
    # Filter out outliers: position errors > 5mm
    errors = [r.final_error_mm for r in successful if r.final_error_mm <= 5.0]
    outliers_count = len(successful) - len(errors)
    
    if not errors:
        print("No successful targets for error histogram (after outlier removal)")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    n, bins, patches = ax.hist(errors, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
    
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    median_err = np.median(errors)
    
    ax.axvline(mean_err, color='red', linestyle='--', linewidth=2)
    ax.axvline(median_err, color='green', linestyle=':', linewidth=2)
    
    ax.set_xlabel('Position Error (mm)', fontsize=12)
    ax.set_ylabel('Number of GT Targets', fontsize=12)
    ax.set_title('Detection Accuracy: Position Error per GT Target (Successful Only)', fontsize=14)
    # Do not show legend here; stats box includes mean/median values
    ax.grid(True, alpha=0.3)
    
    stats_text = (f'N = {len(errors)} targets (excl. {outliers_count} outliers >5mm)\n'
                  f'Mean (--): {mean_err:.1f} mm\n'
                  f'Median (:): {median_err:.1f} mm\n'
                  f'Std = {std_err:.1f} mm\n'
                  f'Max = {max(errors):.1f} mm\n'
                  f'Min = {min(errors):.1f} mm')
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_histogram.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'error_histogram.pdf'))
    plt.close()
    print("Saved: error_histogram.png/pdf")


def plot_cumulative_detection(gt_results: Dict[str, GTTargetResult], gt_data: List[GTData], output_dir: str):
    """
    Cumulative UNIQUE GT targets detected over time.
    Counts both successful and failed detections (attempted targets).
    Each GT target counted only once (on first attempt).
    """
    # Get all attempted targets (successful or failed) with their first attempt time
    attempted = [(r.gt_name, r.first_attempt_time) for r in gt_results.values()]
    attempted.sort(key=lambda x: x[1])
    
    if not attempted:
        print("No attempted detections for cumulative plot")
        return
    
    times = [t for _, t in attempted]
    cumsum = list(range(1, len(times) + 1))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Step plot for unique detections (all attempts)
    ax.step(times, cumsum, where='post', label='Unique GT Targets Detected (all attempts)', 
            color='blue', linewidth=2.5, linestyle='-')
    
    # Overlay: successful detections
    successful = [(r.gt_name, r.first_success_time) for r in gt_results.values() 
                  if r.successful and r.first_success_time is not None]
    successful.sort(key=lambda x: x[1])
    
    if successful:
        s_times = [t for _, t in successful]
        s_cumsum = list(range(1, len(successful) + 1))
        ax.step(s_times, s_cumsum, where='post', label='Successfully Pointed At', 
                color='green', linewidth=2.5, linestyle='--')
    
    # GT total line
    gt_total = len(gt_data) if gt_data else len(gt_results)
    ax.axhline(gt_total, color='red', linestyle='--', linewidth=2, 
               label=f'Total GT Targets ({gt_total})')
    
    # Final detection rate
    final_attempted = len(attempted)
    final_successful = len(successful)
    attempt_rate = final_attempted / gt_total * 100 if gt_total > 0 else 0
    success_rate = final_successful / gt_total * 100 if gt_total > 0 else 0
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Cumulative Unique GT Targets', fontsize=12)
    ax.set_title(f'Detection Progress Over Time\n(Attempted: {final_attempted}/{gt_total} = {attempt_rate:.1f}% | Success: {final_successful}/{gt_total} = {success_rate:.1f}%)', 
                 fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add annotations
    if times:
        ax.annotate(f'{final_attempted} detected', 
                    xy=(times[-1], final_attempted),
                    xytext=(times[-1] - 50, final_attempted - 5),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='blue'))
    
    if s_times:
        ax.annotate(f'{final_successful} successful', 
                    xy=(s_times[-1], final_successful),
                    xytext=(s_times[-1] - 50, final_successful + 2),
                    fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_detections.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'cumulative_detections.pdf'))
    plt.close()
    print("Saved: cumulative_detections.png/pdf")


def plot_summary_bars(gt_results: Dict[str, GTTargetResult], gt_data: List[GTData], 
                      fp_count: int, output_dir: str):
    """
    Summary bar charts with CORRECT metrics:
    - TP = unique GT targets successfully detected
    - FN = GT targets never detected (not attempted OR attempted but failed)
    - FP = detections that didn't match any GT
    """
    # Unique GT targets
    successful_gt = sum(1 for r in gt_results.values() if r.successful)
    failed_gt = sum(1 for r in gt_results.values() if not r.successful)
    
    # GT targets never even attempted
    gt_total = len(gt_data) if gt_data else 0
    attempted_gt_names = set(gt_results.keys())
    if gt_data:
        never_attempted = sum(1 for g in gt_data if g.name not in attempted_gt_names)
    else:
        never_attempted = 0
    
    # True metrics
    tp = successful_gt
    fn = failed_gt + never_attempted  # Failed after trying + never tried
    fp = fp_count
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Detection Outcomes (per unique GT target)
    ax1 = axes[0]
    categories = ['Successful', 'Failed\n(after retries)', 'Never\nAttempted']
    values = [successful_gt, failed_gt, never_attempted]
    colors = ['#2ecc71', '#e74c3c', '#95a5a6']
    
    bars1 = ax1.bar(categories, values, color=colors, edgecolor='black', alpha=0.8)
    ax1.bar_label(bars1, padding=3, fontsize=11)
    ax1.set_ylabel('Number of GT Targets', fontsize=12)
    ax1.set_title(f'GT Target Outcomes\n(Total: {gt_total} targets)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Detection Performance (TP/FP/FN)
    ax2 = axes[1]
    categories2 = ['True\nPositives', 'False\nPositives', 'False\nNegatives']
    values2 = [tp, fp, fn]
    colors2 = ['#2ecc71', '#f39c12', '#e74c3c']
    
    bars2 = ax2.bar(categories2, values2, color=colors2, edgecolor='black', alpha=0.8)
    ax2.bar_label(bars2, padding=3, fontsize=11)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Detection Classification', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Key Metrics (Precision, Recall, Success Rate)
    ax3 = axes[2]
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    success_rate = successful_gt / len(gt_results) * 100 if gt_results else 0
    
    metrics = ['Precision', 'Recall', 'F1 Score', 'Success\nRate']
    metric_values = [precision, recall, f1, success_rate]
    
    bars3 = ax3.bar(metrics, metric_values, color='steelblue', edgecolor='black', alpha=0.8)
    ax3.bar_label(bars3, fmt='%.1f%%', padding=3, fontsize=11)
    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_ylim(0, 110)
    ax3.set_title('Performance Metrics', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_bars.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'summary_bars.pdf'))
    plt.close()
    print("Saved: summary_bars.png/pdf")


def plot_pointing_time_distribution(gt_results: Dict[str, GTTargetResult], output_dir: str):
    """
    Pointing time distribution - total time per GT target (including retries).
    Also shows time-to-success for successful targets.
    Excludes outliers: pointing times > 2s
    """
    successful = [r for r in gt_results.values() if r.successful]
    
    if not successful:
        print("No successful targets for timing plot")
        return
    
    # Filter out outliers: pointing times > 10s
    successful_filtered = [r for r in successful if r.total_time_spent <= 10.0]
    outliers_count = len(successful) - len(successful_filtered)
    
    if not successful_filtered:
        print(f"No targets after outlier removal for timing plot (excluded {outliers_count} outliers >10s)")
        return
    
    # Time to first success (more meaningful than total time)
    times_to_success = [r.time_to_success + (r.attempts[0].pointing_duration_s or 0) 
                        for r in successful_filtered if r.time_to_success is not None]
    
    # Total pointing time per target
    total_times = [r.total_time_spent for r in successful_filtered]
    
    # Single attempt times (for targets that succeeded first try)
    single_attempt = [r for r in successful_filtered if r.total_attempts == 1]
    single_times = [r.total_time_spent for r in single_attempt]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Histogram of pointing times
    ax1 = axes[0]
    ax1.hist(total_times, bins=20, edgecolor='black', alpha=0.7, color='forestgreen', 
             label='All successful')
    if single_times:
        ax1.hist(single_times, bins=15, edgecolor='black', alpha=0.5, color='lightgreen',
                 label='First-try success')
    
    mean_t = np.mean(total_times)
    ax1.axvline(mean_t, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_t:.2f}s')
    
    ax1.set_xlabel('Total Pointing Time (s)', fontsize=12)
    ax1.set_ylabel('Number of GT Targets', fontsize=12)
    ax1.set_title(f'Pointing Time Distribution\n(Excluded {outliers_count} outliers >10s)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Time vs number of attempts (NO threshold - show all)
    ax2 = axes[1]
    attempts = [r.total_attempts for r in successful]
    times = [r.total_time_spent for r in successful]
    
    # Color by success type
    colors = ['green' if a == 1 else 'orange' for a in attempts]
    ax2.scatter(attempts, times, c=colors, alpha=0.7, s=60, edgecolor='black')
    
    ax2.set_xlabel('Number of Attempts', fontsize=12)
    ax2.set_ylabel('Total Pointing Time (s)', fontsize=12)
    ax2.set_title('Pointing Time vs Retry Count (all data)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Legend
    green_patch = mpatches.Patch(color='green', label='First-try success')
    orange_patch = mpatches.Patch(color='orange', label='Required retries')
    ax2.legend(handles=[green_patch, orange_patch])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pointing_times.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'pointing_times.pdf'))
    plt.close()
    print("Saved: pointing_times.png/pdf")


def plot_error_vs_position(gt_results: Dict[str, GTTargetResult], output_dir: str):
    """
    Position error vs target location (X coordinate = along track).
    Shows if error increases with distance/position.
    Excludes outliers: position errors > 5mm
    """
    successful = [r for r in gt_results.values() if r.successful]
    
    if not successful:
        return
    
    # Filter out outliers: position errors > 5mm
    successful_filtered = [r for r in successful if r.final_error_mm <= 5.0]
    outliers_count = len(successful) - len(successful_filtered)
    
    if not successful_filtered:
        print(f"No targets after outlier removal for error_vs_position plot (excluded {outliers_count} outliers)")
        return
    
    x_coords = [r.gt_pos[0] for r in successful_filtered]
    errors = [r.final_error_mm for r in successful_filtered]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.scatter(x_coords, errors, alpha=0.7, c='steelblue', s=50, edgecolor='black')
    
    # Trend line
    if len(x_coords) > 5:
        z = np.polyfit(x_coords, errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_coords), max(x_coords), 100)
        ax.plot(x_line, p(x_line), 'r--', linewidth=2, 
                label=f'Trend (slope={z[0]:.3f} mm/m)')
        ax.legend()
    
    ax.set_xlabel('Target X Position (m)', fontsize=12)
    ax.set_ylabel('Position Error (mm)', fontsize=12)
    ax.set_title(f'Detection Error vs Target Position Along Track\n(Excluded {outliers_count} outliers >5mm)', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_vs_position.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'error_vs_position.pdf'))
    plt.close()
    print("Saved: error_vs_position.png/pdf")


def plot_gt_elevation_and_lateral_profiles(gt_results: Dict[str, GTTargetResult], output_dir: str):
    """
    Plot GT elevation vs target index and GT/robot lateral position vs target index.
    """
    def target_index(gt_name: str) -> int:
        try:
            return int(gt_name.split('_')[-1])
        except ValueError:
            return 0

    ordered = sorted(gt_results.values(), key=lambda r: target_index(r.gt_name))
    if not ordered:
        print("No GT results for elevation/lateral profile plot")
        return

    indices = list(range(1, len(ordered) + 1))
    gt_z = [((r.swincar_gt_z if r.swincar_gt_z is not None else r.gt_pos[2]) + 0.4)*1.4 for r in ordered]
    gt_y = [r.swincar_gt_y if r.swincar_gt_y is not None else r.gt_pos[1] for r in ordered]
    robot_y = [r.attempts[0].robot_y if r.attempts else 0.0 for r in ordered]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)

    axes[0].plot(indices, gt_z, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=4)
    axes[0].set_title('GT Elevation Profile Across Targets', fontsize=14)
    axes[0].set_xlabel('Target Index', fontsize=12)
    axes[0].set_ylabel('GT Z (m)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(indices, gt_y, marker='o', linestyle='-', color='tab:green', linewidth=2, markersize=4, label='GT Y')
    axes[1].plot(indices, robot_y, marker='x', linestyle='--', color='tab:orange', linewidth=1.5, markersize=4, label='Robot Y (first attempt)')
    axes[1].set_title('GT and Robot Lateral Position Across Targets', fontsize=14)
    axes[1].set_xlabel('Target Index', fontsize=12)
    axes[1].set_ylabel('Y Position (m)', fontsize=12)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gt_elevation_and_lateral_profile.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'gt_elevation_and_lateral_profile.pdf'))
    plt.close()
    print("Saved: gt_elevation_and_lateral_profile.png/pdf")


def plot_retry_analysis(gt_results: Dict[str, GTTargetResult], output_dir: str):
    """
    Analysis of retry behavior - how many targets needed retries?
    """
    successful = [r for r in gt_results.values() if r.successful]
    failed = [r for r in gt_results.values() if not r.successful]
    
    # Group by attempt count
    attempt_counts = defaultdict(int)
    for r in successful:
        attempt_counts[r.total_attempts] += 1
    
    failed_attempt_counts = defaultdict(int)
    for r in failed:
        failed_attempt_counts[r.total_attempts] += 1
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Successful targets by attempt count
    ax1 = axes[0]
    attempts = sorted(attempt_counts.keys())
    counts = [attempt_counts[a] for a in attempts]
    
    colors = ['#2ecc71' if a == 1 else '#f39c12' if a <= 3 else '#e74c3c' for a in attempts]
    bars1 = ax1.bar([str(a) for a in attempts], counts, color=colors, edgecolor='black', alpha=0.8)
    ax1.bar_label(bars1, padding=3)
    
    ax1.set_xlabel('Number of Attempts', fontsize=12)
    ax1.set_ylabel('Number of GT Targets', fontsize=12)
    ax1.set_title(f'Successful Targets: Attempts Required\n(Total: {len(successful)})', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Stats annotation
    first_try = attempt_counts.get(1, 0)
    needed_retry = len(successful) - first_try
    ax1.text(0.97, 0.97, 
             f'First-try: {first_try} ({first_try/len(successful)*100:.1f}%)\n'
             f'Needed retry: {needed_retry} ({needed_retry/len(successful)*100:.1f}%)',
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. Failed targets by attempt count
    ax2 = axes[1]
    if failed_attempt_counts:
        f_attempts = sorted(failed_attempt_counts.keys())
        f_counts = [failed_attempt_counts[a] for a in f_attempts]
        
        bars2 = ax2.bar([str(a) for a in f_attempts], f_counts, color='#e74c3c', 
                        edgecolor='black', alpha=0.8)
        ax2.bar_label(bars2, padding=3)
    
    ax2.set_xlabel('Number of Attempts', fontsize=12)
    ax2.set_ylabel('Number of GT Targets', fontsize=12)
    ax2.set_title(f'Failed Targets: Attempts Made\n(Total: {len(failed)})', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'retry_analysis.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'retry_analysis.pdf'))
    plt.close()
    print("Saved: retry_analysis.png/pdf")


def plot_detection_map(gt_results: Dict[str, GTTargetResult], gt_data: List[GTData], output_dir: str):
    """
    Bird's eye view map showing successful vs failed GT targets.
    """
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # All GT targets from status file (if available)
    if gt_data:
        # Never attempted
        attempted_names = set(gt_results.keys())
        never_attempted = [g for g in gt_data if g.name not in attempted_names]
        
        if never_attempted:
            ax.scatter([g.x for g in never_attempted], [g.y for g in never_attempted],
                      marker='o', s=50, c='gray', alpha=0.5, 
                      label=f'Never Attempted ({len(never_attempted)})')
    
    # Failed targets (attempted but never succeeded)
    failed = [r for r in gt_results.values() if not r.successful]
    if failed:
        ax.scatter([r.gt_pos[0] for r in failed], [r.gt_pos[1] for r in failed],
                  marker='x', s=80, c='red', linewidth=2,
                  label=f'Failed ({len(failed)})')
    
    # Successful targets
    successful = [r for r in gt_results.values() if r.successful]
    
    # Split by first-try vs retry
    first_try_success = [r for r in successful if r.total_attempts == 1]
    retry_success = [r for r in successful if r.total_attempts > 1]
    
    if first_try_success:
        ax.scatter([r.gt_pos[0] for r in first_try_success], 
                  [r.gt_pos[1] for r in first_try_success],
                  marker='o', s=60, c='green', alpha=0.7,
                  label=f'Success (1st try): {len(first_try_success)}')
    
    if retry_success:
        ax.scatter([r.gt_pos[0] for r in retry_success], 
                  [r.gt_pos[1] for r in retry_success],
                  marker='s', s=60, c='orange', alpha=0.7,
                  label=f'Success (after retry): {len(retry_success)}')
    
    ax.set_xlabel('World X (m)', fontsize=12)
    ax.set_ylabel('World Y (m)', fontsize=12)
    ax.set_title("Detection Map: GT Target Outcomes", fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detection_map.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'detection_map.pdf'))
    plt.close()
    print("Saved: detection_map.png/pdf")


def plot_throughput(gt_results: Dict[str, GTTargetResult], output_dir: str):
    """
    Throughput analysis - targets detected per unit time.
    """
    successful = [r for r in gt_results.values() if r.successful]
    if not successful:
        return
    
    # Sort by success time
    successful.sort(key=lambda r: r.first_success_time)
    
    times = [r.first_success_time for r in successful]
    
    # Compute instantaneous rate (targets per minute, rolling window)
    window_size = 60  # seconds
    rates = []
    rate_times = []
    
    for i, t in enumerate(times):
        # Count targets in window ending at t
        count = sum(1 for tt in times[:i+1] if t - tt <= window_size)
        rate = count / (window_size / 60)  # per minute
        rates.append(rate)
        rate_times.append(t)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Cumulative with rate overlay
    ax1 = axes[0]
    cumsum = list(range(1, len(times) + 1))
    ax1.plot(times, cumsum, 'g-', linewidth=2, label='Cumulative Detections')
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Cumulative GT Targets', fontsize=12, color='green')
    ax1.tick_params(axis='y', labelcolor='green')
    
    ax1_twin = ax1.twinx()
    ax1_twin.plot(rate_times, rates, 'b--', linewidth=1.5, alpha=0.7, label='Rate')
    ax1_twin.set_ylabel('Rate (targets/min)', fontsize=12, color='blue')
    ax1_twin.tick_params(axis='y', labelcolor='blue')
    
    ax1.set_title('Detection Progress and Rate', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 2. Overall statistics
    ax2 = axes[1]
    total_time = max(times) - min(times) if len(times) > 1 else times[0]
    total_targets = len(successful)
    overall_rate = total_targets / (total_time / 60) if total_time > 0 else 0
    avg_time_per_target = total_time / total_targets if total_targets > 0 else 0
    
    stats = ['Total\nTargets', 'Total\nTime (min)', 'Rate\n(tgt/min)', 'Avg Time\nper Target (s)']
    values = [total_targets, total_time/60, overall_rate, avg_time_per_target]
    
    bars = ax2.bar(stats, values, color=['steelblue', 'steelblue', 'green', 'orange'],
                   edgecolor='black', alpha=0.8)
    
    # Custom labels
    labels = [f'{total_targets}', f'{total_time/60:.1f}', f'{overall_rate:.2f}', f'{avg_time_per_target:.1f}']
    for bar, label in zip(bars, labels):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Throughput Statistics', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throughput.png'), dpi=150)
    plt.savefig(os.path.join(output_dir, 'throughput.pdf'))
    plt.close()
    print("Saved: throughput.png/pdf")


def print_summary(gt_results: Dict[str, GTTargetResult], gt_data: List[GTData], fp_count: int):
    """Print text summary of results"""
    successful = [r for r in gt_results.values() if r.successful]
    failed = [r for r in gt_results.values() if not r.successful]
    
    gt_total = len(gt_data) if gt_data else len(gt_results)
    attempted = len(gt_results)
    
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY (Corrected Metrics)")
    print("=" * 60)
    
    print(f"\nGround Truth Targets: {gt_total}")
    print(f"  - Attempted: {attempted}")
    print(f"  - Never attempted: {gt_total - attempted}")
    
    print(f"\nOutcomes:")
    print(f"  - Successful: {len(successful)} ({len(successful)/gt_total*100:.1f}%)")
    print(f"  - Failed (after retries): {len(failed)}")
    print(f"  - False Positives: {fp_count}")
    
    first_try = sum(1 for r in successful if r.total_attempts == 1)
    print(f"\nRetry Analysis:")
    if len(successful) > 0:
        print(f"  - First-try success: {first_try} ({first_try/len(successful)*100:.1f}% of successful)")
        print(f"  - Needed retries: {len(successful) - first_try}")
    else:
        print(f"  - No successful targets to analyze retries")
    
    if successful:
        errors = [r.final_error_mm for r in successful]
        print(f"\nPosition Accuracy (successful targets):")
        print(f"  - Mean error: {np.mean(errors):.1f} mm")
        print(f"  - Median error: {np.median(errors):.1f} mm")
        print(f"  - Max error: {max(errors):.1f} mm")
        print(f"  - Min error: {min(errors):.1f} mm")
        
        times = [r.total_time_spent for r in successful]
        print(f"\nTiming:")
        print(f"  - Mean pointing time: {np.mean(times):.2f} s")
        print(f"  - Total time for all: {sum(times):.1f} s")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate thesis graphs (v2 - accurate metrics)')
    parser.add_argument('detailed_csv', help='Path to eval_detailed.csv')
    parser.add_argument('--output-dir', '-o', default='plots', help='Output directory')
    parser.add_argument('--gt-status-csv', '-g', help='Path to GT status CSV')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading: {args.detailed_csv}")
    records = load_detailed_csv(args.detailed_csv)
    print(f"  Loaded {len(records)} raw records")
    
    # Load GT status
    gt_status_path = args.gt_status_csv or args.detailed_csv.replace('.csv', '_gt_status.csv')
    gt_data = load_gt_status_csv(gt_status_path)
    if gt_data:
        print(f"  Loaded {len(gt_data)} GT targets from status file")
    
    # Aggregate by GT target
    gt_results = aggregate_by_gt_target(records)
    print(f"  Aggregated to {len(gt_results)} unique GT targets")
    
    # Count FPs
    fp_count = sum(1 for r in records if not r.is_true_positive)
    
    # Print summary
    print_summary(gt_results, gt_data, fp_count)
    
    # Generate plots
    print(f"\nGenerating plots in: {args.output_dir}/")
    print("-" * 40)
    
    plot_position_error_histogram(gt_results, args.output_dir)
    plot_cumulative_detection(gt_results, gt_data, args.output_dir)
    plot_summary_bars(gt_results, gt_data, fp_count, args.output_dir)
    plot_pointing_time_distribution(gt_results, args.output_dir)
    plot_error_vs_position(gt_results, args.output_dir)
    plot_gt_elevation_and_lateral_profiles(gt_results, args.output_dir)
    plot_retry_analysis(gt_results, args.output_dir)
    plot_detection_map(gt_results, gt_data, args.output_dir)
    plot_throughput(gt_results, args.output_dir)
    
    print("-" * 40)
    print(f"Done! All plots saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()