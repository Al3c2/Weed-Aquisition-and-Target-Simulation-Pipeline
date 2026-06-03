#!/usr/bin/env python3
"""
Filter SDF file to keep only models that exist in ground truth CSV.

Usage:
    python3 filter_sdf_by_gt.py gt_world.csv input.sdf output.sdf
    
    # Or to just list what would be removed:
    python3 filter_sdf_by_gt.py gt_world.csv input.sdf --dry-run
"""

import argparse
import csv
import re
import sys
from typing import Set


def load_gt_names(csv_path: str) -> Set[str]:
    """Load target names from ground truth CSV"""
    names = set()
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            names.add(row['name'])
    return names


def filter_sdf(sdf_content: str, gt_names: Set[str], dry_run: bool = False) -> str:
    """
    Remove <model> blocks whose name is NOT in gt_names.
    Only affects models matching pattern 'row*_target_*'
    """
    
    # Pattern to match complete model blocks
    # This handles multi-line model definitions
    model_pattern = re.compile(
        r"<model\s+name=['\"]([^'\"]+)['\"]>.*?</model>",
        re.DOTALL
    )
    
    removed = []
    kept = []
    
    def replace_func(match):
        model_name = match.group(1)
        
        # Only filter target models (row*_target_*)
        if re.match(r'row\d+_target_\d+', model_name):
            if model_name in gt_names:
                kept.append(model_name)
                return match.group(0)  # Keep it
            else:
                removed.append(model_name)
                return ""  # Remove it
        else:
            # Not a target model, keep it
            return match.group(0)
    
    if dry_run:
        # Just analyze, don't modify
        for match in model_pattern.finditer(sdf_content):
            model_name = match.group(1)
            if re.match(r'row\d+_target_\d+', model_name):
                if model_name in gt_names:
                    kept.append(model_name)
                else:
                    removed.append(model_name)
        
        return None, kept, removed
    else:
        filtered = model_pattern.sub(replace_func, sdf_content)
        
        # Clean up multiple blank lines left by removal
        filtered = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered)
        
        return filtered, kept, removed


def main():
    parser = argparse.ArgumentParser(description='Filter SDF to keep only GT models')
    parser.add_argument('gt_csv', help='Ground truth CSV file')
    parser.add_argument('input_sdf', help='Input SDF file')
    parser.add_argument('output_sdf', nargs='?', help='Output SDF file (optional with --dry-run)')
    parser.add_argument('--dry-run', '-n', action='store_true', 
                        help='Just show what would be removed, do not modify')
    
    args = parser.parse_args()
    
    if not args.dry_run and not args.output_sdf:
        print("Error: output_sdf required unless using --dry-run")
        sys.exit(1)
    
    # Load ground truth names
    print(f"Loading ground truth from: {args.gt_csv}")
    gt_names = load_gt_names(args.gt_csv)
    print(f"  Found {len(gt_names)} targets in ground truth")
    
    # Load SDF
    print(f"Loading SDF from: {args.input_sdf}")
    with open(args.input_sdf, 'r') as f:
        sdf_content = f.read()
    
    # Filter
    filtered, kept, removed = filter_sdf(sdf_content, gt_names, dry_run=args.dry_run)
    
    # Report
    print(f"\n{'=' * 60}")
    print(f"RESULTS {'(DRY RUN)' if args.dry_run else ''}")
    print(f"{'=' * 60}")
    print(f"Target models in SDF: {len(kept) + len(removed)}")
    print(f"  - Keeping (in GT): {len(kept)}")
    print(f"  - Removing (not in GT): {len(removed)}")
    
    if removed:
        print(f"\nModels to be REMOVED ({len(removed)}):")
        # Sort by target number for readability
        removed_sorted = sorted(removed, key=lambda x: (
            x.split('_')[0],  # row part
            int(re.search(r'target_(\d+)', x).group(1)) if re.search(r'target_(\d+)', x) else 0
        ))
        for name in removed_sorted:
            print(f"  - {name}")
    
    # Check for GT targets not found in SDF
    kept_set = set(kept)
    missing_from_sdf = gt_names - kept_set
    if missing_from_sdf:
        print(f"\nWARNING: {len(missing_from_sdf)} GT targets NOT FOUND in SDF:")
        for name in sorted(missing_from_sdf):
            print(f"  - {name}")
    
    # Write output
    if not args.dry_run:
        print(f"\nWriting filtered SDF to: {args.output_sdf}")
        with open(args.output_sdf, 'w') as f:
            f.write(filtered)
        print("Done!")
    else:
        print(f"\nDry run complete. Use without --dry-run to actually filter.")


if __name__ == '__main__':
    main()