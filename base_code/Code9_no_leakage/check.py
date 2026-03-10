import os
import re
import random
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any, Optional
from config import DATA_PATHS, VALID_CLASSES, SPLIT_CONFIG

print("hi")

def load_labels(csv_path: str = None) -> pd.DataFrame:
    """Load and preprocess labels CSV"""
    if csv_path is None:
        csv_path = DATA_PATHS["labels_csv"]

    labels = pd.read_csv(csv_path).drop(index=64, errors="ignore").reset_index(drop=True)
    return labels


def get_all_patch_files(patches_dir: str = None) -> List[str]:
    """
    Get list of all patch files.

    CHANGE #1:
    - Return a sorted list so ordering is deterministic.
    """
    if patches_dir is None:
        patches_dir = DATA_PATHS["patches_dir"]

    # Deterministic ordering (important for attention patch_idx -> filename mapping)
    all_files = sorted(os.listdir(patches_dir))
    return all_files


def group_patches_by_slice(all_files: List[str], root_dir: str) -> Dict[Tuple[int, str], List[str]]:
    """
    Group patches by case and slice ID
    Returns: {(case_id, slice_id): [patch_paths]}
    """
    case_slices = defaultdict(list)
    invalid_file_names = []
    flexibility_needed_counter = 0

    for filename in all_files:
        if not filename.endswith(".png"):
            continue

        # Try standard naming convention first
        match = re.match(r"case_(\d+)_([a-z]+_\d+)_", filename)
        if match:
            case_id = int(match.group(1))
            slice_id = match.group(2)
            key = (case_id, slice_id)
            case_slices[key].append(os.path.join(root_dir, filename))
            continue

        # Try without underscore between match/unmatched and number
        match = re.match(r"case_(\d+)_([a-z]+\d+)_", filename)
        if match:
            case_id = int(match.group(1))
            slice_id = match.group(2)
            # Add underscore between letters and numbers
            slice_id = re.sub(r"([A-Za-z])(\d)", r"\1_\2", slice_id)
            key = (case_id, slice_id)
            case_slices[key].append(os.path.join(root_dir, filename))
            flexibility_needed_counter += 1
            continue

        invalid_file_names.append(os.path.join(root_dir, filename))

    # CHANGE #2:
    # Ensure deterministic ordering of patch paths WITHIN each (case_id, slice_id)
    for k in list(case_slices.keys()):
        case_slices[k] = sorted(case_slices[k])

    # Print summary
    if invalid_file_names:
        print(f"Found {len(invalid_file_names)} files not following naming convention:")
        for f in invalid_file_names[:5]:  # Show first 5
            print(f"  {f}")
        if len(invalid_file_names) > 5:
            print(f"  ... and {len(invalid_file_names) - 5} more")
    else:
        print(f"All {flexibility_needed_counter} non-standard file names were handled.")

    return case_slices


def build_slice_to_class_map(patches: Dict, labels: pd.DataFrame) -> Dict[Tuple[int, str], int]:
    """Build mapping from (case_id, slice_id) to class label"""
    slice_to_class = {}

    for (case_id, slice_id), paths in patches.items():
        raw_label = labels.loc[labels["Case"] == case_id, "Class"]
        if not raw_label.empty and raw_label.item() in VALID_CLASSES:
            # Convert to binary: 1.0 -> 0 (benign), 3.0/4.0 -> 1 (high-grade)
            label = 0 if raw_label.item() == 1.0 else 1
            slice_to_class[(case_id, slice_id)] = label

    return slice_to_class


def split_by_case_stratified(slices_by_class: Dict, random_state: int = 42) -> Tuple[List, List, List]:
    """
    Split data by case to prevent leakage, maintaining class balance
    Returns: train_slices, val_slices, test_slices
    """
    # Build case -> label map and validate no mixed-label cases
    case_to_labels = defaultdict(set)
    for label, items in slices_by_class.items():
        for case_id, _ in items:
            case_to_labels[case_id].add(label)

    # Flatten to case list and aligned labels
    case_ids = []
    case_labels = []
    for cid, labs in case_to_labels.items():
        if len(labs) > 1:
            print(f"Warning: Case {cid} has mixed labels: {labs}")
        case_ids.append(cid)
        case_labels.append(next(iter(labs)))  # Take the first (should be only) label

    # Split cases with stratification
    train_ratio = SPLIT_CONFIG["train_ratio"]
    val_ratio = SPLIT_CONFIG["val_ratio"]

    # First split: train vs temp (val + test)
    case_train, case_temp, y_train, y_temp = train_test_split(
        case_ids,
        case_labels,
        test_size=(1 - train_ratio),
        stratify=case_labels,
        random_state=random_state,
    )

    # Second split: val vs test from temp
    val_size = val_ratio / (val_ratio + SPLIT_CONFIG["test_ratio"])
    case_val, case_test, _, _ = train_test_split(
        case_temp,
        y_temp,
        test_size=(1 - val_size),
        stratify=y_temp,
        random_state=random_state,
    )

    case_train = set(case_train)
    case_val = set(case_val)
    case_test = set(case_test)

    # Map case splits back to slice-level lists
    train_slices, val_slices, test_slices = [], [], []
    for label, items in slices_by_class.items():
        for case_id, slice_key in items:
            if case_id in case_train:
                train_slices.append((case_id, slice_key))
            elif case_id in case_val:
                val_slices.append((case_id, slice_key))
            elif case_id in case_test:
                test_slices.append((case_id, slice_key))
            else:
                print(f"Critical error! Case {case_id} not in any split")

    return train_slices, val_slices, test_slices


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--labels_csv", type=str, default=DATA_PATHS["labels_csv"])
parser.add_argument("--patches_dir", type=str, default=DATA_PATHS["patches_dir"])
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# Load labels
labels = load_labels(args.labels_csv)
print(f"Loaded {len(labels)} labels")
    
# Get patch files
all_files = get_all_patch_files(args.patches_dir)
print(f"Found {len(all_files)} patch files")
    
# Group patches by slice
patches = group_patches_by_slice(all_files, args.patches_dir)
print(f"Grouped into {len(patches)} slices")

# Build slice to class mapping
slice_to_class = build_slice_to_class_map(patches, labels)
print(f"Mapped {len(slice_to_class)} slices to classes")
    
# Group slices by class for stratified splitting
slices_by_class = defaultdict(list)
for key, label in slice_to_class.items():
    slices_by_class[label].append(key)

print(f"Class distribution: {dict((k, len(v)) for k, v in slices_by_class.items())}")
    
print("\n" + "-" * 40)
print("SPLITTING DATA")
print("-" * 40)

train_slices, val_slices, test_slices = split_by_case_stratified(slices_by_class, random_state=args.seed)

print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")

tmp_case_dict = defaultdict(lambda: defaultdict(list))
label_map = {}

# IMPORTANT: deterministic ordering of slices
slice_list_sorted = sorted(train_slices, key=lambda x: (int(x[0]), str(x[1])))

print("sorted")
for i in range(10):
    print(slice_list_sorted[i])

#train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)