#!/usr/bin/env python3
"""
Main training script for Hierarchical Attention MIL model
"""
import os
import time
import argparse
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict

EMB_DIR = "/projects/e32998/patches_varsize_pooled4096"  # <-- same as precompute script output

# Import our modules
from config import DATA_PATHS, TRAINING_CONFIG, MODEL_CONFIG
from data_utils import (
    load_labels, get_all_patch_files, group_patches_by_slice,
    build_slice_to_class_map, split_by_case_stratified, build_case_dict,
    report_no_leak, summarize_case_dict
)
from models import create_model
from dataset import StainBagCasePooledFeatureDataset, case_collate_fn
from trainer import MILTrainer, count_patches_by_class
from utils import (
    set_seed, get_device, print_data_summary, create_run_directory,
    save_data_splits, load_data_splits, print_model_summary, check_data_integrity
)
from attention_analysis import analyze_attention_weights


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Hierarchical Attention MIL model')
    
    # Data arguments
    parser.add_argument('--labels_csv', type=str, default=DATA_PATHS['labels_csv'],
                       help='Path to labels CSV file')
    parser.add_argument('--patches_dir', type=str, default=DATA_PATHS['patches_dir'],
                       help='Path to patches directory')
    # checkpoint_dir is now automatically set to {run_dir}/checkpoints
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=TRAINING_CONFIG['epochs'],
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=TRAINING_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=TRAINING_CONFIG['batch_size'],
                       help='Batch size (typically 1 for MIL)')
    parser.add_argument('--num_workers', type=int, default=TRAINING_CONFIG['num_workers'],
                       help='Number of data loader workers')
    
    # Model arguments
    parser.add_argument('--embed_dim', type=int, default=MODEL_CONFIG['embed_dim'],
                       help='Embedding dimension')
    parser.add_argument('--per_slice_cap', type=int, default=MODEL_CONFIG['per_slice_cap'],
                       help='Maximum patches per slice')
    parser.add_argument('--max_slices_per_stain', type=int, default=MODEL_CONFIG['max_slices_per_stain'],
                       help='Maximum slices per stain (None for unlimited)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=TRAINING_CONFIG['random_state'],
                       help='Random seed')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                       help='Only evaluate, do not train')
    parser.add_argument('--analyze_attention', action='store_true',
                       help='Perform attention analysis and visualization')
    parser.add_argument('--attention_top_n', type=int, default=5,
                       help='Number of top/bottom patches to visualize')
    parser.add_argument('--load_splits', type=str, default=None,
                       help='Path to data_splits.npz file to load existing splits')
    
    return parser.parse_args()


def prepare_data(args):
    """Prepare and split the data"""
    print("=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
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
    
    if args.load_splits:
        # Load existing splits
        print(f"Loading existing splits from: {args.load_splits}")
        splits_data = load_data_splits(args.load_splits)
        train_cases_set = set(splits_data['train_cases'])
        val_cases_set = set(splits_data['val_cases'])
        test_cases_set = set(splits_data['test_cases'])
        
        # Map loaded case IDs back to slices
        train_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in train_cases_set]
        val_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in val_cases_set]
        test_slices = [(case_id, slice_id) for (case_id, slice_id) in slice_to_class.keys() if case_id in test_cases_set]
        
        print(f"Loaded splits - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
    else:
        # Split data by case (stratified)
        train_slices, val_slices, test_slices = split_by_case_stratified(
            slices_by_class, random_state=args.seed
        )
        
        print(f"Split sizes - Train: {len(train_slices)}, Val: {len(val_slices)}, Test: {len(test_slices)}")
    
    # Build case dictionaries
    train_case_dict, train_label_map = build_case_dict(train_slices, patches, slice_to_class)
    val_case_dict, val_label_map = build_case_dict(val_slices, patches, slice_to_class)
    test_case_dict, test_label_map = build_case_dict(test_slices, patches, slice_to_class)
    
    # Check for data leakage
    report_no_leak(train_case_dict, val_case_dict, test_case_dict)
    
    # Create summary DataFrames
    train_df = summarize_case_dict(train_case_dict, train_label_map, "train")
    val_df = summarize_case_dict(val_case_dict, val_label_map, "val")
    test_df = summarize_case_dict(test_case_dict, test_label_map, "test")
    
    # Print data summary
    print_data_summary(train_df, val_df, test_df)
    
    # Count patches by class
    count_patches_by_class(train_case_dict, train_label_map, "Train")
    count_patches_by_class(val_case_dict, val_label_map, "Validation")
    count_patches_by_class(test_case_dict, test_label_map, "Test")
    
    # Check data integrity
    check_data_integrity(train_case_dict, train_label_map, "Train")
    check_data_integrity(val_case_dict, val_label_map, "Validation")
    check_data_integrity(test_case_dict, test_label_map, "Test")
    
    return (train_case_dict, train_label_map), (val_case_dict, val_label_map), (test_case_dict, test_label_map)


def create_data_loaders(train_data, val_data, test_data, args):
    """Create data loaders (precomputed pooled features)"""
    print("\n" + "=" * 60)
    print("CREATING DATA LOADERS (POOLED FEATURES)")
    print("=" * 60)

    train_case_dict, train_label_map = train_data
    val_case_dict, val_label_map = val_data
    test_case_dict, test_label_map = test_data

    train_ds = StainBagCasePooledFeatureDataset(
        train_case_dict, train_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=True,
    )

    val_ds = StainBagCasePooledFeatureDataset(
        val_case_dict, val_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=False,
    )

    test_ds = StainBagCasePooledFeatureDataset(
        test_case_dict, test_label_map,
        embeddings_dir=EMB_DIR,
        per_slice_cap=args.per_slice_cap,
        max_slices_per_stain=args.max_slices_per_stain,
        shuffle_patches=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=case_collate_fn,
        persistent_workers=True,
    )

    print(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    args = parse_args()
    start_time = time.time()
    # Set up
    set_seed(args.seed)
    device = get_device()
    
    # Create run directory
    run_dir = create_run_directory()
    
    # Update checkpoint directory to run directory
    args.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 80)
    print("HIERARCHICAL ATTENTION MIL TRAINING")
    print("=" * 80)
    print(f"Arguments: {vars(args)}")
    print(f"Training config (may be overridden by arguments): {TRAINING_CONFIG}")
    
    # Prepare data
    train_data, val_data, test_data = prepare_data(args)
    

if __name__ == "__main__":
    main()