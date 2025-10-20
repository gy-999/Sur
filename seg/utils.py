# utils.py
import math
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import DataLoader
from seg.dataset import *
from seg.config import *
from torchvision import transforms
from seg.preprocess import *
import argparse
import os
import pandas as pd
import json
from datetime import datetime
from seg.config import *
import torch


# Get the specified model
def get_model(model_name, in_channels=4, out_channels=4):
    if model_name not in MODEL_MAP:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_MAP.keys())}")
    return MODEL_MAP[model_name](in_channels, out_channels)


def load_model_weights(model, checkpoint_path):
    """Safely load model weights and handle mismatched keys"""
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Filter out unnecessary keys
    filtered_state_dict = {}
    for key, value in state_dict.items():
        if 'total_ops' not in key and 'total_params' not in key:
            filtered_state_dict[key] = value

    # Get current model state dict
    model_state_dict = model.state_dict()

    # Check key matching
    missing_keys = []
    unexpected_keys = []

    for key in filtered_state_dict:
        if key not in model_state_dict:
            unexpected_keys.append(key)

    for key in model_state_dict:
        if key not in filtered_state_dict:
            missing_keys.append(key)

    if missing_keys:
        print(f"Warning: Missing the following keys: {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys: {unexpected_keys}")

    # Load weights (non-strict to allow partial loading)
    model.load_state_dict(filtered_state_dict, strict=False)
    return checkpoint


def save_results_to_csv(results, csv_file='result/model_results.csv'):
    """Save simplified results to a CSV file"""
    try:
        # Prepare data - only include averages
        data = {
            'model': results['model'],
            'params': results['params'],
            'gflops': results['gflops'],
            'test_loss': results['test_loss'],
            'avg_dice': results['avg_dice'],
            'avg_jaccard': results['avg_jaccard'],
            'dice_necrosis': results.get('dice_necrosis', 0),
            'dice_edema': results.get('dice_edema', 0),
            'dice_enhancing': results.get('dice_enhancing', 0),
            'timestamp': results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        }

        # If the file exists, read and append/update
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            # Check if there is already a record for the same model
            mask = df['model'] == results['model']
            if mask.any():
                # Update existing record
                for key, value in data.items():
                    df.loc[mask, key] = value
            else:
                # Add new record
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        else:
            # Create new file
            df = pd.DataFrame([data])

        # Save to CSV
        df.to_csv(csv_file, index=False)
        print(f"Simplified results saved to: {csv_file}")

    except Exception as e:
        print(f"Error saving results to CSV: {e}")


def load_previous_results(csv_file='model_results.csv'):
    """Load previous results"""
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            return df
        except Exception as e:
            print(f"Error loading previous results: {e}")
    return None


def save_detailed_results(results, output_dir='results'):
    """Save detailed results to a JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{results['model']}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, 'tolist'):
                serializable_results[key] = value.tolist()
            else:
                serializable_results[key] = value

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        print(f"Detailed results saved to: {filepath}")
    except Exception as e:
        print(f"Error saving detailed results: {e}")


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    """Create a cosine learning rate scheduler with optional warmup (per-step values)."""
    scheduler = []
    total_steps = epochs * niter_per_ep
    warmup_steps = warmup_epochs * niter_per_ep

    for step in range(total_steps):
        if step < warmup_steps:
            lr = start_warmup_value + (base_value - start_warmup_value) * step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            lr = final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * progress))
        scheduler.append(lr)

    return np.array(scheduler)


# Update preprocessing pipeline
def preprocess_data():
    mask_dir = os.path.join(DATA_DIR, 'mask')
    all_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii')]

    train_files, test_files = train_test_split(all_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    # Define 2D preprocessing pipeline
    patch_size = (180, 180)
    train_transform = transforms.Compose([
        RandomRotFlip2D(),
        RandomCrop2D(patch_size),
        GaussianNoise(p=0.1),
        ContrastAugmentationTransform(p_per_sample=0.5),
        BrightnessTransform(mu=0.0, sigma=0.1, p_per_sample=0.5),
        EnsureContiguous(),  # Ensure memory contiguity
        ToTensorDict()
    ])

    val_transform = transforms.Compose([
        CenterCrop2D(patch_size),
        EnsureContiguous(),  # Ensure memory contiguity
        ToTensorDict()
    ])

    test_transform = transforms.Compose([
        CenterCrop2D(patch_size),
        EnsureContiguous(),  # Ensure memory contiguity
        ToTensorDict()
    ])

    train_dataset = BrainTumorDataset(train_files, transform=train_transform)
    val_dataset = BrainTumorDataset(val_files, transform=val_transform)
    test_dataset = BrainTumorDataset(test_files, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader