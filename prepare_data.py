# prepare_data.py
# Converts ECG CSV data into images
# for CNN training

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

print("Starting ECG data preparation...")

# ─────────────────────────────────────
# Class names for ECG categories
# ─────────────────────────────────────
CLASS_NAMES = {
    0: 'Normal',
    1: 'Supraventricular',
    2: 'Ventricular',
    3: 'Fusion',
    4: 'Unknown'
}

def create_ecg_image(signal, save_path):
    """Convert ECG signal array to image"""
    fig, ax = plt.subplots(figsize=(2.24, 2.24), dpi=100)
    ax.plot(signal, color='black', linewidth=0.8)
    ax.set_xlim(0, len(signal))
    ax.axis('off')
    fig.patch.set_facecolor('white')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight',
                pad_inches=0, facecolor='white')
    plt.close(fig)

def prepare_dataset(csv_path, output_folder,
                    samples_per_class=500):
    """
    Convert CSV ECG data to images
    samples_per_class: how many images per category
    (500 is enough for training, keeps it fast)
    """
    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, header=None)

    # Last column is the label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values.astype(int)

    print(f"Total samples: {len(y)}")
    print(f"Classes: {np.unique(y)}")

    # Create output folders for each class
    for class_id, class_name in CLASS_NAMES.items():
        folder = os.path.join(output_folder, class_name)
        os.makedirs(folder, exist_ok=True)

    # Convert signals to images
    counts = {i: 0 for i in range(5)}
    total_created = 0

    print(f"Creating ECG images "
          f"({samples_per_class} per class)...")

    for idx in tqdm(range(len(X))):
        label = y[idx]

        # Skip if we have enough for this class
        if counts[label] >= samples_per_class:
            continue

        # Create image path
        class_name = CLASS_NAMES[label]
        filename = f"ecg_{label}_{counts[label]}.png"
        save_path = os.path.join(
            output_folder, class_name, filename
        )

        # Save ECG signal as image
        create_ecg_image(X[idx], save_path)
        counts[label] += 1
        total_created += 1

    print(f"\nImages created: {total_created}")
    for class_id, class_name in CLASS_NAMES.items():
        print(f"  {class_name}: {counts[class_id]} images")

    return total_created

# ─────────────────────────────────────
# Run preparation
# ─────────────────────────────────────
print("\n=== PREPARING TRAINING DATA ===")
prepare_dataset(
    csv_path='dataset/mitbih_train.csv',
    output_folder='dataset/train',
    samples_per_class=500
)

print("\n=== PREPARING TEST DATA ===")
prepare_dataset(
    csv_path='dataset/mitbih_test.csv',
    output_folder='dataset/test',
    samples_per_class=100
)

print("\n✅ Dataset preparation complete!")
print("Check dataset/train/ and dataset/test/ folders")