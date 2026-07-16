"""
src/defect_detection/data/splitting.py

Train/val/test split logic and post-split normalization statistics. Kept separate from
dataset.py since these are split-time operations, not part of the Dataset class itself,
and separate from manifest.py since they operate on an already-built manifest rather than
constructing one.
"""

import numpy as np

from defect_detection.data.features import extract_features


def compute_vibration_feature_stats(train_dataset) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean/std across the training split only,for later use in normalizing        
    train/val/test consistently."""
    all_features = []
    for idx in range(len(train_dataset)):
        row = train_dataset.df.iloc[idx]
        signal = train_dataset._load_de_signal(row.vibration_file)
        start = row.vibration_window_idx * train_dataset.window_size
        window = signal[start:start + train_dataset.window_size].astype(np.float32)
        all_features.append(extract_features(window, fs=train_dataset.fs))

    all_features = np.stack(all_features, axis=0)
    mean = all_features.mean(axis=0)
    std = all_features.std(axis=0)
    std[std == 0] = 1.0  # guard against a zero-variance feature causing division by zero
    return mean, std


# TO_DO: Train/val/test split function to be added here next: stratify on combined
# is_defect + fault_class key, split CWRU at the file level first, window only
# within each split