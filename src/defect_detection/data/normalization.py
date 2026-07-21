
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from defect_detection.data.features import extract_features


@runtime_checkable
class VibrationDataSource(Protocol):
    """Structural interface required by compute_vibration_feature_stats."""

    df: pd.DataFrame
    window_size: int
    fs: int

    def __len__(self) -> int: ...

    def _load_de_signal(self, vibration_file: str) -> np.ndarray: ...


def compute_vibration_feature_stats(train_dataset: VibrationDataSource) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-feature mean and standard deviation across a dataset.

    Args:
        train_dataset: An object satisfying VibrationDataSource, typically built
            without augmentation or normalization applied.

    Returns:
        A (mean, std) tuple, each a (5,) array. Zero-variance features get std=1.0
        to avoid division by zero when the stats are later used for normalization.
    """
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
    std[std == 0] = 1.0
    return mean, std


def apply_vibration_normalization(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize a feature vector using precomputed mean and standard deviation.

    Args:
        features: A (5,) array of raw extracted features.
        mean: A (5,) array, typically from compute_vibration_feature_stats.
        std: A (5,) array, typically from compute_vibration_feature_stats.

    Returns:
        A (5,) array: (features - mean) / std.
    """
    return (features - mean) / std