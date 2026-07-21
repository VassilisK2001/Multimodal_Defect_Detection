
import numpy as np
import pandas as pd
import pytest
import torch

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.data.features import extract_features
from defect_detection.data.manifest import build_manifest
from defect_detection.data.normalization import (
    VibrationDataSource,
    apply_vibration_normalization,
    compute_vibration_feature_stats,
)
from defect_detection.data.splitting import split_manifest
from defect_detection.utils import find_project_root, load_yaml_config


class _FakeTrainDataset:
    """Minimal stand-in satisfying VibrationDataSource, for testing without real files."""

    def __init__(self, signals: dict[str, np.ndarray], df: pd.DataFrame, window_size: int, fs: int):
        self._signals = signals
        self.df = df
        self.window_size = window_size
        self.fs = fs

    def __len__(self) -> int:
        return len(self.df)

    def _load_de_signal(self, vibration_file: str) -> np.ndarray:
        return self._signals[vibration_file]


def test_fake_dataset_satisfies_vibration_data_source_protocol():
    """_FakeTrainDataset should satisfy the VibrationDataSource protocol."""
    fake_dataset = _FakeTrainDataset(
        signals={"a": np.zeros(256, dtype=np.float32)},
        df=pd.DataFrame({"vibration_file": ["a"], "vibration_window_idx": [0]}),
        window_size=256, fs=12000,
    )
    assert isinstance(fake_dataset, VibrationDataSource)


def test_correctness_on_synthetic_signals():
    """Mean and std should match a manual computation of the same features."""
    window_size = 256
    fs = 12000
    rng = np.random.default_rng(0)

    signal_a = rng.standard_normal(window_size * 2).astype(np.float32)
    signal_b = rng.standard_normal(window_size * 2).astype(np.float32) * 3.0  # different scale

    df = pd.DataFrame({
        "vibration_file": ["a", "a", "b", "b"],
        "vibration_window_idx": [0, 1, 0, 1],
    })
    fake_dataset = _FakeTrainDataset(
        signals={"a": signal_a, "b": signal_b}, df=df, window_size=window_size, fs=fs,
    )

    mean, std = compute_vibration_feature_stats(fake_dataset)

    # Manually recompute the same 4 windows' features and compare
    expected_features = []
    for _, row in df.iterrows():
        signal = fake_dataset._signals[row.vibration_file]
        start = row.vibration_window_idx * window_size
        window = signal[start:start + window_size]
        expected_features.append(extract_features(window, fs=fs))
    expected_features = np.stack(expected_features, axis=0)

    assert np.allclose(mean, expected_features.mean(axis=0), atol=1e-5)
    expected_std = expected_features.std(axis=0)
    expected_std[expected_std == 0] = 1.0
    assert np.allclose(std, expected_std, atol=1e-5)


def test_output_shape():
    """Mean and std should each have shape (5,)."""
    window_size = 2048   
    fs = 12000
    signal = np.random.default_rng(1).standard_normal(window_size * 3).astype(np.float32)

    df = pd.DataFrame({"vibration_file": ["a"] * 3, "vibration_window_idx": [0, 1, 2]})
    fake_dataset = _FakeTrainDataset({"a": signal}, df, window_size, fs)

    mean, std = compute_vibration_feature_stats(fake_dataset)
    assert mean.shape == (5,)
    assert std.shape == (5,)


def test_zero_variance_guard():
    """A feature with zero variance across all windows should get std=1.0, not 0.0."""
    window_size = 256
    fs = 12000
    base_window = np.random.default_rng(2).standard_normal(window_size).astype(np.float32)
    signal = np.concatenate([base_window, base_window, base_window])

    df = pd.DataFrame({"vibration_file": ["a"] * 3, "vibration_window_idx": [0, 1, 2]})
    fake_dataset = _FakeTrainDataset({"a": signal}, df, window_size, fs)

    _, std = compute_vibration_feature_stats(fake_dataset)
    assert not np.any(std == 0.0)
    assert np.allclose(std, 1.0)


def test_deterministic_across_calls():
    """Repeated calls on the same data should return identical results."""
    window_size = 2048  # comfortably larger than spectral_kurtosis's internal STFT
                         # window (nperseg=256), to avoid a scipy fallback warning
    fs = 12000
    signal = np.random.default_rng(3).standard_normal(window_size * 4).astype(np.float32)
    df = pd.DataFrame({"vibration_file": ["a"] * 4, "vibration_window_idx": [0, 1, 2, 3]})
    fake_dataset = _FakeTrainDataset({"a": signal}, df, window_size, fs)

    mean1, std1 = compute_vibration_feature_stats(fake_dataset)
    mean2, std2 = compute_vibration_feature_stats(fake_dataset)

    assert np.allclose(mean1, mean2)
    assert np.allclose(std1, std2)


def test_apply_normalization_correctness_with_distinct_values():
    """Output should match (features - mean) / std, using distinct per-feature values
    so a swapped-axis or broadcasting bug would produce a detectably wrong result."""
    features = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    std = np.array([1.0, 2.0, 1.0, 0.5, 10.0], dtype=np.float32)

    result = apply_vibration_normalization(features, mean, std)
    expected = (features - mean) / std
    assert np.allclose(result, expected)


def test_apply_normalization_output_shape():
    features = np.random.default_rng(4).standard_normal(5).astype(np.float32)
    mean = np.zeros(5, dtype=np.float32)
    std = np.ones(5, dtype=np.float32)

    result = apply_vibration_normalization(features, mean, std)
    assert result.shape == (5,)


def test_apply_normalization_recovers_zero_mean_unit_std():
    """Normalizing a set of samples with their own true mean/std should yield
    approximately mean 0 and std 1."""
    rng = np.random.default_rng(5)
    samples = rng.normal(loc=[10, -5, 3, 0, 100], scale=[2, 1, 5, 0.1, 20], size=(200, 5))

    true_mean = samples.mean(axis=0)
    true_std = samples.std(axis=0)

    normalized = np.stack([
        apply_vibration_normalization(row, true_mean, true_std) for row in samples
    ])

    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-6)


def test_apply_normalization_composes_with_compute_stats():
    """Chaining compute_vibration_feature_stats and apply_vibration_normalization
    directly should match manually normalizing with the same computed stats."""
    window_size = 2048
    fs = 12000
    signal = np.random.default_rng(6).standard_normal(window_size * 5).astype(np.float32)
    df = pd.DataFrame({"vibration_file": ["a"] * 5, "vibration_window_idx": [0, 1, 2, 3, 4]})
    fake_dataset = _FakeTrainDataset({"a": signal}, df, window_size, fs)

    mean, std = compute_vibration_feature_stats(fake_dataset)

    raw_features = extract_features(signal[0:window_size], fs=fs)
    result = apply_vibration_normalization(raw_features, mean, std)
    expected = (raw_features - mean) / std

    assert np.allclose(result, expected)


def test_apply_normalization_does_not_mutate_inputs():
    """features, mean, and std should not be modified in place."""
    features = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    std = np.array([1.0, 2.0, 1.0, 0.5, 10.0], dtype=np.float32)

    features_copy = features.copy()
    mean_copy = mean.copy()
    std_copy = std.copy()

    apply_vibration_normalization(features, mean, std)

    assert np.array_equal(features, features_copy)
    assert np.array_equal(mean, mean_copy)
    assert np.array_equal(std, std_copy)


# End-to-end integration with MultimodalDefectDataset

@pytest.fixture(scope="module")
def small_train_df() -> pd.DataFrame:
    """A small real training subset, for a fast integration check."""
    config = load_yaml_config("config/data_config.yaml")
    manifest_df = build_manifest()
    split_df = split_manifest(manifest_df, seed=42)
    train_df = split_df[split_df.split == "train"].drop(columns=["split"])
    return train_df.head(40).reset_index(drop=True)


def test_normalized_features_have_approx_zero_mean_unit_std(small_train_df):
    """Applying computed stats via MultimodalDefectDataset should yield features
    with approximately mean 0 and std 1 on the data they were computed from."""
    config = load_yaml_config("config/data_config.yaml")
    window_size = config["window_size"]
    fs = config["cwru"]["sampling_rate_hz"]

    raw_dataset = MultimodalDefectDataset(
        small_train_df, window_size=window_size, fs=fs, training=False,
    )
    mean, std = compute_vibration_feature_stats(raw_dataset)

    norm_dataset = MultimodalDefectDataset(
        small_train_df, window_size=window_size, fs=fs, training=False,
        vib_mean=mean, vib_std=std,
    )

    all_features = torch.stack([norm_dataset[i][1] for i in range(len(norm_dataset))])

    assert torch.allclose(all_features.mean(dim=0), torch.zeros(5), atol=1e-4)
    assert torch.allclose(all_features.std(dim=0, unbiased=False), torch.ones(5), atol=1e-3)