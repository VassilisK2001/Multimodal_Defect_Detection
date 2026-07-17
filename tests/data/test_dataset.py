
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from defect_detection.data.dataset import MultimodalDefectDataset
from defect_detection.data.manifest import load_config
from defect_detection.utils import find_project_root


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="module")
def manifest_df(project_root, config) -> pd.DataFrame:
    manifest_path = project_root / config["paths"]["manifest_dir"] / "manifest.csv"
    return pd.read_csv(manifest_path)


@pytest.fixture(scope="module")
def eval_dataset(manifest_df, config) -> MultimodalDefectDataset:
    return MultimodalDefectDataset(
        manifest_df, window_size=config["window_size"],
        fs=config["cwru"]["sampling_rate_hz"], training=False,
    )


@pytest.fixture(scope="module")
def train_dataset(manifest_df, config) -> MultimodalDefectDataset:
    return MultimodalDefectDataset(
        manifest_df, window_size=config["window_size"],
        fs=config["cwru"]["sampling_rate_hz"], training=True,
    )


# __len__

def test_len_matches_manifest_row_count(eval_dataset, manifest_df):
    """The Dataset should expose exactly one item per manifest row, no dropped/duplicated rows."""
    assert len(eval_dataset) == len(manifest_df)


# __getitem__ shapes and dtypes

def test_getitem_returns_expected_shapes_and_dtypes(eval_dataset):
    """Output must match what the model architecture expects: (3,224,224) image,
    (5,) vibration features, scalar float32 is_defect, scalar long fault_class_idx,
    scalar float32 area_ratio."""
    image, vib, is_defect, fault_idx, area = eval_dataset[0]

    assert image.shape == (3, 224, 224)
    assert isinstance(image, torch.Tensor)

    assert vib.shape == (5,)
    assert vib.dtype == torch.float32

    assert is_defect.dtype == torch.float32
    assert is_defect.dim() == 0

    assert fault_idx.dtype == torch.long
    assert fault_idx.dim() == 0

    assert area.dtype == torch.float32
    assert area.dim() == 0


def test_getitem_no_nans_in_features(eval_dataset):
    """A real sample's extracted vibration features should never contain NaN/inf"""
    _, vib, _, _, _ = eval_dataset[0]
    assert not torch.isnan(vib).any()
    assert not torch.isinf(vib).any()


# is_defect / fault_class_idx consistency survives the Dataset layer

def test_normal_rows_have_sentinel_fault_class_idx(eval_dataset, manifest_df):
    """Rows where is_defect == 0 in the manifest must come out of the Dataset with
    fault_class_idx == -1, matching the masking convention TwoStageLoss depends on."""
    normal_indices = manifest_df.index[manifest_df.is_defect == 0][:10]
    for idx in normal_indices:
        _, _, is_defect, fault_idx, _ = eval_dataset[idx]
        assert is_defect.item() == 0.0
        assert fault_idx.item() == -1


def test_defective_rows_have_valid_fault_class_idx(eval_dataset, manifest_df):
    """Rows where is_defect == 1 must come out with fault_class_idx in {0, 1, 2}."""
    defective_indices = manifest_df.index[manifest_df.is_defect == 1][:10]
    for idx in defective_indices:
        _, _, is_defect, fault_idx, _ = eval_dataset[idx]
        assert is_defect.item() == 1.0
        assert fault_idx.item() in {0, 1, 2}


# training vs. eval mode

def test_eval_mode_is_deterministic(eval_dataset):
    """No augmentation in eval mode, so fetching the same index twice must give identical
    vibration features and identical images."""
    image1, vib1, _, _, _ = eval_dataset[0]
    image2, vib2, _, _, _ = eval_dataset[0]

    assert torch.allclose(vib1, vib2)
    assert torch.allclose(image1, image2)


def test_training_mode_applies_vibration_augmentation(train_dataset):
    """jitter/scale are applied to the vibration window before feature extraction in
    training mode, so fetching the same index twice should generally yield different
    vibration features."""
    _, vib1, _, _, _ = train_dataset[0]
    _, vib2, _, _, _ = train_dataset[0]

    assert not torch.allclose(vib1, vib2)


# .mat caching

def test_mat_file_is_cached_after_first_access(eval_dataset, manifest_df):
    """Accessing two rows that share the same vibration_file should only populate one
    cache entry, confirming the cache key is the file path and repeated reads are avoided."""
    # Find two rows sharing the same vibration_file
    vib_file_counts = manifest_df["vibration_file"].value_counts()
    shared_file = vib_file_counts[vib_file_counts > 1].index[0]
    matching_rows = manifest_df.index[manifest_df["vibration_file"] == shared_file][:2]

    eval_dataset._mat_cache.clear()
    for idx in matching_rows:
        eval_dataset[idx]

    assert shared_file in eval_dataset._mat_cache
    assert len(eval_dataset._mat_cache) == 1


# Vibration normalization

def test_normalization_applied_when_stats_provided(manifest_df, config):
    """When vib_mean/vib_std are provided, output features should equal
    (raw_features - mean) / std, not the raw extracted values."""
    row_idx = manifest_df.index[manifest_df.is_defect == 0][0]

    raw_dataset = MultimodalDefectDataset(
        manifest_df, window_size=config["window_size"],
        fs=config["cwru"]["sampling_rate_hz"], training=False,
    )
    _, raw_vib, _, _, _ = raw_dataset[row_idx]

    fake_mean = np.ones(5, dtype=np.float32) * 0.1
    fake_std = np.ones(5, dtype=np.float32) * 2.0

    norm_dataset = MultimodalDefectDataset(
        manifest_df, window_size=config["window_size"],
        fs=config["cwru"]["sampling_rate_hz"], training=False,
        vib_mean=fake_mean, vib_std=fake_std,
    )
    _, norm_vib, _, _, _ = norm_dataset[row_idx]

    expected = (raw_vib.numpy() - fake_mean) / fake_std
    assert np.allclose(norm_vib.numpy(), expected, atol=1e-5)


def test_normalization_not_applied_when_stats_are_none(eval_dataset, manifest_df):
    """Default behavior (vib_mean=None, vib_std=None) should leave features unnormalized"""
    row_idx = manifest_df.index[manifest_df.is_defect == 0][0]
    _, vib, _, _, _ = eval_dataset[row_idx]
    assert not torch.allclose(vib, torch.zeros(5), atol=0.5)


# Real end-to-end sample

def test_real_samples_load_without_error(eval_dataset):
    """Sanity check across a broader sample of real manifest rows: the full pipeline
    (path resolution, image load, transform, .mat load, windowing, feature extraction)
    should run cleanly end-to-end"""
    n = len(eval_dataset)
    sample_indices = np.linspace(0, n - 1, num=min(20, n), dtype=int)
    for idx in sample_indices:
        image, vib, is_defect, fault_idx, area = eval_dataset[int(idx)]
        assert image is not None
        assert vib is not None