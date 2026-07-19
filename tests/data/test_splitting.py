
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

from defect_detection.data.splitting import (
    _compute_split_blocks, _get_n_windows, _redraw_window_indices, split_manifest,
)
from defect_detection.utils import find_project_root, load_yaml_config


@pytest.fixture(scope="module")
def config() -> dict:
    return load_yaml_config("config/data_config.yaml")


@pytest.fixture(scope="module")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="module")
def manifest_df(project_root, config) -> pd.DataFrame:
    manifest_path = project_root / config["paths"]["manifest_dir"] / "manifest.csv"
    return pd.read_csv(manifest_path)


@pytest.fixture(scope="module")
def split_df(manifest_df) -> pd.DataFrame:
    return split_manifest(manifest_df, seed=42)


# No window-index overlap across splits for the same file

def test_no_window_index_overlap_across_splits(split_df):
    """For every vibration_file used in more than one split, the set of window indices
    used by train, val, and test must be pairwise disjoint"""
    for vib_file, group in split_df.groupby("vibration_file"):
        indices_by_split = {
            split_name: set(sub_group["vibration_window_idx"])
            for split_name, sub_group in group.groupby("split")
        }
        split_names = list(indices_by_split.keys())
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                overlap = indices_by_split[split_names[i]] & indices_by_split[split_names[j]]
                assert not overlap, (
                    f"Window index overlap for {vib_file} between "
                    f"{split_names[i]} and {split_names[j]}: {overlap}"
                )


# Window indices fall within their split's computed block

def test_window_indices_within_correct_block(split_df, project_root, config):
    """Each row's vibration_window_idx must fall within the block boundaries computed
    for its own split"""
    train_frac = config["split"]["train_frac"]
    val_frac = config["split"]["val_frac"]
    window_size = config["window_size"]

    sample = split_df.groupby(["vibration_file", "split"]).head(3)
    n_windows_cache = {}

    for _, row in sample.iterrows():
        if row.vibration_file not in n_windows_cache:
            mat_path = project_root / row.vibration_file
            n_windows_cache[row.vibration_file] = _get_n_windows(mat_path, window_size)
        n_windows = n_windows_cache[row.vibration_file]

        blocks = _compute_split_blocks(n_windows, train_frac, val_frac)
        block_start, block_end = blocks[row.split]

        assert block_start <= row.vibration_window_idx < block_end, (
            f"Row for {row.vibration_file} in split={row.split} has window_idx="
            f"{row.vibration_window_idx}, outside block [{block_start}, {block_end})"
        )


# Window indices within the file's actual bounds

def test_window_indices_in_file_bounds(split_df, project_root, config):
    """No vibration_window_idx should be negative or exceed the file's actual n_windows,
    re-verified after redrawing"""
    window_size = config["window_size"]
    sample = split_df.sample(n=min(40, len(split_df)), random_state=0)

    for _, row in sample.iterrows():
        mat_path = project_root / row.vibration_file
        n_windows = _get_n_windows(mat_path, window_size)
        assert 0 <= row.vibration_window_idx < n_windows


# Row count and sample_id preservation

def test_all_rows_preserved_no_duplicates(manifest_df, split_df):
    """split_manifest must not drop or duplicate rows: the union of sample_ids across
    train/val/test must exactly equal the original manifest's sample_ids."""
    assert len(split_df) == len(manifest_df)
    assert set(split_df["sample_id"]) == set(manifest_df["sample_id"])
    assert split_df["sample_id"].is_unique


# Stratification balance

def test_is_defect_ratio_consistent_across_splits(split_df):
    """The proportion of defective samples should be similar across train/val/test,
    the direct purpose of stratifying on is_defect."""
    ratios = split_df.groupby("split")["is_defect"].mean()
    assert ratios.max() - ratios.min() < 0.05, (
        f"is_defect ratio varies too much across splits: {ratios.to_dict()}"
    )


def test_fault_class_proportions_consistent_across_splits(split_df):
    """Each fault type's share of defective samples should be similar across splits,
    the direct purpose of stratifying on the combined is_defect + fault_class key."""
    defective = split_df[split_df.is_defect == 1]
    proportions = (
        defective.groupby("split")["fault_class"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
    )
    for fault_class in proportions.columns:
        spread = proportions[fault_class].max() - proportions[fault_class].min()
        assert spread < 0.15, (
            f"{fault_class} proportion varies too much across splits: "
            f"{proportions[fault_class].to_dict()}"
        )


# Split ratios approximately match config

def test_split_sizes_approximately_match_config(split_df, config):
    total = len(split_df)
    actual_fracs = split_df["split"].value_counts(normalize=True)

    for split_name in ["train", "val", "test"]:
        expected = config["split"][f"{split_name}_frac"]
        actual = actual_fracs[split_name]
        assert abs(actual - expected) < 0.03, (
            f"{split_name}: expected ~{expected:.2f}, got {actual:.2f}"
        )


# Reproducibility

def test_same_seed_produces_identical_split(manifest_df):
    df1 = split_manifest(manifest_df, seed=123)
    df2 = split_manifest(manifest_df, seed=123)

    df1_sorted = df1.sort_values("sample_id").reset_index(drop=True)
    df2_sorted = df2.sort_values("sample_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(df1_sorted, df2_sorted)


# Replacement-sampling fallback, tested with a controlled synthetic case

def test_replacement_fallback_triggers_when_block_smaller_than_demand(project_root, config, monkeypatch):
    """When more rows need a window from a given file+split block than the block has
    distinct indices, indices must be drawn with replacement for that block (duplicates
    allowed)"""
    window_size = config["window_size"]

    # Fake a tiny file with only 10 windows total: train block = 7, val = 1, test = 2
    monkeypatch.setattr(
        "defect_detection.data.splitting._get_n_windows",
        lambda mat_path, ws: 10,
    )

    # Construct a synthetic manifest: 5 rows all assigned to the "val" split, sharing one
    # vibration file, val's block only has 1 valid index, so replacement is required.
    fake_df = pd.DataFrame({
        "vibration_file": ["fake_file.mat"] * 5,
        "split": ["val"] * 5,
        "vibration_window_idx": [-1] * 5,
    })

    rng = np.random.default_rng(0)
    result = _redraw_window_indices(
        fake_df, project_root, window_size, rng, train_frac=0.7, val_frac=0.1,
    )

    # val block for n_windows=10, train_frac=0.7, val_frac=0.1 -> block = [7, 8), size 1
    assert result["vibration_window_idx"].nunique() == 1
    assert (result["vibration_window_idx"] == 7).all()