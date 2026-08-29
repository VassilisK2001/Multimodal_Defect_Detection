
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import loadmat

from defect_detection.data.splitting import (
    _compute_split_blocks, 
    _get_n_windows, 
    _redraw_window_indices, 
    generate_stratified_kfold_splits,
    split_manifest,
    select_files_to_hold_out,
    build_leave_file_out_split,
    _redraw_full_range_indices
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


def test_window_indices_in_file_bounds(split_df, project_root, config):
    """No vibration_window_idx should be negative or exceed the file's actual n_windows,
    re-verified after redrawing"""
    window_size = config["window_size"]
    sample = split_df.sample(n=min(40, len(split_df)), random_state=0)

    for _, row in sample.iterrows():
        mat_path = project_root / row.vibration_file
        n_windows = _get_n_windows(mat_path, window_size)
        assert 0 <= row.vibration_window_idx < n_windows


def test_all_rows_preserved_no_duplicates(manifest_df, split_df):
    """split_manifest must not drop or duplicate rows: the union of sample_ids across
    train/val/test must exactly equal the original manifest's sample_ids."""
    assert len(split_df) == len(manifest_df)
    assert set(split_df["sample_id"]) == set(manifest_df["sample_id"])
    assert split_df["sample_id"].is_unique


def test_is_defect_ratio_consistent_across_splits(split_df):
    """The proportion of defective samples should be similar across train/val/test,
    the direct purpose of stratifying on is_defect."""
    ratios = split_df.groupby("split")["is_defect"].mean()
    assert ratios.max() - ratios.min() < 0.05, (
        f"is_defect ratio varies too much across splits: {ratios.to_dict()}"
    )


def test_fault_class_proportions_consistent_across_splits(split_df):
    """Each fault type's share of defective samples should be similar across splits."""
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


def test_split_sizes_approx_match_config(split_df, config):
    total = len(split_df)
    actual_fracs = split_df["split"].value_counts(normalize=True)

    for split_name in ["train", "val", "test"]:
        expected = config["split"][f"{split_name}_frac"]
        actual = actual_fracs[split_name]
        assert abs(actual - expected) < 0.03, (
            f"{split_name}: expected ~{expected:.2f}, got {actual:.2f}"
        )


def test_same_seed_produces_identical_split(manifest_df):
    df1 = split_manifest(manifest_df, seed=123)
    df2 = split_manifest(manifest_df, seed=123)

    df1_sorted = df1.sort_values("sample_id").reset_index(drop=True)
    df2_sorted = df2.sort_values("sample_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(df1_sorted, df2_sorted)


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


def test_kfold_test_sets_partition_the_manifest(manifest_df):
    """Every sample_id must appear as 'test' in exactly one fold."""
    k = 3
    folds = generate_stratified_kfold_splits(manifest_df, k=k, seed=42)
 
    test_id_counts = {}
    for fold_df in folds:
        test_ids = fold_df[fold_df.split == "test"]["sample_id"]
        for sample_id in test_ids:
            test_id_counts[sample_id] = test_id_counts.get(sample_id, 0) + 1
 
    assert set(test_id_counts.keys()) == set(manifest_df["sample_id"])
    assert all(count == 1 for count in test_id_counts.values())
 
 
def test_kfold_reproducible_with_same_seed(manifest_df):
    folds_a = generate_stratified_kfold_splits(manifest_df, k=3, seed=123)
    folds_b = generate_stratified_kfold_splits(manifest_df, k=3, seed=123)
 
    for fold_a, fold_b in zip(folds_a, folds_b):
        a_sorted = fold_a.sort_values("sample_id").reset_index(drop=True)
        b_sorted = fold_b.sort_values("sample_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(a_sorted, b_sorted)
 
 
def test_kfold_folds_differ_from_each_other(manifest_df):
    folds = generate_stratified_kfold_splits(manifest_df, k=3, seed=42)
 
    test_sets = [set(fold_df[fold_df.split == "test"]["sample_id"]) for fold_df in folds]
 
    assert test_sets[0] != test_sets[1]
    assert test_sets[1] != test_sets[2]
    assert test_sets[0] != test_sets[2]
 
 
def test_kfold_split_sizes_approximately_correct(manifest_df, config):
    k = 3
    folds = generate_stratified_kfold_splits(manifest_df, k=k, seed=42)
    val_frac = config["split"]["val_frac"]
 
    for fold_df in folds:
        actual_fracs = fold_df["split"].value_counts(normalize=True)
 
        assert abs(actual_fracs["test"] - (1 / k)) < 0.05, (
            f"test fraction {actual_fracs['test']:.2f} far from expected {1/k:.2f}"
        )
        assert abs(actual_fracs["val"] - val_frac) < 0.05, (
            f"val fraction {actual_fracs['val']:.2f} far from expected {val_frac:.2f}"
        )
 
 
def test_kfold_stratification_balance_within_fold(manifest_df):
    """is_defect ratio should be similar across train/val/test within each fold."""
    folds = generate_stratified_kfold_splits(manifest_df, k=3, seed=42)
 
    for fold_idx, fold_df in enumerate(folds):
        ratios = fold_df.groupby("split")["is_defect"].mean()
        assert ratios.max() - ratios.min() < 0.05, (
            f"fold {fold_idx}: is_defect ratio varies too much across splits: {ratios.to_dict()}"
        )

def test_select_files_returns_one_valid_file_per_group(manifest_df):
    """Should return exactly one held-out file per fault class plus one normal
    file."""
    held_out = select_files_to_hold_out(manifest_df, seed=42)

    assert set(held_out.keys()) == {"outer_race", "inner_race", "ball", "normal"}

    for fault_class in ("outer_race", "inner_race", "ball"):
        vib_file = held_out[fault_class]
        actual_classes = manifest_df[manifest_df.vibration_file == vib_file]["fault_class"].unique()
        assert list(actual_classes) == [fault_class]

    normal_file = held_out["normal"]
    actual_is_defect = manifest_df[manifest_df.vibration_file == normal_file]["is_defect"].unique()
    assert list(actual_is_defect) == [0]
 
 
def test_select_files_reproducible_with_same_seed(manifest_df):
    held_out_a = select_files_to_hold_out(manifest_df, seed=123)
    held_out_b = select_files_to_hold_out(manifest_df, seed=123)
 
    assert held_out_a == held_out_b
 
  
def test_redraw_full_range_not_restricted_to_a_block(project_root, config, monkeypatch):
    """Drawn indices should span the file's entire window range, not a
    train/val/test block boundary."""
    window_size = config["window_size"]
 
    monkeypatch.setattr("defect_detection.data.splitting._get_n_windows", lambda mat_path, ws: 10)
 
    fake_df = pd.DataFrame({
        "vibration_file": ["fake_file.mat"] * 20,
        "vibration_window_idx": [-1] * 20,
    })
 
    rng = np.random.default_rng(0)
    result = _redraw_full_range_indices(fake_df, project_root, window_size, rng)
 
    # With only 10 total windows, indices covering the full [0, 10) range should appear.
    assert result["vibration_window_idx"].min() < 7
    assert result["vibration_window_idx"].max() <= 9
 
  
def test_held_out_files_never_appear_in_train_or_val(manifest_df):
    held_out_files = select_files_to_hold_out(manifest_df, seed=42)
    train_df, val_df, held_out_test_df = build_leave_file_out_split(manifest_df, held_out_files, seed=42)
 
    held_out_set = set(held_out_files.values())
 
    assert not set(train_df["vibration_file"]) & held_out_set
    assert not set(val_df["vibration_file"]) & held_out_set
    assert set(held_out_test_df["vibration_file"]) <= held_out_set
 
 
def test_build_leave_file_out_split_preserves_all_rows_no_duplicates(manifest_df):
    held_out_files = select_files_to_hold_out(manifest_df, seed=42)
    train_df, val_df, held_out_test_df = build_leave_file_out_split(manifest_df, held_out_files, seed=42)
 
    combined_ids = (
        set(train_df["sample_id"]) | set(val_df["sample_id"]) | set(held_out_test_df["sample_id"])
    )
    assert combined_ids == set(manifest_df["sample_id"])
    assert len(train_df) + len(val_df) + len(held_out_test_df) == len(manifest_df)
 