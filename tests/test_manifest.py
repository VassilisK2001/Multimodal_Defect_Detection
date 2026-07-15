
import re
from pathlib import Path

import pandas as pd
import pytest
from scipy.io import loadmat

from defect_detection.data.manifest import build_manifest, load_config
from defect_detection.utils import find_project_root


@pytest.fixture(scope="module")
def manifest_df() -> pd.DataFrame:
    return build_manifest(seed=42)


@pytest.fixture(scope="module")
def project_root() -> Path:
    return find_project_root()


@pytest.fixture(scope="module")
def config() -> dict:
    return load_config()


# Structural integrity
def test_expected_columns_present(manifest_df):
    expected_cols = {
        "image_path", "category", "defect_type", "is_defect", "fault_class",
        "defect_area_ratio", "vibration_file", "vibration_severity",
        "vibration_window_idx", "fault_class_idx", "sample_id",
    }
    assert expected_cols.issubset(set(manifest_df.columns))


def test_sample_ids_unique(manifest_df):
    assert manifest_df["sample_id"].is_unique


def test_is_defect_only_binary(manifest_df):
    assert set(manifest_df["is_defect"].unique()) <= {0, 1}


def test_fault_class_idx_only_valid_values(manifest_df):
    assert set(manifest_df["fault_class_idx"].unique()) <= {-1, 0, 1, 2}


# is_defect / fault_class / fault_class_idx consistency

def test_normal_rows_have_none_fault_class(manifest_df):
    normal_rows = manifest_df[manifest_df.is_defect == 0]
    assert (normal_rows["fault_class"] == "none").all()
    assert (normal_rows["fault_class_idx"] == -1).all()


def test_defective_rows_have_valid_fault_class(manifest_df):
    defective_rows = manifest_df[manifest_df.is_defect == 1]
    assert defective_rows["fault_class"].isin(["outer_race", "inner_race", "ball"]).all()
    assert defective_rows["fault_class_idx"].isin([0, 1, 2]).all()


def test_no_nan_in_critical_columns(manifest_df):
    critical_cols = ["image_path", "vibration_file", "is_defect", "fault_class_idx"]
    for col in critical_cols:
        assert manifest_df[col].isna().sum() == 0


# Paths are relative and resolve to real files

def test_paths_are_relative(manifest_df):
    for col in ["image_path", "vibration_file"]:
        for path_str in manifest_df[col]:
            assert not Path(path_str).is_absolute(), f"Absolute path found in {col}: {path_str}"


def test_sample_of_paths_exist_on_disk(manifest_df, project_root):
    sample = manifest_df.sample(n=min(30, len(manifest_df)), random_state=0)
    for _, row in sample.iterrows():
        img_path = project_root / row.image_path
        vib_path = project_root / row.vibration_file
        assert img_path.exists(), f"Missing image file: {img_path}"
        assert vib_path.exists(), f"Missing vibration file: {vib_path}"


# Vibration window indices in bounds

def test_window_indices_in_bounds(manifest_df, project_root, config):
    window_size = config["window_size"]
    sample = manifest_df.sample(n=min(30, len(manifest_df)), random_state=1)

    for _, row in sample.iterrows():
        mat_path = project_root / row.vibration_file
        mat = loadmat(mat_path)
        de_key = [k for k in mat.keys() if "DE_time" in k][0]
        n_samples = mat[de_key].flatten().shape[0]
        n_windows = n_samples // window_size

        assert 0 <= row.vibration_window_idx < n_windows, (
            f"window_idx {row.vibration_window_idx} out of bounds "
            f"(n_windows={n_windows}) for {mat_path}"
        )


# Fault-type matching correctness
def test_vibration_fault_type_matches_assigned_fault_class(manifest_df, project_root, config):
    code_to_name = {ft["code"]: ft["name"] for ft in config["cwru"]["fault_types"]}
    defective_rows = manifest_df[manifest_df.is_defect == 1]

    for _, row in defective_rows.iterrows():
        vib_filename = Path(row.vibration_file).stem
        match = re.match(r"(B|IR|OR)(\d{3})(\d)?_(\d)", vib_filename)
        assert match is not None, f"Could not parse vibration filename: {vib_filename}"

        fault_code = match.group(1)
        actual_fault_type = code_to_name[fault_code]
        assert actual_fault_type == row.fault_class, (
            f"Mismatch: manifest says fault_class={row.fault_class}, "
            f"but vibration_file={row.vibration_file} is actually {actual_fault_type}"
        )


def test_normal_rows_use_normal_vibration_files(manifest_df, config):
    normal_prefix = config["cwru"]["normal_prefix"]
    normal_rows = manifest_df[manifest_df.is_defect == 0]
    assert normal_rows["vibration_file"].apply(
        lambda p: Path(p).stem.startswith(normal_prefix)
    ).all()


# Severity is actually randomized 
def test_severity_is_not_degenerate(manifest_df):
    defective_rows = manifest_df[manifest_df.is_defect == 1]
    distinct_severities = defective_rows["vibration_severity"].nunique()
    assert distinct_severities > 1, (
        "All defective rows have the same vibration_severity — "
        "random severity selection may be broken"
    )


# Reproducibility

def test_same_seed_produces_identical_manifest():
    df1 = build_manifest(seed=123)
    df2 = build_manifest(seed=123)

    pd.testing.assert_frame_equal(
        df1.sort_values("sample_id").reset_index(drop=True),
        df2.sort_values("sample_id").reset_index(drop=True),
    )


def test_different_seed_can_produce_different_pairing():
    df1 = build_manifest(seed=1)
    df2 = build_manifest(seed=2)

    # Same images/rows, but vibration assignment should differ for at least some rows
    df1_defective = df1[df1.is_defect == 1].sort_values("image_path").reset_index(drop=True)
    df2_defective = df2[df2.is_defect == 1].sort_values("image_path").reset_index(drop=True)

    n_different = (df1_defective["vibration_file"] != df2_defective["vibration_file"]).sum()
    assert n_different > 0, "Different seeds produced identical vibration pairing"


# Category / defect-type scoping

def test_only_configured_categories_present(manifest_df, config):
    assert set(manifest_df["category"].unique()) <= set(config["mvtec_categories"])


def test_only_mapped_defect_types_present_among_defective_rows(manifest_df, config):
    defective_rows = manifest_df[manifest_df.is_defect == 1]
    mapped_defect_types = {
        defect_type
        for category_map in config["defect_to_fault_map"].values()
        for defect_type in category_map.keys()
    }
    assert set(defective_rows["defect_type"].unique()) <= mapped_defect_types