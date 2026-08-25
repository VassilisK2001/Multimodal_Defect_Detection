import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from defect_detection.data.vibration_io import load_vibration_window


def select_demo_rows(test_df: pd.DataFrame, fault_class_names: list[str], n_per_class: int = 3,
                      seed: int = 42, preferred_row_indices: Optional[list[int]] = None) -> dict[str, pd.DataFrame]:
    """Select up to n_per_class real test-set rows for each demo scenario
    ('normal' plus each fault class).

    Args:
        test_df: The full test set.
        fault_class_names: Fault class names.
        n_per_class: Number of rows to select per scenario.
        seed: Random seed for the fallback random selection.
        preferred_row_indices: Optional test_df index values to prefer over
            random selection, topped up randomly if fewer than n_per_class
            are available for a given scenario.

    Returns:
        Dict scenario_name -> DataFrame of selected rows (up to n_per_class).
    """
    preferred = set(preferred_row_indices or [])
    scenarios = {}

    normal_df = test_df[test_df.is_defect == 0]
    scenarios["normal"] = _select_rows_for_scenario(normal_df, n_per_class, seed, preferred)

    for class_name in fault_class_names:
        class_df = test_df[(test_df.is_defect == 1) & (test_df.fault_class == class_name)]
        scenarios[class_name] = _select_rows_for_scenario(class_df, n_per_class, seed, preferred)

    return scenarios


def _select_rows_for_scenario(candidate_df: pd.DataFrame, n_per_class: int, seed: int, preferred: set) -> pd.DataFrame:
    preferred_rows = candidate_df[candidate_df.index.isin(preferred)]
    n_needed = n_per_class - len(preferred_rows)

    if n_needed <= 0:
        return preferred_rows.iloc[:n_per_class]

    remaining = candidate_df[~candidate_df.index.isin(preferred)]
    topped_up = remaining.sample(n=min(n_needed, len(remaining)), random_state=seed)
    return pd.concat([preferred_rows, topped_up])


def write_demo_sample(row: pd.Series, project_root: Path, window_size: int, output_dir: Path) -> None:
    """Write one demo sample's real image and raw vibration window to
    output_dir.

    Args:
        row: A single manifest row.
        project_root: Project root, for resolving file paths.
        window_size: Vibration window size in samples.
        output_dir: Directory to write part.png and vibration_window.npy into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    source_image_path = project_root / row.image_path
    shutil.copy(source_image_path, output_dir / "part.png")

    window = load_vibration_window(row, project_root, window_size)
    np.save(output_dir / "vibration_window.npy", window)


def build_manifest_entry(scenario: str, sample_name: str, row: pd.Series) -> dict:
    """Build one manifest.json entry describing a written demo sample.

    Args:
        scenario: Scenario name ('normal', 'outer_race', 'inner_race', 'ball').
        sample_name: Sample directory name.
        row: The manifest row this sample was built from.

    Returns:
        Dict with 'scenario', 'sample', 'true_is_defect', 'true_fault_class'
        (None for normal samples).
    """
    return {
        "scenario": scenario,
        "sample": sample_name,
        "true_is_defect": bool(row.is_defect),
        "true_fault_class": row.fault_class if row.is_defect else None,
    }