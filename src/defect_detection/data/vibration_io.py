from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


def load_vibration_window(row: pd.Series, project_root: Path, window_size: int) -> np.ndarray:
    """Load one raw vibration window for a single manifest row.

    Args:
        row: A manifest row, with 'vibration_file' and 'vibration_window_idx'
            columns.
        project_root: Project root, for resolving the .mat file path.
        window_size: Vibration window size in samples.

    Returns:
        (window_size,) raw vibration window.
    """
    mat_path = project_root / row.vibration_file
    mat = loadmat(mat_path)
    de_key = [k for k in mat.keys() if "DE_time" in k][0]
    signal = mat[de_key].flatten()

    start = row.vibration_window_idx * window_size
    return signal[start:start + window_size].astype(np.float32)