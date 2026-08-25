import numpy as np
import pandas as pd
from scipy.io import savemat

from defect_detection.data.vibration_io import load_vibration_window


def _write_mat(path, signal: np.ndarray, extra_keys: dict | None = None):
    data = {"X097_DE_time": signal.reshape(-1, 1)}
    if extra_keys:
        data.update(extra_keys)
    savemat(path, data)


def test_correct_window_slice_extracted(tmp_path):
    signal = np.arange(100, dtype=np.float64)
    _write_mat(tmp_path / "signal.mat", signal)
    row = pd.Series({"vibration_file": "signal.mat", "vibration_window_idx": 2})

    result = load_vibration_window(row, tmp_path, window_size=10)

    # window_idx=2, window_size=10 -> samples [20:30]
    assert np.array_equal(result, signal[20:30].astype(np.float32))


def test_de_time_key_found_among_multiple_keys(tmp_path):
    signal = np.arange(50, dtype=np.float64)
    _write_mat(tmp_path / "signal.mat", signal, extra_keys={
        "X097_FE_time": np.zeros((50, 1)),
        "X097RPM": np.array([[1797]]),
    })
    row = pd.Series({"vibration_file": "signal.mat", "vibration_window_idx": 0})

    result = load_vibration_window(row, tmp_path, window_size=10)

    assert np.array_equal(result, signal[0:10].astype(np.float32))


def test_output_dtype_is_float32(tmp_path):
    signal = np.arange(20, dtype=np.float64)
    _write_mat(tmp_path / "signal.mat", signal)
    row = pd.Series({"vibration_file": "signal.mat", "vibration_window_idx": 0})

    result = load_vibration_window(row, tmp_path, window_size=10)

    assert result.dtype == np.float32


def test_output_shape_matches_window_size(tmp_path):
    signal = np.arange(100, dtype=np.float64)
    _write_mat(tmp_path / "signal.mat", signal)
    row = pd.Series({"vibration_file": "signal.mat", "vibration_window_idx": 1})

    result = load_vibration_window(row, tmp_path, window_size=25)

    assert result.shape == (25,)


def test_different_window_idx_gives_different_slice(tmp_path):
    signal = np.arange(60, dtype=np.float64)
    _write_mat(tmp_path / "signal.mat", signal)

    row_0 = pd.Series({"vibration_file": "signal.mat", "vibration_window_idx": 0})
    row_1 = pd.Series({"vibration_file": "signal.mat", "vibration_window_idx": 1})

    result_0 = load_vibration_window(row_0, tmp_path, window_size=20)
    result_1 = load_vibration_window(row_1, tmp_path, window_size=20)

    assert not np.array_equal(result_0, result_1)
    assert np.array_equal(result_1, signal[20:40].astype(np.float32))