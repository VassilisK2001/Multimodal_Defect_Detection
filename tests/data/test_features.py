import pathlib

import numpy as np
import pandas as pd
import pytest

from defect_detection.data.features import (
    rms, 
    peak, 
    crest_factor, 
    spectral_kurtosis, 
    tkeo_energy, 
    extract_features,
    extract_raw_vib_features_from_df,
)


def test_rms_constant_signal():
    x = np.full(100, 3.0)
    assert rms(x) == pytest.approx(3.0)


def test_rms_zero_signal():
    x = np.zeros(100)
    assert rms(x) == 0.0


def test_peak_simple():
    x = np.array([1.0, -5.0, 3.0, 2.0])
    assert peak(x) == 5.0


def test_crest_factor_impulsive_vs_flat():
    flat = np.ones(2048) * 0.5
    impulsive = np.zeros(2048)
    impulsive[1024] = 10.0  # single sharp spike

    assert crest_factor(impulsive) > crest_factor(flat)


def test_crest_factor_zero_signal_no_crash():
    x = np.zeros(100)
    assert crest_factor(x) == 0.0

def test_spectral_kurtosis_higher_for_impulsive_transients():
    np.random.seed(0)
    n = 4096
    fs = 12000

    # Gaussian noise: roughly flat energy across time/frequency
    gaussian_noise = np.random.randn(n).astype(np.float32)

    # Same noise floor, plus a few sharp periodic impulses
    impulsive = gaussian_noise.copy()
    impulse_positions = np.arange(200, n - 200, 400)
    impulsive[impulse_positions] += 15.0

    sk_gaussian = spectral_kurtosis(gaussian_noise, fs=fs)
    sk_impulsive = spectral_kurtosis(impulsive, fs=fs)

    assert sk_impulsive > sk_gaussian

def test_tkeo_energy_shape_and_type():
    x = np.random.randn(2048).astype(np.float32)
    result = tkeo_energy(x)
    assert isinstance(result, float)


def test_extract_features_output_shape():
    x = np.random.randn(2048).astype(np.float32)
    features = extract_features(x)
    assert features.shape == (5,)
    assert features.dtype == np.float32


def test_extract_features_no_nans():
    x = np.random.randn(2048).astype(np.float32)
    features = extract_features(x)
    assert not np.isnan(features).any()

def _mock_mat(signal: np.ndarray) -> dict:
    return {"X100_DE_time": signal.reshape(-1, 1)}
 
 
def _patch_project_root(monkeypatch, root: str = "/fake"):
    monkeypatch.setattr("defect_detection.data.features.find_project_root", lambda: pathlib.Path(root))
 
 
def test_extract_raw_vib_features_matches_manual_extraction(monkeypatch):
    np.random.seed(0)
    signal = np.random.randn(4096).astype(np.float32)
 
    _patch_project_root(monkeypatch)
    monkeypatch.setattr("defect_detection.data.features.loadmat", lambda path: _mock_mat(signal))
 
    df = pd.DataFrame({"vibration_file": ["file_a.mat"], "vibration_window_idx": [1]})
    window_size = 1024
 
    result = extract_raw_vib_features_from_df(df, window_size=window_size, fs=12000)
 
    expected_window = signal[window_size:window_size * 2]
    expected = extract_features(expected_window, fs=12000)
    assert np.allclose(result[0], expected)
 
 
def test_extract_raw_vib_features_loads_each_file_once(monkeypatch):
    call_count = {"n": 0}
 
    def counting_loadmat(path):
        call_count["n"] += 1
        return _mock_mat(np.random.randn(4096).astype(np.float32))
 
    _patch_project_root(monkeypatch)
    monkeypatch.setattr("defect_detection.data.features.loadmat", counting_loadmat)
 
    df = pd.DataFrame({
        "vibration_file": ["file_a.mat", "file_a.mat", "file_b.mat", "file_a.mat"],
        "vibration_window_idx": [0, 1, 0, 2],
    })
 
    extract_raw_vib_features_from_df(df, window_size=512, fs=12000)
 
    assert call_count["n"] == 2
 
 
def test_extract_raw_vib_features_preserves_row_order(monkeypatch):
    signal_a = np.full(4096, 1.0, dtype=np.float32)
    signal_b = np.full(4096, 100.0, dtype=np.float32)
    mats = {"file_a.mat": _mock_mat(signal_a), "file_b.mat": _mock_mat(signal_b)}
 
    _patch_project_root(monkeypatch)
    monkeypatch.setattr("defect_detection.data.features.loadmat", lambda path: mats[path.name])
 
    df = pd.DataFrame({
        "vibration_file": ["file_b.mat", "file_a.mat"],
        "vibration_window_idx": [0, 0],
    })
 
    result = extract_raw_vib_features_from_df(df, window_size=512, fs=12000)
 
    assert result[0][0] > result[1][0]
 
 
def test_extract_raw_vib_features_output_shape(monkeypatch):
    _patch_project_root(monkeypatch)
    monkeypatch.setattr(
        "defect_detection.data.features.loadmat",
        lambda path: _mock_mat(np.random.randn(4096).astype(np.float32)),
    )
 
    df = pd.DataFrame({
        "vibration_file": ["file_a.mat"] * 3,
        "vibration_window_idx": [0, 1, 2],
    })
 
    result = extract_raw_vib_features_from_df(df, window_size=512, fs=12000)
 
    assert result.shape == (3, 5)
 