"""
Computes the 5 vibration features (RMS, Peak, Crest Factor,
Spectral Kurtosis, TKEO) from a raw signal window.
"""


import numpy as np
import pandas as pd
from scipy.signal import stft
from scipy.stats import kurtosis
from scipy.io import loadmat
from defect_detection.utils import find_project_root

FEATURE_NAMES = ["RMS", "Peak", "Crest Factor", "Spectral Kurtosis", "TKEO"]

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def crest_factor(x: np.ndarray) -> float:
    r = rms(x)
    if r == 0:
        return 0.0
    return peak(x) / r


def spectral_kurtosis(x: np.ndarray, fs: int = 12000, nperseg: int = 256) -> float:
    _, _, Zxx = stft(x, fs=fs, nperseg=nperseg)
    mag = np.abs(Zxx)
    sk = kurtosis(mag, axis=1, fisher=True)
    return float(np.mean(sk))


def tkeo_energy(x: np.ndarray) -> float:
    tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
    return float(np.mean(tkeo))


def extract_features(window: np.ndarray, fs: int = 12000) -> np.ndarray:
    return np.array([
        rms(window),
        peak(window),
        crest_factor(window),
        spectral_kurtosis(window, fs=fs),
        tkeo_energy(window),
    ], dtype=np.float32)

def extract_raw_vib_features_from_df(df: pd.DataFrame, window_size: int, fs: int) -> np.ndarray:
    """Extract raw (unnormalized) vibration features for a set of manifest rows.
 
    Args:
        df: Manifest rows, with 'vibration_file' and 'vibration_window_idx' columns.
        window_size: Vibration window size in samples.
        fs: Vibration sampling rate in Hz.
 
    Returns:
        (len(df), 5) raw feature array, in df's row order.
    """
    project_root = find_project_root()
    mat_cache: dict[str, np.ndarray] = {}
    all_features = []
 
    for _, row in df.iterrows():
        if row.vibration_file not in mat_cache:
            mat_path = project_root / row.vibration_file
            mat = loadmat(mat_path)
            de_key = [k for k in mat.keys() if "DE_time" in k][0]
            mat_cache[row.vibration_file] = mat[de_key].flatten()
        signal = mat_cache[row.vibration_file]
 
        start = row.vibration_window_idx * window_size
        window = signal[start:start + window_size].astype(np.float32)
        all_features.append(extract_features(window, fs=fs))
 
    return np.stack(all_features, axis=0)