import numpy as np
from scipy.signal import stft
from scipy.stats import kurtosis


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