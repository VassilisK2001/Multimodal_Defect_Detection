import numpy as np
from PIL import Image
from scipy.signal import stft
from scipy.stats import kurtosis

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FEATURE_NAMES = ["RMS", "Peak", "Crest Factor", "Spectral Kurtosis", "TKEO"]


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess a PIL image.

    Args:
        image: A PIL image, any mode/size.

    Returns:
        (1, 3, 224, 224) float32 array, ready for the ONNX session's
        'image' input.
    """
    image = image.convert("RGB")
    image = image.resize((224, 224), Image.Resampling.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    array = np.transpose(array, (2, 0, 1))
    return np.expand_dims(array, axis=0).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x)))


def crest_factor(x: np.ndarray) -> float:
    r = rms(x)
    if r == 0:
        return 0.0
    return peak(x) / r


def spectral_kurtosis(x: np.ndarray, fs: int, nperseg: int = 256) -> float:
    _, _, zxx = stft(x, fs=fs, nperseg=nperseg)
    mag = np.abs(zxx)
    sk = kurtosis(mag, axis=1, fisher=True)
    return float(np.mean(sk))


def tkeo_energy(x: np.ndarray) -> float:
    tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
    return float(np.mean(tkeo))


def extract_features(window: np.ndarray, fs: int) -> np.ndarray:
    """Compute the 5 vibration features from a raw window.

    Args:
        window: (window_size,) raw vibration window.
        fs: Vibration sampling rate in Hz.

    Returns:
        (5,) float32 array: RMS, Peak, Crest Factor, Spectral Kurtosis, TKEO.
    """
    return np.array([
        rms(window),
        peak(window),
        crest_factor(window),
        spectral_kurtosis(window, fs=fs),
        tkeo_energy(window),
    ], dtype=np.float32)


def preprocess_vibration(window: np.ndarray, fs: int, vib_mean: np.ndarray,
                          vib_std: np.ndarray) -> np.ndarray:
    """Extract and normalize vibration features from a raw window.

    Args:
        window: (window_size,) raw vibration window.
        fs: Vibration sampling rate in Hz.
        vib_mean: (5,) normalization mean, from normalization_stats.json.
        vib_std: (5,) normalization std, from the same file.

    Returns:
        (1, 5) float32 array, ready for the ONNX session's 'vib_features'
        input.
    """
    features = extract_features(window, fs=fs)
    normalized = (features - vib_mean) / vib_std
    return np.expand_dims(normalized, axis=0).astype(np.float32)