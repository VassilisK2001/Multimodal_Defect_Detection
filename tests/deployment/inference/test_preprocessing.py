"""
Tests for src/defect_detection/deployment/inference/preprocessing.py.
"""


import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

DEPLOYMENT_DIR = Path(__file__).resolve().parents[3] / "deployment"
sys.path.insert(0, str(DEPLOYMENT_DIR))

from inference.preprocessing import ( 
    FEATURE_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    crest_factor,
    peak,
    rms,
    tkeo_energy,
    extract_features,
    preprocess_image,
    preprocess_vibration,
)


def test_preprocess_image_output_shape():
    image = Image.new("RGB", (100, 50), color=(128, 64, 32))

    result = preprocess_image(image)

    assert result.shape == (1, 3, 224, 224)
    assert result.dtype == np.float32


def test_preprocess_image_normalization_on_constant_color():
    """A constant-color image makes the exact expected normalized value
    hand-computable, since every pixel resizes to the same value."""
    r, g, b = 128, 64, 32
    image = Image.new("RGB", (50, 50), color=(r, g, b))

    result = preprocess_image(image)

    scaled = np.array([r, g, b], dtype=np.float32) / 255.0
    expected = (scaled - IMAGENET_MEAN) / IMAGENET_STD

    for channel in range(3):
        assert np.allclose(result[0, channel], expected[channel], atol=1e-5)


def test_preprocess_image_resizes_non_square_input():
    image = Image.new("RGB", (300, 100), color=(10, 10, 10))

    result = preprocess_image(image)

    assert result.shape == (1, 3, 224, 224)


def test_rms_hand_verifiable():
    x = np.array([3.0, 4.0], dtype=np.float32)  

    assert rms(x) == pytest.approx(np.sqrt(12.5), abs=1e-5)


def test_peak_hand_verifiable():
    x = np.array([-5.0, 2.0, 3.0], dtype=np.float32)

    assert peak(x) == pytest.approx(5.0)


def test_crest_factor_hand_verifiable():
    x = np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)  

    assert crest_factor(x) == pytest.approx(1.0)


def test_crest_factor_zero_rms_returns_zero():
    x = np.zeros(10, dtype=np.float32)

    assert crest_factor(x) == 0.0


def test_tkeo_energy_hand_verifiable():
    x = np.array([1.0, 2.0, 1.0], dtype=np.float32)
    # tkeo[0] = x[1]^2 - x[0]*x[2] = 4 - 1 = 3
    expected = 3.0

    assert tkeo_energy(x) == pytest.approx(expected)


def test_extract_features_shape_and_order():
    window = np.random.randn(2048).astype(np.float32)

    result = extract_features(window, fs=12000)

    assert result.shape == (5,)
    assert result.dtype == np.float32
    assert len(FEATURE_NAMES) == 5


def test_preprocess_vibration_shape_and_normalization():
    window = np.random.randn(2048).astype(np.float32)
    vib_mean = np.zeros(5, dtype=np.float32)
    vib_std = np.ones(5, dtype=np.float32)

    result = preprocess_vibration(window, fs=12000, vib_mean=vib_mean, vib_std=vib_std)

    assert result.shape == (1, 5)
    assert result.dtype == np.float32
    
    raw = extract_features(window, fs=12000)
    assert np.allclose(result[0], raw, atol=1e-5)


def test_preprocess_vibration_applies_normalization_correctly():
    window = np.random.randn(2048).astype(np.float32)
    raw = extract_features(window, fs=12000)
    vib_mean = raw.copy()  
    vib_std = np.ones(5, dtype=np.float32)

    result = preprocess_vibration(window, fs=12000, vib_mean=vib_mean, vib_std=vib_std)

    assert np.allclose(result[0], np.zeros(5), atol=1e-5)