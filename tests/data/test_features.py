import numpy as np
import pytest

from defect_detection.data.features import (
    rms, peak, crest_factor, spectral_kurtosis, tkeo_energy, extract_features
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