
import numpy as np
import pytest
import torch
from PIL import Image

from defect_detection.data.augmentations import (
    build_image_transform, jitter, scale,
)


# jitter

def test_jitter_output_shape_matches_input():
    """Output shape must match input shape."""
    x = np.random.randn(2048).astype(np.float32)
    result = jitter(x)
    assert result.shape == x.shape


def test_jitter_changes_the_signal():
    """On a realistic (non-constant) signal, jitter should actually add noise."""
    np.random.seed(0)
    x = np.random.randn(2048).astype(np.float32)
    result = jitter(x, sigma=0.05)
    assert not np.allclose(result, x)


def test_jitter_constant_signal_is_a_no_op():
    """Noise is scaled by std(window); a constant signal has std=0, so no noise is added
    regardless of sigma-expected behavior"""
    x = np.ones(2048, dtype=np.float32)
    result = jitter(x, sigma=0.5)
    assert np.allclose(result, x)


def test_jitter_zero_sigma_is_a_no_op():
    """sigma=0 should leave the signal unchanged."""
    x = np.random.randn(2048).astype(np.float32)
    result = jitter(x, sigma=0.0)
    assert np.allclose(result, x)


def test_jitter_scales_with_sigma():
    """A larger sigma should produce larger average deviation from the original signal."""
    np.random.seed(1)
    x = np.random.randn(4096).astype(np.float32)

    small_sigma_dev = np.mean(np.abs(jitter(x, sigma=0.01) - x))
    large_sigma_dev = np.mean(np.abs(jitter(x, sigma=0.5) - x))

    assert large_sigma_dev > small_sigma_dev


# scale 

def test_scale_output_shape_matches_input():
    """Output shape must match input shape."""
    x = np.random.randn(2048).astype(np.float32)
    result = scale(x)
    assert result.shape == x.shape


def test_scale_preserves_shape_of_signal():
    """scale is a single multiplicative factor applied to the whole window, so the scaled
    signal must be perfectly correlated with the original."""
    np.random.seed(2)
    x = np.random.randn(2048).astype(np.float32)
    result = scale(x, sigma=0.1)

    correlation = np.corrcoef(x, result)[0, 1]
    assert correlation == pytest.approx(1.0, abs=1e-6) or correlation == pytest.approx(-1.0, abs=1e-6)


def test_scale_zero_signal_stays_zero():
    """Scaling an all-zero signal must still produce all zeros."""
    x = np.zeros(2048, dtype=np.float32)
    result = scale(x, sigma=0.1)
    assert np.allclose(result, 0.0)


def test_scale_zero_sigma_is_near_identity():
    """sigma=0 means the random factor is always 1.0, so the signal should be unchanged."""
    np.random.seed(3)
    x = np.random.randn(2048).astype(np.float32)
    result = scale(x, sigma=0.0)
    assert np.allclose(result, x)


# build_image_transform

@pytest.fixture
def dummy_image() -> Image.Image:
    """Random RGB image standing in for a real MVTec image; only shape/type matter here."""
    arr = (np.random.rand(300, 300, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def test_train_transform_includes_augmentation_steps():
    """Training pipeline must include RandomRotation and ColorJitter."""
    train_transform = build_image_transform(training=True)
    transform_types = [type(t).__name__ for t in train_transform.transforms]

    assert "RandomRotation" in transform_types
    assert "ColorJitter" in transform_types


def test_eval_transform_excludes_augmentation_steps():
    """Eval/inference pipeline must not include RandomRotation or ColorJitter. Predictions
    should be based on deterministic preprocessing of the actual input."""
    eval_transform = build_image_transform(training=False)
    transform_types = [type(t).__name__ for t in eval_transform.transforms]

    assert "RandomRotation" not in transform_types
    assert "ColorJitter" not in transform_types


def test_transforms_produce_correct_output_shape(dummy_image):
    """Both pipelines must resize to 224x224 (expected by the ResNet18 encoder) and
    return a 3-channel tensor, regardless of the original image size."""
    for training in [True, False]:
        transform = build_image_transform(training=training)
        result = transform(dummy_image)
        assert result.shape == (3, 224, 224)
        assert isinstance(result, torch.Tensor)


def test_eval_transform_is_deterministic(dummy_image):
    """No random steps in the eval pipeline, so the same image must always produce the
    same output. This is required for consistent inference."""
    eval_transform = build_image_transform(training=False)
    result1 = eval_transform(dummy_image)
    result2 = eval_transform(dummy_image)
    assert torch.allclose(result1, result2)


def test_train_transform_is_stochastic(dummy_image):
    """RandomRotation/ColorJitter sample new parameters each call, so repeated calls on
    the same image should generally produce different output."""
    train_transform = build_image_transform(training=True)
    result1 = train_transform(dummy_image)
    result2 = train_transform(dummy_image)
    assert not torch.allclose(result1, result2)