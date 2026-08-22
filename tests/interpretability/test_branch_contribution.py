import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from defect_detection.interpretability.branch_contribution import (
    _get_forward_and_target,
    _make_defect_forward,
    _make_fault_forward,
    build_feature_masks,
    check_shapley_additivity_sample,
    compute_branch_contributions,
    prepare_background_samples,
)
from defect_detection.models.fusion_model import MultimodalDefectClassifier


@pytest.fixture
def model() -> MultimodalDefectClassifier:
    return MultimodalDefectClassifier(modality="both")


def test_build_feature_masks_correct_shapes_and_group_ids():
    image_mask, vib_mask = build_feature_masks((3, 224, 224), (5,))

    assert image_mask.shape == (3, 224, 224)
    assert vib_mask.shape == (5,)
    assert torch.all(image_mask == 0)
    assert torch.all(vib_mask == 1)
    assert not torch.any(image_mask == vib_mask.max())


def test_prepare_background_samples_returns_k_pairs_with_correct_shapes(monkeypatch, tmp_path):
    for i in range(5):
        Image.new("RGB", (64, 64)).save(tmp_path / f"img_{i}.png")
    df = pd.DataFrame({
        "image_path": [f"img_{i}.png" for i in range(5)],
        "vibration_file": ["f.mat"] * 5,
        "vibration_window_idx": list(range(5)),
    })

    monkeypatch.setattr(
        "defect_detection.interpretability.branch_contribution.extract_raw_vib_features_from_df",
        lambda df, window_size, fs: np.random.randn(len(df), 5).astype(np.float32),
    )

    result = prepare_background_samples(
        df, tmp_path, window_size=1024, fs=12000,
        vib_mean=np.zeros(5), vib_std=np.ones(5), k=3,
    )

    assert len(result) == 3
    for image, vib in result:
        assert image.shape == (3, 224, 224)
        assert vib.shape == (5,)


def test_get_forward_and_target_raises_for_unknown_head(model):
    with pytest.raises(ValueError):
        _get_forward_and_target(model, "not_a_real_head", None)


def test_defect_forward_returns_valid_probability(model):
    forward = _make_defect_forward(model)
    image = torch.randn(2, 3, 224, 224)
    vib = torch.randn(2, 5)

    result = forward(image, vib)

    assert result.shape == (2,)
    assert torch.all(result >= 0) and torch.all(result <= 1)


def test_fault_forward_returns_valid_probability_distribution(model):
    forward = _make_fault_forward(model)
    image = torch.randn(2, 3, 224, 224)
    vib = torch.randn(2, 5)

    result = forward(image, vib)

    assert result.shape == (2, 3)
    assert torch.allclose(result.sum(dim=1), torch.ones(2), atol=1e-5)


def test_compute_branch_contributions_shapes_and_k(model):
    images = torch.randn(3, 3, 224, 224)
    vib = torch.randn(3, 5)
    backgrounds = [(torch.randn(3, 224, 224), torch.randn(5)) for _ in range(4)]

    phi_image, phi_vib, se_info = compute_branch_contributions(model, "defect", images, vib, backgrounds)

    assert phi_image.shape == (3,)
    assert phi_vib.shape == (3,)
    assert se_info["se_image"].shape == (3,)
    assert se_info["se_vib"].shape == (3,)
    assert se_info["k"] == 4


def test_compute_branch_contributions_additivity(model):
    """The Shapley property: phi_image + phi_vib must equal
    v(joint) - v(baseline), averaged across background samples."""
    images = torch.randn(2, 3, 224, 224)
    vib = torch.randn(2, 5)
    backgrounds = [(torch.randn(3, 224, 224), torch.randn(5)) for _ in range(3)]

    phi_image, phi_vib, _ = compute_branch_contributions(model, "defect", images, vib, backgrounds)

    forward = _make_defect_forward(model)
    with torch.no_grad():
        v_joint = forward(images, vib).numpy()
        v_baselines = np.array([
            forward(bg_image.unsqueeze(0), bg_vib.unsqueeze(0)).item() for bg_image, bg_vib in backgrounds
        ])
    expected_sum = v_joint - v_baselines.mean()

    assert np.allclose(phi_image + phi_vib, expected_sum, atol=1e-4)


def test_compute_branch_contributions_fault_head_target_class_differs(model):
    images = torch.randn(2, 3, 224, 224)
    vib = torch.randn(2, 5)
    backgrounds = [(torch.randn(3, 224, 224), torch.randn(5)) for _ in range(3)]

    phi_image_0, phi_vib_0, _ = compute_branch_contributions(
        model, "fault", images, vib, backgrounds, target_class=0)
    phi_image_1, phi_vib_1, _ = compute_branch_contributions(
        model, "fault", images, vib, backgrounds, target_class=1)

    assert not np.allclose(phi_image_0, phi_image_1)
    assert not np.allclose(phi_vib_0, phi_vib_1)


def test_additivity_sample_residual_near_zero_for_real_model(model):
    images = torch.randn(3, 3, 224, 224)
    vib = torch.randn(3, 5)
    backgrounds = [(torch.randn(3, 224, 224), torch.randn(5)) for _ in range(3)]

    result = check_shapley_additivity_sample(model, "defect", images, vib, backgrounds, tol=1e-3)

    assert result["max_residual"] < 1e-3
    assert result["within_tolerance"] is True


def test_additivity_sample_respects_tolerance(model):
    images = torch.randn(2, 3, 224, 224)
    vib = torch.randn(2, 5)
    backgrounds = [(torch.randn(3, 224, 224), torch.randn(5)) for _ in range(2)]

    result = check_shapley_additivity_sample(model, "defect", images, vib, backgrounds, tol=-1.0)

    assert result["within_tolerance"] is False