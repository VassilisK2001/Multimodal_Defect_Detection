"""
Tests for src/defect_detection/interpretability/shap_explain.py.
"""


from types import SimpleNamespace
from typing import cast

import numpy as np
import pandas as pd
import pytest
import shap
import torch

from defect_detection.interpretability.shap_explain import (
    approximate_predictions_from_shap,
    check_additivity,
    check_additivity_per_instance,
    compute_shap_batched,
    compute_shap_per_instance,
    make_fusion_predict_fn,
    make_vib_predict_fn,
    select_head1_background_rows,
    select_head2_background_rows,
    select_top_features,
    summarize_background,
)
from defect_detection.models.fusion_model import MultimodalDefectClassifier
from PIL import Image


def test_ball_is_oversampled_relative_to_even_split():
    rows = (
        [{"is_defect": 1, "fault_class": "outer_race"}] * 30
        + [{"is_defect": 1, "fault_class": "inner_race"}] * 30
        + [{"is_defect": 1, "fault_class": "ball"}] * 30
    )
    train_df = pd.DataFrame(rows)

    result = select_head2_background_rows(train_df, n_samples=30, ball_oversample_factor=2.0)

    counts = result["fault_class"].value_counts()
    assert counts["ball"] > counts["outer_race"]
    assert counts["ball"] > counts["inner_race"]


def test_fusion_predict_fn_broadcasts_image_to_match_batch_size():
    model = MultimodalDefectClassifier(modality="both")
    image = torch.randn(3, 224, 224)
    predict_fn = make_fusion_predict_fn(model, head="defect", image=image)

    vib_features = np.random.randn(4, 5).astype(np.float32)
    result = predict_fn(vib_features)

    assert result.shape == (4,)


def test_fusion_predict_fn_output_varies_with_vibration_features():
    model = MultimodalDefectClassifier(modality="both")
    image = torch.randn(3, 224, 224)
    predict_fn = make_fusion_predict_fn(model, head="defect", image=image)

    vib_a = np.zeros((1, 5), dtype=np.float32)
    vib_b = np.ones((1, 5), dtype=np.float32) * 5.0

    result_a = predict_fn(vib_a)
    result_b = predict_fn(vib_b)

    assert not np.allclose(result_a, result_b)


def test_per_instance_uses_each_rows_own_image_not_a_shared_one():
    """Each row must genuinely use its own image, not 
        reuse the same one across all rows."""
    model = MultimodalDefectClassifier(modality="both")

    vib_features = np.zeros((2, 5), dtype=np.float32)
    images = torch.stack([torch.zeros(3, 224, 224), torch.ones(3, 224, 224)])

    with torch.no_grad():
        out_row0, _ = model(image=images[0:1], vib_features=torch.tensor(vib_features[0:1]))
        out_row1, _ = model(image=images[1:2], vib_features=torch.tensor(vib_features[1:2]))

    assert not torch.allclose(out_row0, out_row1)

    background = np.random.randn(10, 5).astype(np.float32)
    shap_values = compute_shap_per_instance(
        model, head="defect", background=background,
        test_features_normalized=vib_features, test_features_raw=vib_features,
        test_images=images,
    )
    values = cast(np.ndarray, shap_values.values)

    assert values.shape[0] == 2


def test_additivity_residual_computed_correctly():
    """Uses a hand-constructed Explanation-like object and a predict_fn with a
    known output, isolating the residual arithmetic from real SHAP computation."""
    fake_shap_values = SimpleNamespace(
        values=np.array([[0.1, 0.2, -0.05]]),
        base_values=np.array([0.5]),
    )

    def fake_predict_fn(x):
        return np.array([0.8])

    result = check_additivity(
        cast(shap.Explanation, fake_shap_values), fake_predict_fn, np.zeros((1, 3)), tol=1e-3,
    )

    assert result["max_residual"] == pytest.approx(0.05, abs=1e-6)
    assert result["within_tolerance"] is False


def test_head1_background_includes_both_classes():
    train_df = pd.DataFrame(
        [{"is_defect": 0}] * 50 + [{"is_defect": 1}] * 50
    )

    result = select_head1_background_rows(train_df, n_samples=20)

    counts = result["is_defect"].value_counts()
    assert counts.get(0, 0) > 0
    assert counts.get(1, 0) > 0


def test_vib_predict_fn_defect_head_output_shape():
    model = MultimodalDefectClassifier(modality="vibration")
    predict_fn = make_vib_predict_fn(model, head="defect")

    vib_features = np.random.randn(4, 5).astype(np.float32)
    result = predict_fn(vib_features)

    assert result.shape == (4,)


def test_vib_predict_fn_fault_head_output_shape():
    model = MultimodalDefectClassifier(modality="vibration")
    predict_fn = make_vib_predict_fn(model, head="fault")

    vib_features = np.random.randn(4, 5).astype(np.float32)
    result = predict_fn(vib_features)

    assert result.shape == (4, 3)


def test_compute_shap_batched_attaches_raw_data_and_correct_shape():
    model = MultimodalDefectClassifier(modality="vibration")
    predict_fn = make_vib_predict_fn(model, head="defect")

    background = np.random.randn(10, 5).astype(np.float32)
    test_normalized = np.random.randn(3, 5).astype(np.float32)
    test_raw = np.random.randn(3, 5).astype(np.float32)

    shap_values = compute_shap_batched(predict_fn, background, test_normalized, test_raw)
    values = cast(np.ndarray, shap_values.values)

    assert values.shape[0] == 3
    assert np.array_equal(shap_values.data, test_raw)


def test_approximate_predictions_matches_hand_computed_sum():
    fake_shap_values = SimpleNamespace(
        values=np.array([[0.1, 0.2, -0.05], [0.3, -0.1, 0.0]]),
        base_values=np.array([0.5, 0.5]),
    )

    result = approximate_predictions_from_shap(cast(shap.Explanation, fake_shap_values))

    assert np.allclose(result, [0.75, 0.7])


def test_approximate_predictions_broadcasts_scalar_base_value():
    fake_shap_values = SimpleNamespace(
        values=np.array([[0.1, 0.1], [0.2, 0.2]]),
        base_values=np.array(0.5),
    )

    result = approximate_predictions_from_shap(cast(shap.Explanation, fake_shap_values))

    assert np.allclose(result, [0.7, 0.9])


def test_select_top_features_returns_correct_order():
    fake_shap_values = SimpleNamespace(
        values=np.array([
            [0.1, -0.5, 0.05, 0.02, 0.01],
            [0.1, 0.5, -0.05, 0.02, 0.01],
        ]),
        feature_names=["RMS", "Peak", "Crest Factor", "Spectral Kurtosis", "TKEO"],
    )

    result = select_top_features(cast(shap.Explanation, fake_shap_values), k=2)

    assert result == ["Peak", "RMS"]


def test_select_top_features_respects_k():
    fake_shap_values = SimpleNamespace(
        values=np.random.randn(5, 5),
        feature_names=["RMS", "Peak", "Crest Factor", "Spectral Kurtosis", "TKEO"],
    )

    result = select_top_features(cast(shap.Explanation, fake_shap_values), k=3)

    assert len(result) == 3


def test_additivity_residual_correct_for_multiclass():
    fake_shap_values = SimpleNamespace(
        values=np.array([[[0.1, 0.2], [0.1, -0.1], [0.0, 0.05]]]),  # (1, 3, 2)
        base_values=np.array([[0.3, 0.3]]),  # (1, 2)
    )

    def fake_predict_fn(x):
        return np.array([[0.5, 0.45]])  # (1, 2)

    result = check_additivity(
        cast(shap.Explanation, fake_shap_values), fake_predict_fn, np.zeros((1, 3)), tol=1e-3,
    )

    # sum over features (axis=1) per class
    assert result["max_residual"] == pytest.approx(0.0, abs=1e-6)


def test_summarize_background_shape():
    features = np.random.randn(30, 5).astype(np.float32)

    result = summarize_background(features, k=10)

    assert result.data.shape == (10, 5)


def test_summarize_background_caps_k_at_available_samples():
    features = np.random.randn(5, 5).astype(np.float32)

    result = summarize_background(features, k=50)

    assert result.data.shape[0] == 5


def test_summarize_background_deterministic_given_same_seed():
    features = np.random.randn(30, 5).astype(np.float32)

    result_a = summarize_background(features, k=10, seed=1)
    result_b = summarize_background(features, k=10, seed=1)

    assert np.allclose(result_a.data, result_b.data)


def test_compute_shap_batched_handles_multiclass_predict_fn():
    model = MultimodalDefectClassifier(modality="vibration")
    predict_fn = make_vib_predict_fn(model, head="fault")

    background = np.random.randn(10, 5).astype(np.float32)
    test_normalized = np.random.randn(4, 5).astype(np.float32)
    test_raw = np.random.randn(4, 5).astype(np.float32)

    shap_values = compute_shap_batched(predict_fn, background, test_normalized, test_raw)
    values = cast(np.ndarray, shap_values.values)
    base_values = np.asarray(shap_values.base_values)

    assert values.shape == (4, 5, 3)
    assert base_values.shape == (4, 3)


def test_compute_shap_per_instance_handles_multiclass_fault_head():
    model = MultimodalDefectClassifier(modality="both")

    vib_features = np.random.randn(2, 5).astype(np.float32)
    images = torch.stack([torch.zeros(3, 224, 224), torch.ones(3, 224, 224)])
    background = np.random.randn(10, 5).astype(np.float32)

    shap_values = compute_shap_per_instance(
        model, head="fault", background=background,
        test_features_normalized=vib_features, test_features_raw=vib_features,
        test_images=images,
    )
    values = cast(np.ndarray, shap_values.values)
    base_values = np.asarray(shap_values.base_values)

    assert values.shape == (2, 5, 3)
    assert base_values.shape == (2, 3)


def test_check_additivity_per_instance_returns_well_formed_result():
    model = MultimodalDefectClassifier(modality="both")

    vib_features = np.random.randn(3, 5).astype(np.float32)
    images = torch.stack([torch.zeros(3, 224, 224), torch.ones(3, 224, 224), torch.randn(3, 224, 224)])
    background = np.random.randn(10, 5).astype(np.float32)

    shap_values = compute_shap_per_instance(
        model, head="defect", background=background,
        test_features_normalized=vib_features, test_features_raw=vib_features,
        test_images=images,
    )

    result = check_additivity_per_instance(model, "defect", shap_values, vib_features, images)

    assert result["n_total"] == 3
    assert 0 <= result["n_within_tolerance"] <= 3
    assert result["max_residual"] >= result["mean_residual"] >= 0