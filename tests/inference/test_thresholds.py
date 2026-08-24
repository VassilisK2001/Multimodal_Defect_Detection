import numpy as np
import pytest

from defect_detection.inference.thresholds import bootstrap_threshold_distribution


def test_returns_correct_array_shapes():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    result = bootstrap_threshold_distribution(y_true, y_proba, target_recall=0.9, n_bootstrap=50)

    assert result["thresholds"].shape == (50,)
    assert result["precisions"].shape == (50,)
    assert result["recalls"].shape == (50,)
    assert result["n_bootstrap"] == 50


def test_mean_and_std_match_returned_arrays():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    result = bootstrap_threshold_distribution(y_true, y_proba, target_recall=0.9, n_bootstrap=100)

    assert result["mean_threshold"] == pytest.approx(result["thresholds"].mean())
    assert result["std_threshold"] == pytest.approx(result["thresholds"].std())
    assert result["mean_recall"] == pytest.approx(result["recalls"].mean())
    assert result["std_recall"] == pytest.approx(result["recalls"].std())
    assert result["mean_precision"] == pytest.approx(result["precisions"].mean())
    assert result["std_precision"] == pytest.approx(result["precisions"].std())


def test_deterministic_given_same_seed():
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.6, 0.5, 0.4, 0.3, 0.2])

    result_a = bootstrap_threshold_distribution(y_true, y_proba, n_bootstrap=30, seed=7)
    result_b = bootstrap_threshold_distribution(y_true, y_proba, n_bootstrap=30, seed=7)

    assert np.array_equal(result_a["thresholds"], result_b["thresholds"])


def test_n_target_achieved_within_valid_range():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    result = bootstrap_threshold_distribution(y_true, y_proba, target_recall=0.5, n_bootstrap=100)

    assert 0 <= result["n_target_achieved"] <= 100


def test_single_positive_and_negative_give_identical_resamples():
    """With exactly one positive and one negative, every stratified resample is
    identical to the original two-row dataset."""
    y_true = np.array([1, 0])
    y_proba = np.array([0.8, 0.2])

    result = bootstrap_threshold_distribution(y_true, y_proba, target_recall=1.0, n_bootstrap=20)

    assert np.all(result["thresholds"] == result["thresholds"][0])
    assert result["std_threshold"] == pytest.approx(0.0)


def test_handles_one_class_entirely_absent_without_crashing():
    y_true = np.array([1, 1, 1])
    y_proba = np.array([0.9, 0.8, 0.7])

    result = bootstrap_threshold_distribution(y_true, y_proba, n_bootstrap=10)

    assert result["thresholds"].shape == (10,)
    assert np.all(result["precisions"] == 1.0)