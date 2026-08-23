import numpy as np
import pytest

from defect_detection.inference.thresholds import (
    check_threshold_transfers_to_final_model,
    select_recall_constrained_threshold,
)

def test_hand_verifiable_threshold_selection():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55])

    result = select_recall_constrained_threshold(y_true, y_proba, target_recall=1.0)

    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)
    assert result["target_recall_achieved"] is True


def test_selects_highest_threshold_among_those_meeting_recall_floor():
    """Among thresholds meeting the recall floor, the one giving the best
    precision must be chosen, not merely any threshold that satisfies it."""
    y_true = np.array([1, 1, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.5, 0.85, 0.6, 0.3, 0.2])

    result = select_recall_constrained_threshold(y_true, y_proba, target_recall=0.5)

    assert result["recall"] >= 0.5
    assert result["target_recall_achieved"] is True
    assert result["precision"] == pytest.approx(1.0)


def test_falls_back_when_target_recall_unachievable():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.6, 0.5, 0.4, 0.3])

    result = select_recall_constrained_threshold(y_true, y_proba, target_recall=1.5)

    assert result["target_recall_achieved"] is False
    assert result["recall"] == pytest.approx(1.0)


def test_lower_target_recall_gives_higher_or_equal_precision():
    """Relaxing the recall requirement should never produce worse precision."""
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.6, 0.3, 0.8, 0.7, 0.5, 0.2, 0.1])

    strict = select_recall_constrained_threshold(y_true, y_proba, target_recall=0.95)
    relaxed = select_recall_constrained_threshold(y_true, y_proba, target_recall=0.3)

    assert relaxed["precision"] >= strict["precision"]


def test_hand_verifiable_recall_precision():
    y_true = np.array([1, 1, 0, 1, 0])
    y_proba = np.array([0.9, 0.6, 0.4, 0.55, 0.3])

    result = check_threshold_transfers_to_final_model(0.5, y_true, y_proba, target_recall=0.9)

    assert result["recall"] == pytest.approx(1.0)
    assert result["precision"] == pytest.approx(1.0)


def test_diverges_flagged_when_recall_drops_below_tolerance():
    y_true = np.array([1, 1, 1, 1, 0])
    y_proba = np.array([0.9, 0.2, 0.1, 0.05, 0.01])

    result = check_threshold_transfers_to_final_model(
        0.5, y_true, y_proba, target_recall=0.95, tolerance=0.05,
    )

    assert result["recall"] == pytest.approx(0.25)
    assert result["diverges"] is True


def test_does_not_diverge_when_recall_is_close_to_target():
    y_true = np.array([1, 1, 1, 1, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.4, 0.1])

    result = check_threshold_transfers_to_final_model(
        0.5, y_true, y_proba, target_recall=0.7, tolerance=0.1,
    )

    assert result["recall"] == pytest.approx(0.75)
    assert result["diverges"] is False


def test_handles_no_predicted_positives_without_crashing():
    y_true = np.array([1, 1, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.05, 0.15])

    result = check_threshold_transfers_to_final_model(0.9, y_true, y_proba, target_recall=0.5)

    assert result["recall"] == pytest.approx(0.0)
    assert np.isnan(result["precision"])
    assert result["diverges"] is True


def test_handles_no_actual_positives_without_crashing():
    y_true = np.array([0, 0, 0, 0])
    y_proba = np.array([0.9, 0.1, 0.2, 0.8])

    result = check_threshold_transfers_to_final_model(0.5, y_true, y_proba, target_recall=0.5)

    assert np.isnan(result["recall"])