"""
Tests for src/defect_detection/evaluation/metrics.py.
"""


import numpy as np
import pytest

from defect_detection.evaluation.metrics import (
    compute_defect_gate_metrics,
    compute_fault_type_metrics,
)


def test_defect_gate_metrics_hand_verifiable():
    """Precision/recall/F1 should match a manually computed example."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])

    result = compute_defect_gate_metrics(y_true, y_pred)

    assert result["defect"]["precision"] == pytest.approx(2 / 3)
    assert result["defect"]["recall"] == pytest.approx(2 / 3)
    assert result["defect"]["f1"] == pytest.approx(2 / 3)
    assert result["normal"]["precision"] == pytest.approx(2 / 3)
    assert result["normal"]["recall"] == pytest.approx(2 / 3)


def test_defect_gate_metrics_support_for_both_classes():
    """Support should reflect the true count of each class."""
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])

    result = compute_defect_gate_metrics(y_true, y_pred)

    assert result["normal"]["support"] == 4
    assert result["defect"]["support"] == 2


def test_defect_gate_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    result = compute_defect_gate_metrics(y_true, y_pred)

    assert result["normal"]["f1"] == pytest.approx(1.0)
    assert result["defect"]["f1"] == pytest.approx(1.0)


def test_defect_gate_metrics_zero_division_class_never_predicted():
    """Precision for a class that is never predicted should be 0."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0])

    result = compute_defect_gate_metrics(y_true, y_pred)

    assert result["defect"]["precision"] == 0.0
    assert not np.isnan(result["defect"]["precision"])


def test_defect_gate_metrics_zero_division_class_never_in_true():
    """Recall for a class absent from y_true should be 0."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 1, 0])

    result = compute_defect_gate_metrics(y_true, y_pred)

    assert result["defect"]["recall"] == 0.0
    assert not np.isnan(result["defect"]["recall"])


def test_fault_type_metrics_hand_verifiable():
    """Per-class precision/recall/F1 should match a manually computed example."""
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 0, 1, 1, 2])
    y_pred = np.array([0, 0, 1, 2, 2])

    result = compute_fault_type_metrics(y_true, y_pred, class_names)

    assert result["per_class"]["outer_race"]["precision"] == pytest.approx(1.0)
    assert result["per_class"]["outer_race"]["recall"] == pytest.approx(1.0)
    assert result["per_class"]["inner_race"]["recall"] == pytest.approx(0.5)


def test_fault_type_metrics_support_per_class():
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 0, 0, 1, 2])

    result = compute_fault_type_metrics(y_true, y_pred, class_names)

    assert result["per_class"]["outer_race"]["support"] == 3
    assert result["per_class"]["inner_race"]["support"] == 1
    assert result["per_class"]["ball"]["support"] == 1


def test_fault_type_metrics_macro_f1_is_mean_of_per_class_f1():
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 0])

    result = compute_fault_type_metrics(y_true, y_pred, class_names)

    per_class_f1 = [result["per_class"][name]["f1"] for name in class_names]
    assert result["macro_f1"] == pytest.approx(np.mean(per_class_f1))


def test_fault_type_metrics_zero_division_class_never_predicted():
    """A class never predicted should have precision 0."""
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 0, 0])

    result = compute_fault_type_metrics(y_true, y_pred, class_names)

    assert result["per_class"]["inner_race"]["precision"] == 0.0
    assert result["per_class"]["ball"]["precision"] == 0.0


def test_fault_type_metrics_perfect_predictions():
    class_names = ["outer_race", "inner_race", "ball"]
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    result = compute_fault_type_metrics(y_true, y_pred, class_names)

    assert result["macro_f1"] == pytest.approx(1.0)


def test_both_functions_return_consistent_per_class_key_shape():
    """Both functions' per-class entries should expose the same keys."""
    defect_result = compute_defect_gate_metrics(np.array([0, 1]), np.array([0, 1]))
    fault_result = compute_fault_type_metrics(
        np.array([0, 1]), np.array([0, 1]), ["outer_race", "inner_race"],
    )

    defect_keys = set(defect_result["normal"].keys())
    fault_keys = set(fault_result["per_class"]["outer_race"].keys())

    assert defect_keys == fault_keys == {"precision", "recall", "f1", "support"}