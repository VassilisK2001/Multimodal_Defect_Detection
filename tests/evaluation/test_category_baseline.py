
import numpy as np
import pandas as pd
import pytest

from tests.factories import CLASS_NAMES
from defect_detection.evaluation.category_baseline import (
    compare_predictions_by_category,
    predict_category_only_baseline,
    train_category_only_baseline,
)


def _manifest_row(category: str, is_defect: int, fault_class: str | None) -> dict:
    return {"category": category, "is_defect": is_defect, "fault_class": fault_class}


def test_only_defective_rows_used():
    """Normal rows should be excluded from training and evaluation."""
    train_df = pd.DataFrame([
        _manifest_row("bottle", 0, None),
        _manifest_row("bottle", 0, None),
        _manifest_row("bottle", 1, "outer_race"),
        _manifest_row("screw", 1, "ball"),
    ])
    test_df = pd.DataFrame([
        _manifest_row("bottle", 0, None),
        _manifest_row("screw", 1, "ball"),
    ])

    # Should not raise despite normal rows having fault_class=None.
    result = train_category_only_baseline(train_df, test_df, CLASS_NAMES)
    assert result["per_class"]["ball"]["support"] == 1


def test_deterministic_category_mapping_achieves_high_accuracy():
    """A category that deterministically maps to one fault class should be
    classified with near-perfect macro-F1."""
    rows = []
    category_to_class = {"bottle": "outer_race", "screw": "ball", "capsule": "inner_race"}
    for category, fault_class in category_to_class.items():
        for _ in range(10):
            rows.append(_manifest_row(category, 1, fault_class))
    train_df = pd.DataFrame(rows)
    test_df = train_df.copy()

    result = train_category_only_baseline(train_df, test_df, CLASS_NAMES)

    assert result["macro_f1"] > 0.95


def test_handles_unseen_category_in_test_set():
    """A category present in the test set but not in training should not raise."""
    train_df = pd.DataFrame([
        _manifest_row("bottle", 1, "outer_race"),
        _manifest_row("screw", 1, "ball"),
    ])
    test_df = pd.DataFrame([
        _manifest_row("capsule", 1, "inner_race"),
    ])

    result = train_category_only_baseline(train_df, test_df, CLASS_NAMES)
    assert "macro_f1" in result


def test_predictions_respect_class_names_order():
    """Predictions should be correctly labeled under a non-default class_names
    order, not shifted to the wrong class."""
    rows = (
        [_manifest_row("bottle", 1, "outer_race") for _ in range(10)]
        + [_manifest_row("screw", 1, "ball") for _ in range(10)]
    )
    train_df = pd.DataFrame(rows)
    test_df = train_df.copy()

    class_names = ["ball", "outer_race", "inner_race"]
    result = train_category_only_baseline(train_df, test_df, class_names)

    assert result["per_class"]["outer_race"]["support"] == 10
    assert result["per_class"]["ball"]["support"] == 10
    assert result["per_class"]["inner_race"]["support"] == 0
    assert result["per_class"]["outer_race"]["recall"] > 0.9
    assert result["per_class"]["ball"]["recall"] > 0.9


def test_predict_returns_arrays_aligned_with_defective_rows():
    """Returned y_test/y_pred must correspond, in order, to test_df's defective
    rows only."""
    train_df = pd.DataFrame([
        _manifest_row("bottle", 1, "outer_race"),
        _manifest_row("screw", 1, "ball"),
    ])
    test_df = pd.DataFrame([
        _manifest_row("bottle", 0, None),
        _manifest_row("bottle", 1, "outer_race"),
        _manifest_row("screw", 0, None),
        _manifest_row("screw", 1, "ball"),
    ])

    y_test, y_pred = predict_category_only_baseline(train_df, test_df, CLASS_NAMES)

    assert len(y_test) == 2  
    assert len(y_pred) == 2


def _category_test_df() -> pd.DataFrame:
    return pd.DataFrame([
        _manifest_row("bottle", 1, "x"), _manifest_row("bottle", 1, "x"), _manifest_row("bottle", 1, "x"),
        _manifest_row("screw", 1, "x"), _manifest_row("screw", 1, "x"),
    ])


def test_per_category_accuracy_is_correct():
    test_df = _category_test_df()
    y_true = np.array([0, 0, 0, 2, 2])
    category_pred = np.array([0, 1, 1, 2, 2])  
    both_pred = np.array([0, 1, 0, 2, 1])       

    result = compare_predictions_by_category(test_df, y_true, both_pred, category_pred)

    assert result.loc["bottle", "category_accuracy"] == pytest.approx(1 / 3)
    assert result.loc["bottle", "both_accuracy"] == pytest.approx(2 / 3)
    assert result.loc["screw", "category_accuracy"] == pytest.approx(1.0)
    assert result.loc["screw", "both_accuracy"] == pytest.approx(0.5)


def test_agreement_only_computed_on_category_errors():
    """agreement_on_category_errors must be computed only over rows where the
    category-only baseline is wrong, not over all rows."""
    test_df = _category_test_df()
    y_true = np.array([0, 0, 0, 2, 2])
    category_pred = np.array([0, 1, 1, 2, 2]) 
    both_pred = np.array([0, 1, 0, 2, 2])

    result = compare_predictions_by_category(test_df, y_true, both_pred, category_pred)

    assert result.loc["bottle", "agreement_on_category_errors"] == pytest.approx(0.5)


def test_agreement_is_nan_when_category_never_wrong():
    test_df = _category_test_df()
    y_true = np.array([0, 0, 0, 2, 2])
    category_pred = np.array([0, 1, 1, 2, 2]) 
    both_pred = np.array([0, 1, 0, 2, 1])

    result = compare_predictions_by_category(test_df, y_true, both_pred, category_pred)

    assert np.isnan(result.loc["screw", "agreement_on_category_errors"])