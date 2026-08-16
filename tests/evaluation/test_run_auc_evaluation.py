from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from defect_detection.evaluation.run_auc_evaluation import (
    _compute_fold_fault_auc,
    _safe_pr_auc,
    _safe_roc_auc,
    compute_global_oof_metrics,
    initialize_oof_arrays,
    run_oof_cross_validation,
)

CLASS_NAMES = ["outer_race", "inner_race", "ball"]

def test_initialize_oof_arrays_correct_shapes_and_zeroed():
    arrays = initialize_oof_arrays(n_samples=10, num_fault_classes=3, modalities=("image", "both"))

    assert set(arrays.keys()) == {"image", "both"}
    for modality in ("image", "both"):
        assert arrays[modality]["oof_defect_proba"].shape == (10,)
        assert arrays[modality]["oof_fault_proba"].shape == (10, 3)
        assert np.all(arrays[modality]["oof_defect_proba"] == 0)
        assert np.all(arrays[modality]["oof_fault_proba"] == 0)


def test_safe_roc_auc_returns_nan_for_single_class():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    assert np.isnan(_safe_roc_auc(y_true, y_score))


def test_safe_pr_auc_returns_nan_for_no_positives():
    y_true = np.array([0, 0, 0, 0])
    y_score = np.array([0.1, 0.2, 0.3, 0.4])

    assert np.isnan(_safe_pr_auc(y_true, y_score))


def test_safe_auc_returns_real_value_for_well_formed_input():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])

    assert _safe_roc_auc(y_true, y_score) == pytest.approx(1.0)
    assert _safe_pr_auc(y_true, y_score) == pytest.approx(1.0)


def test_compute_fold_fault_auc_nan_for_absent_class():
    """A completely absent class returns NaN; classes that do occur (and have
    negatives available from each other) get a real value."""
    y_true = np.array([0, 0, 1, 1])  
    y_score = np.array([[0.8, 0.1, 0.1], [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])

    result = _compute_fold_fault_auc(y_true, y_score, CLASS_NAMES)

    assert np.isnan(result["ball"]["roc_auc"])
    assert not np.isnan(result["outer_race"]["roc_auc"])
    assert not np.isnan(result["inner_race"]["roc_auc"])


def _fake_manifest_df(n_samples: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "is_defect": [0, 0, 0, 0, 1, 1, 1, 1],
        "fault_class": [None, None, None, None, "outer_race", "outer_race", "inner_race", "ball"],
    })


def _fake_fold_df(test_positions: list[int], all_positions: list[int]) -> pd.DataFrame:
    """A fold dataframe where rows at test_positions are 'test', the rest 'train'."""
    manifest = _fake_manifest_df()
    df = manifest.copy()
    df["split"] = "train"
    df.loc[test_positions, "split"] = "test"
    return df


@contextmanager
def _patch_oof_dependencies(folds: list[pd.DataFrame], predictions_per_fold: list[dict]):
    model = MagicMock()
    vib_mean = np.zeros(5)
    vib_std = np.ones(5)

    mocks = {
        "generate_stratified_kfold_splits": MagicMock(return_value=folds),
        "train_from_dataframes": MagicMock(return_value=(model, vib_mean, vib_std)),
        "MultimodalDefectDataset": MagicMock(),
        "collect_test_predictions": MagicMock(side_effect=predictions_per_fold * 3),  # x3 modalities
    }
    with patch.multiple("defect_detection.evaluation.run_auc_evaluation", **mocks):
        yield mocks


def test_oof_arrays_populated_at_correct_global_positions():
    """Predictions from each fold's test rows must land at exactly those global
    row positions, not shifted or overwritten by other folds."""
    fold_0 = _fake_fold_df(test_positions=[0, 1], all_positions=list(range(8)))
    fold_1 = _fake_fold_df(test_positions=[2, 3], all_positions=list(range(8)))
    folds = [fold_0, fold_1]

    predictions_fold_0 = {
        "is_defect_true": np.array([0, 0]), "is_defect_pred": np.array([0, 0]),
        "defect_proba": np.array([0.11, 0.22]),
        "fault_class_true": np.array([], dtype=int), "fault_class_pred": np.array([], dtype=int),
        "fault_class_proba": np.empty((0, 3)),
    }
    predictions_fold_1 = {
        "is_defect_true": np.array([0, 0]), "is_defect_pred": np.array([0, 0]),
        "defect_proba": np.array([0.33, 0.44]),
        "fault_class_true": np.array([], dtype=int), "fault_class_pred": np.array([], dtype=int),
        "fault_class_proba": np.empty((0, 3)),
    }

    with _patch_oof_dependencies(folds, [predictions_fold_0, predictions_fold_1]):
        result = run_oof_cross_validation(
            _fake_manifest_df(), CLASS_NAMES, window_size=2048, fs=12000,
            k=2, device=torch.device("cpu"), modalities=("image",),
        )

    oof_defect_proba = result["models"]["image"]["oof_defect_proba"]
    assert oof_defect_proba[0] == pytest.approx(0.11)
    assert oof_defect_proba[1] == pytest.approx(0.22)
    assert oof_defect_proba[2] == pytest.approx(0.33)
    assert oof_defect_proba[3] == pytest.approx(0.44)
    assert oof_defect_proba[4] == 0.0


def test_ground_truth_arrays_correctly_populated():
    """is_defect_true should reflect real labels; fault_class_true should be -1
    for normal rows and the correct class index for defective rows."""
    fold_0 = _fake_fold_df(test_positions=[4, 5, 6, 7], all_positions=list(range(8)))
    predictions = {
        "is_defect_true": np.array([1, 1, 1, 1]), "is_defect_pred": np.array([1, 1, 1, 1]),
        "defect_proba": np.array([0.9, 0.9, 0.9, 0.9]),
        "fault_class_true": np.array([0, 0, 1, 2]), "fault_class_pred": np.array([0, 0, 1, 2]),
        "fault_class_proba": np.random.rand(4, 3),
    }

    with _patch_oof_dependencies([fold_0], [predictions]):
        result = run_oof_cross_validation(
            _fake_manifest_df(), CLASS_NAMES, window_size=2048, fs=12000,
            k=1, device=torch.device("cpu"), modalities=("image",),
        )

    assert list(result["is_defect_true"][:4]) == [0, 0, 0, 0]
    assert list(result["fault_class_true"][:4]) == [-1, -1, -1, -1]
    assert list(result["fault_class_true"][4:]) == [0, 0, 1, 2]


def test_every_fold_trained_with_register_model_false():
    fold_0 = _fake_fold_df(test_positions=[0, 1], all_positions=list(range(8)))
    predictions = {
        "is_defect_true": np.array([0, 0]), "is_defect_pred": np.array([0, 0]),
        "defect_proba": np.array([0.1, 0.2]),
        "fault_class_true": np.array([], dtype=int), "fault_class_pred": np.array([], dtype=int),
        "fault_class_proba": np.empty((0, 3)),
    }

    with _patch_oof_dependencies([fold_0], [predictions]) as mocks:
        run_oof_cross_validation(
            _fake_manifest_df(), CLASS_NAMES, window_size=2048, fs=12000,
            k=1, device=torch.device("cpu"), modalities=("image", "vibration"),
        )

    for call in mocks["train_from_dataframes"].call_args_list:
        assert call.kwargs["register_model"] is False


def test_fault_metrics_correctly_mask_out_normal_samples():
    """Fault-type AUC must be computed only on defective rows."""
    is_defect_true = np.array([0, 0, 1, 1, 1])
    fault_class_true = np.array([-1, -1, 0, 0, 1])  

    oof_results = {
        "is_defect_true": is_defect_true,
        "fault_class_true": fault_class_true,
        "models": {
            "image": {
                "oof_defect_proba": np.array([0.1, 0.2, 0.8, 0.9, 0.7]),
                "oof_fault_proba": np.array([
                    [0, 0, 0], [0, 0, 0],
                    [0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.1, 0.8, 0.1],
                ]),
            },
        },
    }

    result = compute_global_oof_metrics(oof_results, CLASS_NAMES)

    assert result["image"]["fault"]["outer_race"]["roc_auc"] == pytest.approx(1.0)


def test_defect_metrics_use_all_samples_unmasked():
    """Head 1 (defect gate) metrics should use every sample, not just defective ones."""
    oof_results = {
        "is_defect_true": np.array([0, 0, 1, 1]),
        "fault_class_true": np.array([-1, -1, 0, 1]),
        "models": {
            "image": {
                "oof_defect_proba": np.array([0.1, 0.2, 0.8, 0.9]),
                "oof_fault_proba": np.zeros((4, 3)),
            },
        },
    }

    result = compute_global_oof_metrics(oof_results, CLASS_NAMES)

    assert result["image"]["defect"]["roc_auc"] == pytest.approx(1.0)