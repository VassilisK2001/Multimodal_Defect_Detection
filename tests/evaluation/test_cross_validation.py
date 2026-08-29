
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from defect_detection.evaluation.cross_validation import aggregate_cv_results, run_kfold_cv


def _fake_fold_df(fold_idx: int) -> pd.DataFrame:
    return pd.DataFrame({
        "split": ["train", "train", "val", "test"],
        "fold_idx": [fold_idx] * 4,
    })


@contextmanager
def _patch_run_kfold_cv_dependencies(k: int = 3):
    folds = [_fake_fold_df(i) for i in range(k)]
    model = MagicMock()
    vib_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    vib_std = np.array([0.5] * 5)
    predictions = {
        "is_defect_true": np.array([0, 1]), "is_defect_pred": np.array([0, 1]),
        "defect_proba": np.array([0.1, 0.9]),
        "fault_class_true": np.array([1]), "fault_class_pred": np.array([1]),
    }

    mocks = {
        "generate_stratified_kfold_splits": MagicMock(return_value=folds),
        "train_from_dataframes": MagicMock(return_value=(model, vib_mean, vib_std)),
        "MultimodalDefectDataset": MagicMock(),
        "collect_test_predictions": MagicMock(return_value=predictions),
        "compute_defect_gate_metrics": MagicMock(return_value={"defect": "metrics"}),
        "compute_fault_type_metrics": MagicMock(return_value={"fault": "metrics"}),
    }

    with patch.multiple("defect_detection.evaluation.cross_validation", **mocks):
        yield mocks


def _call_run_kfold_cv(k: int = 3):
    return run_kfold_cv(
        modality="both", manifest_df=pd.DataFrame(), window_size=2048, fs=12000,
        class_names=["outer_race", "inner_race", "ball"], k=k, device=torch.device("cpu"),
    )


def test_returns_one_result_per_fold():
    with _patch_run_kfold_cv_dependencies(k=3):
        results = _call_run_kfold_cv(k=3)

    assert len(results) == 3
    assert all(set(r.keys()) == {"defect_metrics", "fault_metrics"} for r in results)


def test_every_fold_trained_with_register_model_false():
    with _patch_run_kfold_cv_dependencies(k=3) as mocks:
        _call_run_kfold_cv(k=3)

    for call in mocks["train_from_dataframes"].call_args_list:
        assert call.kwargs["register_model"] is False


def test_each_fold_uses_distinct_seed():
    with _patch_run_kfold_cv_dependencies(k=3) as mocks:
        _call_run_kfold_cv(k=3)

    seeds_used = [call.kwargs["seed"] for call in mocks["train_from_dataframes"].call_args_list]
    assert len(set(seeds_used)) == 3


def test_train_val_test_correctly_extracted_from_fold_split_labels():
    with _patch_run_kfold_cv_dependencies(k=3) as mocks:
        _call_run_kfold_cv(k=3)

    for call in mocks["train_from_dataframes"].call_args_list:
        train_df_arg, val_df_arg = call.args[0], call.args[1]
        assert (train_df_arg["split"] == "train").all()
        assert (val_df_arg["split"] == "val").all()


def test_vib_stats_reach_test_dataset():
    with _patch_run_kfold_cv_dependencies(k=3) as mocks:
        _call_run_kfold_cv(k=3)

    for call in mocks["MultimodalDefectDataset"].call_args_list:
        assert np.array_equal(call.kwargs["vib_mean"], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert np.array_equal(call.kwargs["vib_std"], np.array([0.5] * 5))


def test_hand_verifiable_mean_and_std():
    fold_results = [
        {"defect_metrics": {"f1": 0.6}},
        {"defect_metrics": {"f1": 0.8}},
        {"defect_metrics": {"f1": 1.0}},
    ]

    result = aggregate_cv_results(fold_results)

    assert result["defect_metrics"]["f1"]["mean"] == pytest.approx(0.8)
    assert result["defect_metrics"]["f1"]["std"] == pytest.approx(np.std([0.6, 0.8, 1.0]))


def test_preserves_nested_structure():
    fold_results = [
        {"fault_metrics": {"per_class": {"outer_race": {"f1": 0.7}}, "macro_f1": 0.75}},
        {"fault_metrics": {"per_class": {"outer_race": {"f1": 0.9}}, "macro_f1": 0.85}},
    ]

    result = aggregate_cv_results(fold_results)

    assert "mean" in result["fault_metrics"]["per_class"]["outer_race"]["f1"]
    assert "mean" in result["fault_metrics"]["macro_f1"]


def test_single_fold_std_is_zero_not_error():
    fold_results = [{"defect_metrics": {"f1": 0.9}}]

    result = aggregate_cv_results(fold_results)

    assert result["defect_metrics"]["f1"]["mean"] == pytest.approx(0.9)
    assert result["defect_metrics"]["f1"]["std"] == 0.0


def test_realistic_full_metrics_shape():
    """Uses the actual shape compute_defect_gate_metrics/compute_fault_type_metrics
    produce, not a simplified example."""
    def fake_fold(defect_f1, macro_f1):
        return {
            "defect_metrics": {
                "normal": {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 161},
                "defect": {"precision": 0.8, "recall": 0.8, "f1": defect_f1, "support": 41},
            },
            "fault_metrics": {
                "per_class": {
                    "outer_race": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 17},
                    "inner_race": {"precision": 0.8, "recall": 0.8, "f1": 0.8, "support": 14},
                    "ball": {"precision": 0.6, "recall": 0.6, "f1": 0.6, "support": 10},
                },
                "macro_f1": macro_f1,
            },
        }

    fold_results = [fake_fold(0.9, 0.7), fake_fold(0.95, 0.75), fake_fold(1.0, 0.8)]

    result = aggregate_cv_results(fold_results)

    assert result["defect_metrics"]["defect"]["f1"]["mean"] == pytest.approx(0.95)
    assert result["fault_metrics"]["macro_f1"]["mean"] == pytest.approx(0.75)
    assert result["fault_metrics"]["per_class"]["ball"]["support"]["mean"] == pytest.approx(10)