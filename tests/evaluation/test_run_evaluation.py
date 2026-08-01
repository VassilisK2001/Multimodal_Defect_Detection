
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch

from defect_detection.evaluation.run_evaluation import evaluate_model


@contextmanager
def _patch_all_dependencies():
    """Mock every function evaluate_model calls and yield the mocks by name."""
    predictions = {
        "is_defect_true": np.array([0, 1]),
        "is_defect_pred": np.array([0, 1]),
        "defect_proba": np.array([0.1, 0.9]),
        "fault_class_true": np.array([1, 2]),
        "fault_class_pred": np.array([1, 0]),
    }

    mocks = {
        "load_model_and_stats": MagicMock(return_value=(
            MagicMock(), np.array([1.0, 2.0, 3.0, 4.0, 5.0]), np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        )),
        "MultimodalDefectDataset": MagicMock(),
        "collect_test_predictions": MagicMock(return_value=predictions),
        "compute_defect_gate_metrics": MagicMock(return_value={"defect": "metrics"}),
        "compute_fault_type_metrics": MagicMock(return_value={"fault": "metrics"}),
        "plot_defect_gate_confusion_matrix": MagicMock(return_value="defect_fig"),
        "plot_fault_type_confusion_matrix": MagicMock(return_value="fault_fig"),
    }

    with patch.multiple("defect_detection.evaluation.run_evaluation", **mocks):
        yield mocks


def _call_evaluate_model(**overrides: Any) -> dict:
    kwargs: dict[str, Any] = dict(
        modality="both", test_df=pd.DataFrame(), window_size=2048, fs=12000,
        class_names=["outer_race", "inner_race", "ball"], device=torch.device("cpu"),
    )
    kwargs.update(overrides)
    return evaluate_model(**kwargs)


def test_returns_expected_top_level_keys():
    with _patch_all_dependencies():
        result = _call_evaluate_model()

    assert set(result.keys()) == {"defect_metrics", "fault_metrics", "figures"}


def test_vib_stats_reach_dataset_construction():
    """vib_mean and vib_std should reach the Dataset's constructor unchanged."""
    with _patch_all_dependencies() as mocks:
        _call_evaluate_model()

    dataset_call_kwargs = mocks["MultimodalDefectDataset"].call_args.kwargs
    assert np.array_equal(dataset_call_kwargs["vib_mean"], np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert np.array_equal(dataset_call_kwargs["vib_std"], np.array([0.5, 0.5, 0.5, 0.5, 0.5]))


def test_device_passed_to_both_model_loading_and_prediction_collection():
    """device should be passed to both load_model_and_stats and 
    collect_test_predictions."""
    device = torch.device("cuda:0")

    with _patch_all_dependencies() as mocks:
        _call_evaluate_model(device=device)

    assert mocks["load_model_and_stats"].call_args.kwargs["device"] == device
    assert mocks["collect_test_predictions"].call_args.kwargs["device"] == device


def test_defect_and_fault_arrays_are_not_crossed():
    """Defect and fault-type arrays should each reach their own metric function,
    not be swapped."""
    with _patch_all_dependencies() as mocks:
        _call_evaluate_model()

    defect_call_args = mocks["compute_defect_gate_metrics"].call_args.args
    fault_call_args = mocks["compute_fault_type_metrics"].call_args.args

    assert np.array_equal(defect_call_args[0], np.array([0, 1]))       # is_defect_true
    assert np.array_equal(defect_call_args[1], np.array([0, 1]))       # is_defect_pred
    assert np.array_equal(fault_call_args[0], np.array([1, 2]))        # fault_class_true
    assert np.array_equal(fault_call_args[1], np.array([1, 0]))        # fault_class_pred
    assert fault_call_args[2] == ["outer_race", "inner_race", "ball"]  # class_names


def test_figures_dict_contains_both_plots():
    with _patch_all_dependencies():
        result = _call_evaluate_model()

    assert result["figures"] == {
        "defect_confusion_matrix": "defect_fig",
        "fault_type_confusion_matrix": "fault_fig",
    }