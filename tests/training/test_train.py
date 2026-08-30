"""
Tests for src/defect_detection/training/train.py.
"""


import math

import pytest
import torch

from unittest.mock import MagicMock, patch
import pandas as pd
from defect_detection.training.train import train_from_dataframes

from defect_detection.models.fusion_model import MultimodalDefectClassifier
from defect_detection.training.losses import TwoStageLoss
from defect_detection.training.train import (
    build_optimizer,
    evaluate,
    train_one_epoch,
)
from tests.factories import make_synthetic_loader

@pytest.fixture
def train_config() -> dict:
    return {"optimizer": {"lr": 1e-3, "weight_decay": 1e-5, "fine_tune_lr_multiplier": 0.1}}


@pytest.fixture
def criterion() -> TwoStageLoss:
    return TwoStageLoss(defect_pos_weight=torch.tensor(2.0), fault_type_class_weights=torch.tensor([1.0, 1.0, 1.0]))


@pytest.mark.parametrize("modality,expect_image,expect_vib", [
    ("both", True, True), ("image", True, False), ("vibration", False, True),
])
def test_optimizer_param_groups_match_modality(modality, expect_image, expect_vib, train_config):
    model = MultimodalDefectClassifier(modality=modality)
    optimizer = build_optimizer(model, train_config)

    param_ids_by_group = [set(id(p) for p in group["params"]) for group in optimizer.param_groups]
    image_param_ids = set(id(p) for p in model.image_encoder.parameters()) if model.image_encoder else set()
    vib_param_ids = set(id(p) for p in model.vibration_encoder.parameters()) if model.vibration_encoder else set()

    has_image_group = any(image_param_ids and image_param_ids <= ids for ids in param_ids_by_group)
    has_vib_group = any(vib_param_ids and vib_param_ids <= ids for ids in param_ids_by_group)

    assert has_image_group == expect_image
    assert has_vib_group == expect_vib


def test_image_encoder_uses_fine_tune_lr(train_config):
    model = MultimodalDefectClassifier(modality="both")
    optimizer = build_optimizer(model, train_config)

    base_lr = train_config["optimizer"]["lr"]
    expected_fine_tune_lr = base_lr * train_config["optimizer"]["fine_tune_lr_multiplier"]

    image_param_ids = set(id(p) for p in model.image_encoder.parameters())
    for group in optimizer.param_groups:
        group_param_ids = set(id(p) for p in group["params"])
        if group_param_ids == image_param_ids:
            assert group["lr"] == pytest.approx(expected_fine_tune_lr)
            return
    pytest.fail("No param group found matching image_encoder parameters")


def test_non_image_groups_use_base_lr(train_config):
    model = MultimodalDefectClassifier(modality="both")
    optimizer = build_optimizer(model, train_config)

    base_lr = train_config["optimizer"]["lr"]
    image_param_ids = set(id(p) for p in model.image_encoder.parameters())

    for group in optimizer.param_groups:
        group_param_ids = set(id(p) for p in group["params"])
        if group_param_ids != image_param_ids:
            assert group["lr"] == pytest.approx(base_lr)


def test_weight_decay_applied(train_config):
    model = MultimodalDefectClassifier(modality="both")
    optimizer = build_optimizer(model, train_config)
    assert optimizer.defaults["weight_decay"] == train_config["optimizer"]["weight_decay"]


@pytest.mark.parametrize("modality", ["both", "image", "vibration"])
def test_train_one_epoch_works_for_all_modalities(modality, criterion, train_config):
    """train_one_epoch should run without error and return valid metrics for every
    modality."""
    model = MultimodalDefectClassifier(modality=modality)
    optimizer = build_optimizer(model, train_config)
    loader = make_synthetic_loader(n_samples=8, n_defective=3)
 
    metrics = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
 
    assert set(metrics.keys()) == {"defect_loss", "fault_type_loss", "defect_accuracy", "fault_type_accuracy"}
    assert not math.isnan(metrics["defect_loss"])


def test_train_one_epoch_returns_expected_keys(criterion, train_config):
    model = MultimodalDefectClassifier(modality="both")
    optimizer = build_optimizer(model, train_config)
    loader = make_synthetic_loader(n_samples=16, n_defective=6)

    metrics = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert set(metrics.keys()) == {"defect_loss", "fault_type_loss", "defect_accuracy", "fault_type_accuracy"}


def test_loss_decreases_over_several_epochs(criterion, train_config):
    """Training on a synthetic task for a few epochs should reduce loss."""
    torch.manual_seed(0)
    model = MultimodalDefectClassifier(modality="both")
    optimizer = build_optimizer(model, train_config)
    loader = make_synthetic_loader(n_samples=32, n_defective=10)

    first_epoch_loss = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))["defect_loss"]
    for _ in range(5):
        last_epoch_loss = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))["defect_loss"]

    assert last_epoch_loss < first_epoch_loss


def test_optimizer_updates_model_weights(criterion, train_config):
    model = MultimodalDefectClassifier(modality="both")
    optimizer = build_optimizer(model, train_config)
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    param_before = next(model.fusion_mlp.parameters()).clone()
    train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    param_after = next(model.fusion_mlp.parameters())

    assert not torch.allclose(param_before, param_after)


def test_train_one_epoch_handles_zero_defective_samples(criterion, train_config):
    """An epoch with no defective samples should return NaN fault-type metrics,
    without crashing or affecting defect-gate training."""
    model = MultimodalDefectClassifier(modality="both")
    optimizer = build_optimizer(model, train_config)
    loader = make_synthetic_loader(n_samples=8, n_defective=0)

    metrics = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))

    assert math.isnan(metrics["fault_type_loss"])
    assert math.isnan(metrics["fault_type_accuracy"])
    assert not math.isnan(metrics["defect_loss"])
    assert not math.isnan(metrics["defect_accuracy"])


def test_evaluate_does_not_change_model_weights(criterion):
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    param_before = next(model.fusion_mlp.parameters()).clone()
    evaluate(model, loader, criterion, torch.device("cpu"))
    param_after = next(model.fusion_mlp.parameters())

    assert torch.allclose(param_before, param_after)


def test_evaluate_is_deterministic(criterion):
    model = MultimodalDefectClassifier(modality="both")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    metrics_1 = evaluate(model, loader, criterion, torch.device("cpu"))
    metrics_2 = evaluate(model, loader, criterion, torch.device("cpu"))

    assert metrics_1 == metrics_2


def test_evaluate_returns_same_keys_as_train_one_epoch(criterion, train_config):
    model = MultimodalDefectClassifier(modality="both")
    optimizer = build_optimizer(model, train_config)
    loader = make_synthetic_loader(n_samples=8, n_defective=3)

    train_metrics = train_one_epoch(model, loader, criterion, optimizer, torch.device("cpu"))
    eval_metrics = evaluate(model, loader, criterion, torch.device("cpu"))

    assert set(train_metrics.keys()) == set(eval_metrics.keys())


def _fake_train_val_df() -> pd.DataFrame:
    """A minimal real DataFrame satisfying compute_defect_gate_pos_weight and
    compute_fault_type_class_weights (all three fault classes present)."""
    return pd.DataFrame({
        "is_defect": [0, 0, 0, 0, 0, 1, 1, 1],
        "fault_class": [None, None, None, None, None, "outer_race", "inner_race", "ball"],
    })
 
 
def _fake_config_side_effect(path: str) -> dict:
    """Returns minimal, self-contained configs for train_from_dataframes, keyed by
    the path argument load_yaml_config is called with.
    """
    if "data_config" in path:
        return {"window_size": 2048, "cwru": {"sampling_rate_hz": 12000}}
    if "model_config" in path:
        return {"image_encoder": {"unfreeze_from": "layer4"}}
    if "train_config" in path:
        return {
            "optimizer": {"lr": 1e-3, "weight_decay": 1e-5, "fine_tune_lr_multiplier": 0.1},
            "training": {"batch_size": 8, "max_epochs": 2, "early_stopping_patience": 5, "min_delta": 0.0},
        }
    raise ValueError(f"Unexpected config path in test: {path}")
 
 
def _run_train_from_dataframes_with_mocks(register_model: bool):
    """Runs train_from_dataframes with every heavy/external dependency mocked,
    returning the mock objects used for mlflow and mlflow.pytorch so callers can
    assert on what was (or wasn't) called."""
    fake_loader_source = make_synthetic_loader(n_samples=8, n_defective=3, batch_size=8)
    fake_dataset = fake_loader_source.dataset
    vib_mean = torch.zeros(5).numpy()
    vib_std = torch.ones(5).numpy()
 
    train_df = _fake_train_val_df()
    val_df = _fake_train_val_df()
 
    with patch("defect_detection.training.train.load_yaml_config", side_effect=_fake_config_side_effect), \
         patch("defect_detection.training.train.build_datasets",
               return_value=(fake_dataset, fake_dataset, vib_mean, vib_std)), \
         patch("defect_detection.training.train.mlflow") as mock_mlflow, \
         patch("defect_detection.training.train.pt") as mock_pt:
 
        mock_run = MagicMock()
        mock_run.info.run_id = "fake_run_id"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
 
        from defect_detection.training.train import train_from_dataframes
        train_from_dataframes(
            train_df, val_df, modality="vibration", register_model=register_model,
        )
 
    return mock_mlflow, mock_pt
 
  
def test_register_model_false_skips_registration():
    """register_model=False should skip both model logging and Model Registry
    registration."""
    mock_mlflow, mock_pt = _run_train_from_dataframes_with_mocks(register_model=False)
 
    mock_pt.log_model.assert_not_called()
    mock_mlflow.register_model.assert_not_called()
 
 
def test_register_model_true_logs_and_registers():
    """register_model=True (the default) should log and register the model, as
    before this flag was introduced."""
    mock_mlflow, mock_pt = _run_train_from_dataframes_with_mocks(register_model=True)
 
    mock_pt.log_model.assert_called_once()
    mock_mlflow.register_model.assert_called_once()
   