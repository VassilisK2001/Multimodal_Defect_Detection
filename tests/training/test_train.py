
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from defect_detection.models.fusion_model import MultimodalDefectClassifier
from defect_detection.training.losses import TwoStageLoss
from defect_detection.training.train import (
    _forward_batch,
    build_optimizer,
    evaluate,
    train_one_epoch,
)


def make_synthetic_loader(n_samples: int, n_defective: int, batch_size: int = 8) -> DataLoader:
    """Build a synthetic DataLoader yielding (image, vib_features, is_defect,
    fault_class_idx, area_ratio) batches."""
    images = torch.randn(n_samples, 3, 224, 224)
    vib_features = torch.randn(n_samples, 5)

    is_defect = torch.zeros(n_samples)
    is_defect[:n_defective] = 1.0

    fault_class_idx = torch.full((n_samples,), -1, dtype=torch.long)
    if n_defective > 0:
        fault_class_idx[:n_defective] = torch.randint(0, 3, (n_defective,))

    area_ratio = torch.zeros(n_samples)

    dataset = TensorDataset(images, vib_features, is_defect, fault_class_idx, area_ratio)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


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
def test_forward_batch_routes_inputs_correctly(modality):
    model = MultimodalDefectClassifier(modality=modality)
    device = torch.device("cpu")
    loader = make_synthetic_loader(n_samples=8, n_defective=2)
    batch = next(iter(loader))

    defect_logit, fault_type_logits, is_defect, fault_class_idx = _forward_batch(model, batch, device)

    assert defect_logit.shape == (8, 1)
    assert fault_type_logits.shape == (8, 3)
    assert is_defect.device == device
    assert fault_class_idx.device == device


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