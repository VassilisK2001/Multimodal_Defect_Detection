
import pandas as pd
import pytest
import torch
import torch.nn as nn

from defect_detection.training.losses import (
    TwoStageLoss,
    _fault_class_order,
    compute_defect_gate_pos_weight,
    compute_fault_type_class_weights,
)


# Fault class order
def test_fault_class_order_has_three_classes():
    """Config should define exactly the three expected fault classes."""
    order = _fault_class_order()
    assert len(order) == 3
    assert set(order) == {"outer_race", "inner_race", "ball"}


# compute_fault_type_class_weights

def test_fault_type_weights_hand_verifiable():
    """Equal class counts should produce equal weights (1.0 after normalization)."""
    order = _fault_class_order()
    df = pd.DataFrame({
        "is_defect": [1] * (10 * len(order)),
        "fault_class": [cls for cls in order for _ in range(10)],
    })
    weights = compute_fault_type_class_weights(df, beta=0.99)
    assert torch.allclose(weights, torch.ones(len(order)), atol=1e-4)


def test_fault_type_weights_monotonic_with_class_size():
    """Rarer classes should receive higher weight than more common ones."""
    order = _fault_class_order()
    sizes = [50, 30, 20]
    df = pd.DataFrame({
        "is_defect": [1] * sum(sizes),
        "fault_class": [cls for cls, n in zip(order, sizes) for _ in range(n)],
    })
    weights = compute_fault_type_class_weights(df, beta=0.99)
    weight_dict = dict(zip(order, weights.tolist()))

    largest, middle, smallest = order
    assert weight_dict[smallest] > weight_dict[middle] > weight_dict[largest]


def test_fault_type_weights_normalized_to_mean_one():
    """Weights should be normalized to average 1.0."""
    order = _fault_class_order()
    sizes = [81, 63, 45]
    df = pd.DataFrame({
        "is_defect": [1] * sum(sizes),
        "fault_class": [cls for cls, n in zip(order, sizes) for _ in range(n)],
    })
    weights = compute_fault_type_class_weights(df)
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-4)


def test_fault_type_weights_default_beta_matches_train_config():
    """Omitting beta should use config/train_config.yaml's loss.fault_type_beta."""
    from defect_detection.utils import load_yaml_config

    order = _fault_class_order()
    df = pd.DataFrame({
        "is_defect": [1] * (10 * len(order)),
        "fault_class": [cls for cls in order for _ in range(10)],
    })
    config_beta = load_yaml_config("config/train_config.yaml")["loss"]["fault_type_beta"]

    default_weights = compute_fault_type_class_weights(df)
    explicit_weights = compute_fault_type_class_weights(df, beta=config_beta)

    assert torch.allclose(default_weights, explicit_weights)


def test_fault_type_weights_raises_on_missing_class():
    """Should raise if any fault class has zero samples in train_df."""
    order = _fault_class_order()
    df = pd.DataFrame({
        "is_defect": [1] * (10 * (len(order) - 1)),
        "fault_class": [cls for cls in order[:-1] for _ in range(10)],
    })
    with pytest.raises(ValueError):
        compute_fault_type_class_weights(df)


# compute_defect_gate_pos_weight

def test_defect_gate_pos_weight_hand_verifiable():
    """pos_weight should equal N_negative / N_positive."""
    df = pd.DataFrame({"is_defect": [0] * 100 + [1] * 25})
    pos_weight = compute_defect_gate_pos_weight(df)
    assert pos_weight.item() == pytest.approx(4.0)


def test_defect_gate_pos_weight_raises_when_no_positives():
    """Should raise if train_df has no positive samples."""
    df = pd.DataFrame({"is_defect": [0] * 50})
    with pytest.raises(ValueError):
        compute_defect_gate_pos_weight(df)


# TwoStageLoss: batch with defective samples

@pytest.fixture
def loss_fn():
    """Fixed, arbitrary weight values used to test TwoStageLoss's mechanics."""
    pos_weight = torch.tensor(4.0)
    fault_weights = torch.tensor([0.8, 1.0, 1.4])
    return TwoStageLoss(pos_weight, fault_weights)


def test_batch_with_defective_samples_returns_valid_fault_loss(loss_fn):
    """A batch containing defective samples should return a valid fault-type loss
    and the correct defective-sample count."""
    defect_logit = torch.randn(6, 1)
    fault_logits = torch.randn(6, 3)
    is_defect = torch.tensor([1., 0., 1., 0., 0., 1.])
    fault_class_idx = torch.tensor([0, -1, 1, -1, -1, 2])

    total_loss, defect_loss, fault_type_loss, n_defective = loss_fn(
        defect_logit, fault_logits, is_defect, fault_class_idx,
    )

    assert n_defective == 3
    assert fault_type_loss is not None
    assert torch.isclose(total_loss, defect_loss + fault_type_loss)


def test_masking_excludes_normal_samples_correctly(loss_fn):
    """Fault-type loss should match computing CrossEntropyLoss directly on only the
    defective rows."""
    defect_logit = torch.randn(5, 1)
    fault_logits = torch.randn(5, 3)
    is_defect = torch.tensor([1., 0., 1., 0., 1.])
    fault_class_idx = torch.tensor([0, -1, 2, -1, 1])

    _, _, fault_type_loss, _ = loss_fn(defect_logit, fault_logits, is_defect, fault_class_idx)

    defect_mask = is_defect.bool()
    reference_criterion = nn.CrossEntropyLoss(weight=loss_fn.fault_type_criterion.weight)
    expected = reference_criterion(fault_logits[defect_mask], fault_class_idx[defect_mask])

    assert torch.isclose(fault_type_loss, expected)


# TwoStageLoss: batch with zero defective samples

def test_batch_with_no_defective_samples_returns_none_fault_loss(loss_fn):
    """A batch with zero defective samples should return fault_type_loss=None and
    n_defective=0, with total_loss equal to just the defect loss."""
    defect_logit = torch.randn(4, 1)
    fault_logits = torch.randn(4, 3)
    is_defect = torch.zeros(4)
    fault_class_idx = torch.full((4,), -1)

    total_loss, defect_loss, fault_type_loss, n_defective = loss_fn(
        defect_logit, fault_logits, is_defect, fault_class_idx,
    )

    assert fault_type_loss is None
    assert n_defective == 0
    assert torch.isclose(total_loss, defect_loss)


# Gradient flow

def test_backward_works_with_defective_samples(loss_fn):
    """Backward pass should populate gradients for both heads' inputs."""
    defect_logit = torch.randn(4, 1, requires_grad=True)
    fault_logits = torch.randn(4, 3, requires_grad=True)
    is_defect = torch.tensor([1., 1., 0., 0.])
    fault_class_idx = torch.tensor([0, 1, -1, -1])

    total_loss, _, _, _ = loss_fn(defect_logit, fault_logits, is_defect, fault_class_idx)
    total_loss.backward()

    assert defect_logit.grad is not None
    assert fault_logits.grad is not None


def test_backward_works_with_no_defective_samples(loss_fn):
    """Backward pass should succeed even when fault_type_loss is None."""
    defect_logit = torch.randn(4, 1, requires_grad=True)
    fault_logits = torch.randn(4, 3, requires_grad=True)
    is_defect = torch.zeros(4)
    fault_class_idx = torch.full((4,), -1)

    total_loss, _, _, _ = loss_fn(defect_logit, fault_logits, is_defect, fault_class_idx)
    total_loss.backward()

    assert defect_logit.grad is not None


# Weighting actually applied

def test_pos_weight_changes_loss_value():
    """A weighted BCE loss should differ from an unweighted one on an imbalanced batch."""
    defect_logit = torch.randn(10, 1)
    is_defect = torch.tensor([1.] + [0.] * 9)

    unweighted = nn.functional.binary_cross_entropy_with_logits(
        defect_logit.squeeze(1), is_defect,
    )
    weighted = nn.functional.binary_cross_entropy_with_logits(
        defect_logit.squeeze(1), is_defect, pos_weight=torch.tensor(9.0),
    )

    assert not torch.isclose(unweighted, weighted)