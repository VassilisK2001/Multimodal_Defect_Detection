"""
Tests for src/defect_detection/models/batch_utils.py.
"""


import pytest
import torch

from defect_detection.models.batch_utils import forward_batch
from defect_detection.models.fusion_model import MultimodalDefectClassifier
from tests.factories import make_synthetic_loader


@pytest.mark.parametrize("modality", ["both", "image", "vibration"])
def test_forward_batch_returns_correct_output_shapes(modality):
    """Output logits should have the expected shape regardless of modality."""
    model = MultimodalDefectClassifier(modality=modality)
    device = torch.device("cpu")
    loader = make_synthetic_loader(n_samples=8, n_defective=2)
    batch = next(iter(loader))

    defect_logit, fault_type_logits, is_defect, fault_class_idx = forward_batch(model, batch, device)

    assert defect_logit.shape == (8, 1)
    assert fault_type_logits.shape == (8, 3)
    assert is_defect.shape == (8,)
    assert fault_class_idx.shape == (8,)


@pytest.mark.parametrize("modality", ["both", "image", "vibration"])
def test_forward_batch_moves_labels_to_device(modality):
    """is_defect and fault_class_idx should be moved to the requested device."""
    model = MultimodalDefectClassifier(modality=modality)
    device = torch.device("cpu")
    loader = make_synthetic_loader(n_samples=8, n_defective=2)
    batch = next(iter(loader))

    _, _, is_defect, fault_class_idx = forward_batch(model, batch, device)

    assert is_defect.device == device
    assert fault_class_idx.device == device


def test_forward_batch_preserves_label_values():
    """is_defect and fault_class_idx values should be unchanged by the pass through
    forward_batch, only moved to device."""
    model = MultimodalDefectClassifier(modality="both")
    device = torch.device("cpu")
    loader = make_synthetic_loader(n_samples=8, n_defective=3)
    batch = next(iter(loader))
    _, _, original_is_defect, original_fault_class_idx, _ = batch

    _, _, is_defect, fault_class_idx = forward_batch(model, batch, device)

    assert torch.equal(is_defect.cpu(), original_is_defect)
    assert torch.equal(fault_class_idx.cpu(), original_fault_class_idx)