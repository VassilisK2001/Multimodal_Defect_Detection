"""
Tests for src/defect_detection/interpretability/gradcam.py.
"""


import numpy as np
import pytest
import torch

from defect_detection.interpretability.gradcam import GradCAM
from defect_detection.models.fusion_model import MultimodalDefectClassifier


@pytest.fixture
def model() -> MultimodalDefectClassifier:
    return MultimodalDefectClassifier(modality="image")


@pytest.fixture
def image() -> torch.Tensor:
    return torch.randn(1, 3, 224, 224)


def test_hooks_removed_after_remove_hooks(model, image):
    target_layer = model.image_encoder.layer4[-1]
    n_forward_hooks_before = len(target_layer._forward_hooks)
    n_backward_hooks_before = len(target_layer._backward_hooks)

    cam = GradCAM(model, target_layer)
    cam.remove_hooks()

    assert len(target_layer._forward_hooks) == n_forward_hooks_before
    assert len(target_layer._backward_hooks) == n_backward_hooks_before


def test_hooks_removed_on_context_exit(model, image):
    target_layer = model.image_encoder.layer4[-1]

    with GradCAM(model, target_layer) as cam:
        cam.generate(image, target="defect")

    # A fresh forward pass after exit must not update the captured activations
    stale_activations = cam._activations
    model(image=image)
    assert cam._activations is stale_activations


def test_hooks_removed_even_if_generate_raises(model, image):
    target_layer = model.image_encoder.layer4[-1]

    with pytest.raises(ValueError):
        with GradCAM(model, target_layer) as cam:
            cam.generate(torch.randn(2, 3, 224, 224), target="defect")  # invalid batch size

    assert len(target_layer._forward_hooks) == 0
    assert len(target_layer._backward_hooks) == 0


def test_generate_returns_correct_shape_and_range(model, image):
    with GradCAM(model, model.image_encoder.layer4[-1]) as cam:
        heatmap = cam.generate(image, target="defect")

    assert heatmap.shape == (7, 7)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0


def test_defect_and_fault_targets_produce_different_heatmaps(model, image):
    with GradCAM(model, model.image_encoder.layer4[-1]) as cam:
        defect_heatmap = cam.generate(image, target="defect")
    with GradCAM(model, model.image_encoder.layer4[-1]) as cam:
        fault_heatmap = cam.generate(image, target="fault", target_class=0)

    assert not np.allclose(defect_heatmap, fault_heatmap)


def test_target_class_defaults_to_predicted_class(model, image):
    with torch.no_grad():
        _, fault_logits = model(image=image)
    predicted_class = int(fault_logits.argmax(dim=1).item())

    with GradCAM(model, model.image_encoder.layer4[-1]) as cam:
        default_heatmap = cam.generate(image, target="fault")
    with GradCAM(model, model.image_encoder.layer4[-1]) as cam:
        explicit_heatmap = cam.generate(image, target="fault", target_class=predicted_class)

    assert np.allclose(default_heatmap, explicit_heatmap)


def test_invalid_target_raises_value_error(model, image):
    with GradCAM(model, model.image_encoder.layer4[-1]) as cam:
        with pytest.raises(ValueError):
            cam.generate(image, target="not_a_real_target")


def test_batch_size_greater_than_one_raises_value_error(model):
    with GradCAM(model, model.image_encoder.layer4[-1]) as cam:
        with pytest.raises(ValueError):
            cam.generate(torch.randn(2, 3, 224, 224), target="defect")