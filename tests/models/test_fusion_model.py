
import pytest
import torch

from defect_detection.models.fusion_model import MultimodalDefectClassifier
from defect_detection.utils import load_yaml_config


# Output shapes per modality

def test_both_modality_output_shapes():
    """modality='both' should return (batch, 1) and (batch, num_fault_classes)."""
    model = MultimodalDefectClassifier(modality="both")
    image = torch.randn(4, 3, 224, 224)
    vib = torch.randn(4, 5)

    defect_logit, fault_logits = model(image=image, vib_features=vib)

    assert defect_logit.shape == (4, 1)
    assert fault_logits.shape == (4, 3)


def test_image_only_modality_output_shapes():
    """modality='image' should produce the same output shapes using only image input."""
    model = MultimodalDefectClassifier(modality="image")
    image = torch.randn(4, 3, 224, 224)

    defect_logit, fault_logits = model(image=image)

    assert defect_logit.shape == (4, 1)
    assert fault_logits.shape == (4, 3)


def test_vibration_only_modality_output_shapes():
    """modality='vibration' should produce the same output shapes using only
    vibration input."""
    model = MultimodalDefectClassifier(modality="vibration")
    vib = torch.randn(4, 5)

    defect_logit, fault_logits = model(vib_features=vib)

    assert defect_logit.shape == (4, 1)
    assert fault_logits.shape == (4, 3)


# Missing input / invalid modality handling

def test_image_modality_requires_image_input():
    """Should raise if image is missing for modality='image'."""
    model = MultimodalDefectClassifier(modality="image")
    with pytest.raises(ValueError):
        model(vib_features=torch.randn(2, 5))


def test_vibration_modality_requires_vibration_input():
    """Should raise if vib_features is missing for modality='vibration'."""
    model = MultimodalDefectClassifier(modality="vibration")
    with pytest.raises(ValueError):
        model(image=torch.randn(2, 3, 224, 224))


def test_both_modality_requires_both_inputs():
    """Should raise if either input is missing for modality='both'."""
    model = MultimodalDefectClassifier(modality="both")
    with pytest.raises(ValueError):
        model(image=torch.randn(2, 3, 224, 224))
    with pytest.raises(ValueError):
        model(vib_features=torch.randn(2, 5))


def test_invalid_modality_raises_at_construction():
    """Should raise for an unrecognized modality value."""
    with pytest.raises(ValueError):
        MultimodalDefectClassifier(modality="images")


# Fusion input dimension correctness

def test_fusion_input_dim_for_both():
    """Fusion layer input size should equal img_dim + vib_dim for modality='both'."""
    model = MultimodalDefectClassifier(modality="both", img_embedding_dim=128, vib_embedding_dim=64)
    first_layer = model.fusion_mlp[0]
    assert first_layer.in_features == 128 + 64


def test_fusion_input_dim_for_image_only():
    """Fusion layer input size should equal img_dim for modality='image'."""
    model = MultimodalDefectClassifier(modality="image", img_embedding_dim=128)
    first_layer = model.fusion_mlp[0]
    assert first_layer.in_features == 128


def test_fusion_input_dim_for_vibration_only():
    """Fusion layer input size should equal vib_dim for modality='vibration'."""
    model = MultimodalDefectClassifier(modality="vibration", vib_embedding_dim=64)
    first_layer = model.fusion_mlp[0]
    assert first_layer.in_features == 64


# Architecture consistency across modalities

def test_fusion_and_head_architecture_identical_across_modalities():
    """Fusion hidden size and head output sizes should be identical regardless of
    modality."""
    both_model = MultimodalDefectClassifier(modality="both")
    image_model = MultimodalDefectClassifier(modality="image")
    vib_model = MultimodalDefectClassifier(modality="vibration")

    for model in [both_model, image_model, vib_model]:
        assert model.fusion_mlp[0].out_features == 128
        assert model.defect_head.out_features == 1
        assert model.fault_type_head.out_features == 3


# Config-driven defaults

def test_defaults_loaded_from_model_config():
    """Omitted arguments should fall back to config/model_config.yaml."""
    config = load_yaml_config("config/model_config.yaml")["fusion"]
    model = MultimodalDefectClassifier(modality="both")

    assert model.fusion_mlp[0].out_features == config["hidden_dim"]
    assert model.fault_type_head.out_features == config["num_fault_classes"]


def test_explicit_arguments_override_config():
    """Explicit arguments should take precedence over config defaults."""
    model = MultimodalDefectClassifier(modality="both", fusion_hidden_dim=64, num_fault_classes=5)
    assert model.fusion_mlp[0].out_features == 64
    assert model.fault_type_head.out_features == 5


def test_zero_dropout_not_overridden_by_config():
    """dropout=0.0 should be applied as is, not replaced by the config default."""
    model = MultimodalDefectClassifier(modality="both", dropout=0.0)
    dropout_layer = model.fusion_mlp[2]
    assert isinstance(dropout_layer, torch.nn.Dropout)
    assert dropout_layer.p == 0.0


# Gradient flow

@pytest.mark.parametrize("modality", ["both", "image", "vibration"])
def test_gradients_flow_through_full_model(modality):
    """After a forward and backward pass, the fusion layer and both heads should have
    gradients, for every modality setting."""
    model = MultimodalDefectClassifier(modality=modality)
    kwargs = {}
    if modality in ("both", "image"):
        kwargs["image"] = torch.randn(2, 3, 224, 224)
    if modality in ("both", "vibration"):
        kwargs["vib_features"] = torch.randn(2, 5)

    defect_logit, fault_logits = model(**kwargs)
    loss = defect_logit.sum() + fault_logits.sum()
    loss.backward()

    for name, param in model.fusion_mlp.named_parameters():
        assert param.grad is not None, f"fusion_mlp.{name} has no gradient"
    for name, param in model.defect_head.named_parameters():
        assert param.grad is not None, f"defect_head.{name} has no gradient"
    for name, param in model.fault_type_head.named_parameters():
        assert param.grad is not None, f"fault_type_head.{name} has no gradient"


# Output sanity

def test_different_inputs_produce_different_outputs():
    """Distinct random inputs should not produce identical logits."""
    model = MultimodalDefectClassifier(modality="both")
    model.eval()

    image1, vib1 = torch.randn(1, 3, 224, 224), torch.randn(1, 5)
    image2, vib2 = torch.randn(1, 3, 224, 224), torch.randn(1, 5)

    with torch.no_grad():
        defect1, fault1 = model(image=image1, vib_features=vib1)
        defect2, fault2 = model(image=image2, vib_features=vib2)

    assert not torch.allclose(defect1, defect2)
    assert not torch.allclose(fault1, fault2)