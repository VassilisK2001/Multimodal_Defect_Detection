
import torch
from torchvision.models import resnet18

from defect_detection.models.image_encoder import (
    build_image_encoder, count_trainable_parameters,
)


# Output shape 

def test_output_shape_matches_embedding_dim():
    """Output shape should be (batch, embedding_dim)."""
    encoder = build_image_encoder(embedding_dim=128)
    dummy_batch = torch.randn(4, 3, 224, 224)
    output = encoder(dummy_batch)
    assert output.shape == (4, 128)


def test_different_embedding_dims_are_respected():
    """embedding_dim should control the output size for any value passed."""
    for dim in [32, 64, 256]:
        encoder = build_image_encoder(embedding_dim=dim)
        output = encoder(torch.randn(2, 3, 224, 224))
        assert output.shape == (2, dim)


# Layer freezing

def test_layers_before_unfreeze_point_are_frozen():
    """Modules preceding unfreeze_from should have requires_grad=False."""
    encoder = build_image_encoder(unfreeze_from="layer4")
    frozen_submodules = ["conv1", "bn1", "layer1", "layer2", "layer3"]

    for name in frozen_submodules:
        submodule = getattr(encoder, name)
        for param in submodule.parameters():
            assert not param.requires_grad, f"{name} should be frozen but is trainable"


def test_layers_from_unfreeze_point_are_trainable():
    """unfreeze_from and later modules, including the replaced fc layer, should have
    requires_grad=True."""
    encoder = build_image_encoder(unfreeze_from="layer4")
    trainable_submodules = ["layer4", "fc"]

    for name in trainable_submodules:
        submodule = getattr(encoder, name)
        for param in submodule.parameters():
            assert param.requires_grad, f"{name} should be trainable but is frozen"


def test_different_unfreeze_point_changes_which_layers_are_trainable():
    """unfreeze_from should determine the freeze boundary for any valid module name."""
    encoder = build_image_encoder(unfreeze_from="layer3")
    for param in encoder.layer3.parameters():
        assert param.requires_grad


# count_trainable_parameters

def test_count_trainable_parameters_matches_requires_grad():
    """Reported counts should match a direct requires_grad-based count."""
    encoder = build_image_encoder(unfreeze_from="layer4")
    trainable, total = count_trainable_parameters(encoder)

    manual_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    manual_total = sum(p.numel() for p in encoder.parameters())

    assert trainable == manual_trainable
    assert total == manual_total
    assert 0 < trainable < total


# Pretrained weights

def test_pretrained_weights_differ_from_random_init():
    """Pretrained weights should differ from a freshly random-initialized model."""
    pretrained_encoder = build_image_encoder()
    random_model = resnet18(weights=None)

    pretrained_conv1 = pretrained_encoder.conv1.weight.detach()
    random_conv1 = random_model.conv1.weight.detach()

    assert not torch.allclose(pretrained_conv1, random_conv1)


# Gradient flow

def test_frozen_parameters_receive_no_gradient():
    """After a forward and backward pass, frozen layers should have no gradient and
    trainable layers should have one."""
    encoder = build_image_encoder(unfreeze_from="layer4")
    dummy_batch = torch.randn(2, 3, 224, 224)

    output = encoder(dummy_batch)
    loss = output.sum()
    loss.backward()

    for param in encoder.layer1.parameters():
        assert param.grad is None, "Frozen layer1 should have no gradient"

    for param in encoder.layer4.parameters():
        assert param.grad is not None, "Trainable layer4 should have a gradient"

    for param in encoder.fc.parameters():
        assert param.grad is not None, "Trainable fc should have a gradient"