
import torch

from defect_detection.models.vibration_encoder import build_vibration_encoder
from defect_detection.utils import load_yaml_config


# Output shape

def test_output_shape_matches_defaults():
    """Default arguments should map (batch, 5) -> (batch, 64)."""
    encoder = build_vibration_encoder()
    output = encoder(torch.randn(4, 5))
    assert output.shape == (4, 64)


def test_output_shape_with_custom_dims():
    """input_dim and embedding_dim should control input/output size."""
    encoder = build_vibration_encoder(input_dim=8, embedding_dim=32)
    output = encoder(torch.randn(4, 8))
    assert output.shape == (4, 32)


# Config-driven defaults

def test_defaults_match_model_config():
    """Omitted arguments should fall back to config/model_config.yaml."""
    model_config = load_yaml_config("config/model_config.yaml")["vibration_encoder"]

    default_encoder = build_vibration_encoder()
    explicit_encoder = build_vibration_encoder(
        input_dim=model_config["input_dim"], embedding_dim=model_config["embedding_dim"],
    )

    dummy_input = torch.randn(2, model_config["input_dim"])
    assert default_encoder(dummy_input).shape == explicit_encoder(dummy_input).shape


def test_explicit_arguments_override_config():
    """Explicit arguments should take precedence over config defaults."""
    encoder = build_vibration_encoder(input_dim=10, embedding_dim=16)
    output = encoder(torch.randn(2, 10))
    assert output.shape == (2, 16)


# All parameters trainable

def test_all_parameters_are_trainable():
    """Every parameter should have requires_grad=True."""
    encoder = build_vibration_encoder()
    for param in encoder.parameters():
        assert param.requires_grad


# Gradient flow

def test_all_parameters_receive_gradient():
    """After a forward and backward pass, every parameter should have a gradient."""
    encoder = build_vibration_encoder()
    output = encoder(torch.randn(4, 5))
    loss = output.sum()
    loss.backward()

    for param in encoder.parameters():
        assert param.grad is not None
        assert not torch.isnan(param.grad).any()


# Determinism

def test_output_is_deterministic():
    """Identical input should produce identical output."""
    encoder = build_vibration_encoder()
    encoder.eval()
    dummy_input = torch.randn(4, 5)

    output1 = encoder(dummy_input)
    output2 = encoder(dummy_input)
    assert torch.allclose(output1, output2)


# Batch size handling

def test_handles_various_batch_sizes():
    """Should produce correctly shaped output for any batch size."""
    encoder = build_vibration_encoder()
    for batch_size in [1, 4, 32]:
        output = encoder(torch.randn(batch_size, 5))
        assert output.shape == (batch_size, 64)


# Output validity

def test_output_has_no_nans_or_infs():
    """Output should never contain NaN or Inf values."""
    encoder = build_vibration_encoder()
    output = encoder(torch.randn(8, 5))
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()