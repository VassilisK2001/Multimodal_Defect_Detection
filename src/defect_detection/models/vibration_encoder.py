

import torch.nn as nn

from defect_detection.utils import load_yaml_config


def build_vibration_encoder(input_dim: int | None = None,
                             embedding_dim: int | None = None) -> nn.Module:
    """Build an MLP that encodes a vibration feature vector into a fixed-size embedding.

    Args:
        input_dim: Number of input features. Defaults to config/model_config.yaml's
            vibration_encoder.input_dim if not given.
        embedding_dim: Size of the output embedding. Defaults to
            config/model_config.yaml's vibration_encoder.embedding_dim if not given.

    Returns:
        An nn.Sequential mapping (batch, input_dim) -> (batch, embedding_dim).
    """
    if input_dim is None or embedding_dim is None:
        config = load_yaml_config("config/model_config.yaml")["vibration_encoder"]
        input_dim = input_dim if input_dim is not None else config["input_dim"]
        embedding_dim = embedding_dim if embedding_dim is not None else config["embedding_dim"]

    return nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(),
        nn.Linear(32, embedding_dim),
        nn.ReLU(),
    )