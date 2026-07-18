
from typing import Literal

import torch
import torch.nn as nn

from defect_detection.models.image_encoder import build_image_encoder
from defect_detection.models.vibration_encoder import build_vibration_encoder
from defect_detection.utils import load_yaml_config

Modality = Literal["both", "image", "vibration"]


class MultimodalDefectClassifier(nn.Module):
    def __init__(self, modality: Modality = "both",
                 img_embedding_dim: int | None = None,
                 vib_embedding_dim: int | None = None,
                 fusion_hidden_dim: int | None = None,
                 dropout: float | None = None,
                 num_fault_classes: int | None = None):
        """Build the classifier for a given modality configuration.

        Args:
            modality: "both" (full fusion model), "image", or "vibration" —
                controls which encoder(s) are built and what feeds the fusion layer.
            img_embedding_dim: Image encoder output size. Defaults to
                config/model_config.yaml's image_encoder.embedding_dim.
            vib_embedding_dim: Vibration encoder output size. Defaults to
                config/model_config.yaml's vibration_encoder.embedding_dim.
            fusion_hidden_dim: Fusion MLP hidden size. Defaults to
                config/model_config.yaml's fusion.hidden_dim.
            dropout: Fusion MLP dropout rate. Defaults to
                config/model_config.yaml's fusion.dropout.
            num_fault_classes: Number of fault-type classes. Defaults to
                config/model_config.yaml's fusion.num_fault_classes.
        """
        super().__init__()

        config = load_yaml_config("config/model_config.yaml")
        img_embedding_dim = img_embedding_dim or config["image_encoder"]["embedding_dim"]
        vib_embedding_dim = vib_embedding_dim or config["vibration_encoder"]["embedding_dim"]
        fusion_hidden_dim = fusion_hidden_dim or config["fusion"]["hidden_dim"]
        dropout = dropout if dropout is not None else config["fusion"]["dropout"]
        num_fault_classes = num_fault_classes or config["fusion"]["num_fault_classes"]

        if modality not in ("both", "image", "vibration"):
            raise ValueError(f"Unknown modality: {modality}")
        self.modality = modality

        self.image_encoder = (
            build_image_encoder(embedding_dim=img_embedding_dim)
            if modality in ("both", "image") else None
        )
        self.vibration_encoder = (
            build_vibration_encoder(embedding_dim=vib_embedding_dim)
            if modality in ("both", "vibration") else None
        )

        fusion_input_dim = {
            "both": img_embedding_dim + vib_embedding_dim,
            "image": img_embedding_dim,
            "vibration": vib_embedding_dim,
        }[modality]

        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.defect_head = nn.Linear(fusion_hidden_dim, 1)
        self.fault_type_head = nn.Linear(fusion_hidden_dim, num_fault_classes)

    def forward(self, image: torch.Tensor | None = None,
                vib_features: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            image: (batch, 3, 224, 224) tensor. Required unless modality == "vibration".
            vib_features: (batch, 5) tensor. Required unless modality == "image".

        Returns:
            (defect_logit, fault_type_logits): shapes (batch, 1) and
            (batch, num_fault_classes).
        """
        embeddings = []

        if self.image_encoder is not None:
            if image is None:
                raise ValueError(f"modality='{self.modality}' requires `image` input")
            embeddings.append(self.image_encoder(image))

        if self.vibration_encoder is not None:
            if vib_features is None:
                raise ValueError(f"modality='{self.modality}' requires `vib_features` input")
            embeddings.append(self.vibration_encoder(vib_features))

        fused_input = torch.cat(embeddings, dim=1) if len(embeddings) > 1 else embeddings[0]
        fused = self.fusion_mlp(fused_input)

        defect_logit = self.defect_head(fused)
        fault_type_logits = self.fault_type_head(fused)

        return defect_logit, fault_type_logits