
import pandas as pd
import torch
import torch.nn as nn

from defect_detection.utils import load_yaml_config


def _fault_class_order() -> list[str]:
    """Return fault class names in the order matching fault_class_idx (0/1/2)."""
    config = load_yaml_config("config/data_config.yaml")
    return [ft["name"] for ft in config["cwru"]["fault_types"]]


def compute_fault_type_class_weights(train_df: pd.DataFrame, beta: float | None = None) -> torch.Tensor:
    """Compute per-class weights using the inverse effective number of samples.

    weight_i = (1 - beta) / (1 - beta^n_i), normalized to average 1.0, where n_i is
    the number of training rows for class i.

    Args:
        train_df: Training split manifest with 'is_defect' and 'fault_class' columns.
        beta: Effective-number hyperparameter, in [0, 1). Defaults to
            config/train_config.yaml's loss.fault_type_beta if not given.

    Returns:
        A (num_fault_classes,) tensor, ordered to match fault_class_idx.

    Raises:
        ValueError: If any fault class has zero samples in train_df.
    """
    if beta is None:
        beta = load_yaml_config("config/train_config.yaml")["loss"]["fault_type_beta"]

    fault_classes = _fault_class_order()
    defective = train_df[train_df.is_defect == 1]
    counts = defective["fault_class"].value_counts().to_dict()

    missing = [cls for cls in fault_classes if cls not in counts]
    if missing:
        raise ValueError(f"train_df has no samples for fault class(es): {missing}")

    effective_nums = {cls: (1 - beta ** counts[cls]) / (1 - beta) for cls in fault_classes}
    raw_weights = {cls: 1.0 / effective_nums[cls] for cls in fault_classes}
    mean_weight = sum(raw_weights.values()) / len(raw_weights)

    return torch.tensor([raw_weights[cls] / mean_weight for cls in fault_classes], dtype=torch.float32)


def compute_defect_gate_pos_weight(train_df: pd.DataFrame) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss as the ratio of negative to positive samples.

    Args:
        train_df: Training split manifest with an 'is_defect' column.

    Returns:
        A scalar tensor equal to N_negative / N_positive.

    Raises:
        ValueError: If train_df has no positive (is_defect == 1) samples.
    """
    n_neg = int((train_df.is_defect == 0).sum())
    n_pos = int((train_df.is_defect == 1).sum())

    if n_pos == 0:
        raise ValueError("train_df has no defective (is_defect == 1) samples")

    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


class TwoStageLoss(nn.Module):
    """Combined loss for a binary defect gate and a multi-class fault-type head."""

    def __init__(self, defect_pos_weight: torch.Tensor, fault_type_class_weights: torch.Tensor):
        """
        Args:
            defect_pos_weight: pos_weight passed to the defect gate's BCEWithLogitsLoss.
            fault_type_class_weights: Per-class weights passed to the fault-type
                head's CrossEntropyLoss.
        """
        super().__init__()
        self.defect_criterion = nn.BCEWithLogitsLoss(pos_weight=defect_pos_weight)
        self.fault_type_criterion = nn.CrossEntropyLoss(weight=fault_type_class_weights)

    def forward(self, defect_logit: torch.Tensor, fault_type_logits: torch.Tensor,
                is_defect: torch.Tensor, fault_class_idx: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
        """Compute the combined loss for one batch.

        The fault-type loss is computed only over samples where is_defect == 1.

        Args:
            defect_logit: (batch, 1) raw logits from the defect gate.
            fault_type_logits: (batch, num_fault_classes) raw logits.
            is_defect: (batch,) float tensor, 1.0 if defective else 0.0.
            fault_class_idx: (batch,) long tensor, fault class index for defective
                samples.

        Returns:
            total_loss: Combined loss, always a valid tensor.
            defect_loss: Defect gate loss.
            fault_type_loss: Fault-type loss, or None if the batch contains no
                defective samples.
            n_defective: Number of defective samples in the batch.
        """
        defect_loss = self.defect_criterion(defect_logit.squeeze(1), is_defect)

        defect_mask = is_defect.bool()
        n_defective = int(defect_mask.sum().item())

        if n_defective > 0:
            fault_type_loss = self.fault_type_criterion(
                fault_type_logits[defect_mask], fault_class_idx[defect_mask],
            )
            total_loss = defect_loss + fault_type_loss
        else:
            fault_type_loss = None
            total_loss = defect_loss

        return total_loss, defect_loss, fault_type_loss, n_defective