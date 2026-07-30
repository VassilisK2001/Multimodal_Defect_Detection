
import torch

from defect_detection.models.fusion_model import MultimodalDefectClassifier


def forward_batch(model: MultimodalDefectClassifier, batch: tuple, device: torch.device
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move a batch to device, route inputs to the model based on its modality, and
    run a forward pass.

    Args:
        model: A MultimodalDefectClassifier in any modality configuration.
        batch: One batch yielded by a DataLoader over MultimodalDefectDataset.
        device: Device to run the forward pass on.

    Returns:
        (defect_logit, fault_type_logits, is_defect, fault_class_idx).
    """
    images, vib_features, is_defect, fault_class_idx, _ = batch
    is_defect = is_defect.to(device)
    fault_class_idx = fault_class_idx.to(device)

    kwargs = {}
    if model.image_encoder is not None:
        kwargs["image"] = images.to(device)
    if model.vibration_encoder is not None:
        kwargs["vib_features"] = vib_features.to(device)

    defect_logit, fault_type_logits = model(**kwargs)
    return defect_logit, fault_type_logits, is_defect, fault_class_idx