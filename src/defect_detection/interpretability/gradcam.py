from typing import Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from defect_detection.models.fusion_model import MultimodalDefectClassifier


class GradCAM:
    """Grad-CAM heatmap generator for a specific model and target layer."""

    def __init__(self, model: MultimodalDefectClassifier, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove_hooks()

    def generate(self, image: torch.Tensor, vib_features: Optional[torch.Tensor] = None,
                 target: Literal["defect", "fault"] = "defect",
                 target_class: Optional[int] = None) -> np.ndarray:
        """Generate a Grad-CAM heatmap for one image.

        Args:
            image: (1, C, H, W) input image tensor.
            vib_features: (1, 5) vibration features. Required if the model has a
                vibration_encoder (modality "both").
            target: "defect" (explain the defect-gate logit) or "fault" (explain
                a fault-type class logit).
            target_class: For target="fault", which class index to explain.
                Defaults to the model's own predicted class if not given.

        Returns:
            (H, W) heatmap, normalized to [0, 1], at the target layer's spatial
            resolution.
        """
        self.model.eval()
        self.model.zero_grad()

        if image.shape[0] != 1:
            raise ValueError(
                f"GradCAM.generate() only supports a batch size of 1, got {image.shape[0]}."
            )

        kwargs = {"image": image}
        if vib_features is not None:
            kwargs["vib_features"] = vib_features

        defect_logit, fault_type_logits = self.model(**kwargs)

        if target == "defect":
            score = defect_logit[0, 0]
        elif target == "fault":
            if target_class is None:
                target_class = int(fault_type_logits.argmax(dim=1).item())
            score = fault_type_logits[0, target_class]
        else:
            raise ValueError(f"Unknown target: {target!r}, expected 'defect' or 'fault'")

        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "No activations/gradients captured. The target layer may not have "
                "been reached during the forward/backward pass."
            )

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        heatmap = F.relu((weights * self._activations).sum(dim=1)).squeeze(0)

        heatmap_np = heatmap.cpu().numpy()
        if heatmap_np.max() > 0:
            heatmap_np = heatmap_np / heatmap_np.max()
        return heatmap_np