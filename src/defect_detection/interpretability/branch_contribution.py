from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from captum.attr import ShapleyValues

from defect_detection.data.features import extract_raw_vib_features_from_df
from defect_detection.data.normalization import apply_vibration_normalization
from defect_detection.data.image_io import load_images_for_df
from defect_detection.models.fusion_model import MultimodalDefectClassifier


def build_feature_masks(image_shape: tuple, vib_shape: tuple) -> tuple:
    """Build Captum feature_mask tensors grouping each entire modality into one
    Shapley player.

    Args:
        image_shape: Shape of a single image tensor.
        vib_shape: Shape of a single vibration-features tensor.

    Returns:
        (image_mask, vib_mask), matching image_shape/vib_shape.
    """
    image_mask = torch.zeros(image_shape, dtype=torch.long)
    vib_mask = torch.ones(vib_shape, dtype=torch.long)
    return image_mask, vib_mask


def prepare_background_samples(background_df: pd.DataFrame, project_root, window_size: int, fs: int,
                                vib_mean: np.ndarray, vib_std: np.ndarray, k: int = 15,
                                seed: int = 42) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Build k real (image, normalized vibration features) background pairs.

    Args:
        background_df: Candidate background rows.
        project_root: Project root, for resolving image paths.
        window_size: Vibration window size in samples.
        fs: Vibration sampling rate in Hz.
        vib_mean: This model's own vibration normalization mean.
        vib_std: This model's own vibration normalization std.
        k: Number of background samples to draw.
        seed: Random seed.

    Returns:
        A list of k (image, vib_features) tensor pairs, each unbatched.
    """
    sampled = background_df.sample(n=min(k, len(background_df)), random_state=seed)
    images = load_images_for_df(sampled, project_root)
    raw_vib = extract_raw_vib_features_from_df(sampled, window_size, fs)
    norm_vib = apply_vibration_normalization(raw_vib, vib_mean, vib_std)
    vib_tensor = torch.tensor(norm_vib, dtype=torch.float32)

    return [(images[i], vib_tensor[i]) for i in range(len(sampled))]


def _make_defect_forward(model: MultimodalDefectClassifier) -> Callable:
    def forward(image: torch.Tensor, vib_features: torch.Tensor) -> torch.Tensor:
        logit, _ = model(image=image, vib_features=vib_features)
        return torch.sigmoid(logit).squeeze(1)
    return forward


def _make_fault_forward(model: MultimodalDefectClassifier) -> Callable:
    def forward(image: torch.Tensor, vib_features: torch.Tensor) -> torch.Tensor:
        _, logits = model(image=image, vib_features=vib_features)
        return torch.softmax(logits, dim=1)
    return forward


def _get_forward_and_target(model: MultimodalDefectClassifier, head: str,
                             target_class: Optional[int]) -> tuple[Callable, Optional[int]]:
    if head == "defect":
        return _make_defect_forward(model), None
    if head == "fault":
        return _make_fault_forward(model), target_class
    raise ValueError(f"Unknown head: {head!r}, expected 'defect' or 'fault'")


def compute_branch_contributions(model: MultimodalDefectClassifier, head: str, images: torch.Tensor,
                                  vib_features: torch.Tensor,
                                  background_samples: list[tuple[torch.Tensor, torch.Tensor]],
                                  target_class: Optional[int] = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute exact 2-player Shapley attribution to the image and vibration
    branches, for a batch of test samples, averaged over the given background
    samples.

    Args:
        model: A MultimodalDefectClassifier with modality="both".
        head: "defect" or "fault".
        images: (N, 3, H, W) test images.
        vib_features: (N, 5) normalized test vibration features.
        background_samples: Output of prepare_background_samples().
        target_class: For head="fault", which class index to attribute.
            Ignored for head="defect".

    Returns:
        (phi_image, phi_vib, se_info): phi_image/phi_vib each (N,), averaged
        over all background samples; se_info is a dict with 'se_image',
        'se_vib' (N,) standard errors of the mean across the K background
        draws, and 'k' (the number of background samples used).
    """
    forward_fn, target = _get_forward_and_target(model, head, target_class)
    model.eval()
    image_mask, vib_mask = build_feature_masks(tuple(images.shape[1:]), tuple(vib_features.shape[1:]))
    image_mask_batched = image_mask.unsqueeze(0)
    vib_mask_batched = vib_mask.unsqueeze(0)

    explainer = ShapleyValues(forward_fn)

    per_background_image = []
    per_background_vib = []

    for bg_image, bg_vib in background_samples:
        baselines = (bg_image.unsqueeze(0), bg_vib.unsqueeze(0))
        image_attr, vib_attr = explainer.attribute(
            inputs=(images, vib_features), baselines=baselines,
            feature_mask=(image_mask_batched, vib_mask_batched), target=target,
        )
    
        per_background_image.append(
            image_attr.mean(dim=tuple(range(1, image_attr.ndim))).detach().numpy()
        )
        per_background_vib.append(
            vib_attr.mean(dim=tuple(range(1, vib_attr.ndim))).detach().numpy()
        )

    k = len(background_samples)
    phi_image = np.mean(per_background_image, axis=0)
    phi_vib = np.mean(per_background_vib, axis=0)
    se_image = np.std(per_background_image, axis=0, ddof=1) / np.sqrt(k)
    se_vib = np.std(per_background_vib, axis=0, ddof=1) / np.sqrt(k)

    return phi_image, phi_vib, {"se_image": se_image, "se_vib": se_vib, "k": k}


def check_shapley_additivity_sample(model: MultimodalDefectClassifier, head: str, images: torch.Tensor,
                                     vib_features: torch.Tensor,
                                     background_samples: list[tuple[torch.Tensor, torch.Tensor]],
                                     target_class: Optional[int] = None, n_check: int = 3,
                                     tol: float = 1e-4) -> dict:
    """Verify exact 2-player Shapley additivity on a handful of (test row,
    background sample) pairs.

    Args:
        model: A MultimodalDefectClassifier with modality="both".
        head: "defect" or "fault".
        images: Test images to spot-check against.
        vib_features: Test vibration features to spot-check against.
        background_samples: Output of prepare_background_samples().
        target_class: For head="fault", which class index to check.
        n_check: Number of (row, background) pairs to spot-check.
        tol: Maximum acceptable residual.

    Returns:
        Dict with 'max_residual' and 'within_tolerance'.
    """
    forward_fn, target = _get_forward_and_target(model, head, target_class)
    model.eval()
    image_mask, vib_mask = build_feature_masks(tuple(images.shape[1:]), tuple(vib_features.shape[1:]))
    explainer = ShapleyValues(forward_fn)

    residuals = []
    for i in range(min(n_check, images.shape[0])):
        bg_image, bg_vib = background_samples[i % len(background_samples)]
        row_image = images[i:i + 1]
        row_vib = vib_features[i:i + 1]
        baselines = (bg_image.unsqueeze(0), bg_vib.unsqueeze(0))

        image_attr, vib_attr = explainer.attribute(
            inputs=(row_image, row_vib), baselines=baselines,
            feature_mask=(image_mask.unsqueeze(0), vib_mask.unsqueeze(0)), target=target,
        )
        phi_image = image_attr.mean().item()
        phi_vib = vib_attr.mean().item()

        with torch.no_grad():
            v_joint = forward_fn(row_image, row_vib)
            v_base = forward_fn(bg_image.unsqueeze(0), bg_vib.unsqueeze(0))
            if head == "fault":
                v_joint = v_joint[0, target].item()
                v_base = v_base[0, target].item()
            else:
                v_joint = v_joint.item()
                v_base = v_base.item()

        residuals.append(abs((phi_image + phi_vib) - (v_joint - v_base)))

    return {"max_residual": float(max(residuals)), "within_tolerance": max(residuals) <= tol}