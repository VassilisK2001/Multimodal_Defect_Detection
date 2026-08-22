from typing import Callable, cast

import numpy as np
import pandas as pd
import shap
import torch

from defect_detection.data.features import FEATURE_NAMES
from defect_detection.models.fusion_model import MultimodalDefectClassifier


def select_head1_background_rows(train_df: pd.DataFrame, n_samples: int = 75,
                                  seed: int = 42) -> pd.DataFrame:
    """Select a stratified (is_defect) background sample for Head 1.

    Args:
        train_df: Training split manifest.
        n_samples: Target number of background rows.
        seed: Random seed.

    Returns:
        A subset of train_df.
    """
    return train_df.groupby("is_defect", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), n_samples // 2), random_state=seed)
    )


def select_head2_background_rows(train_df: pd.DataFrame, n_samples: int = 75,
                                  ball_oversample_factor: float = 2.0, seed: int = 42) -> pd.DataFrame:
    """Select a background sample for Head 2 (defective rows only), with 'ball'
    deliberately over-represented relative to its true frequency, given its
    small sample count.

    Args:
        train_df: Training split manifest.
        n_samples: Target number of background rows.
        ball_oversample_factor: Relative weight given to 'ball' rows vs. an even
            per-class split.
        seed: Random seed.

    Returns:
        A subset of train_df, defective rows only.
    """
    defective = cast(pd.DataFrame, train_df[train_df.is_defect == 1])
    class_names = cast(pd.Series, defective["fault_class"]).unique()
    weights = {c: (ball_oversample_factor if c == "ball" else 1.0) for c in class_names}
    total_weight = sum(weights.values())

    parts = []
    for class_name, weight in weights.items():
        class_rows = cast(pd.DataFrame, defective[defective.fault_class == class_name])
        n_class = max(1, round(n_samples * weight / total_weight))
        parts.append(class_rows.sample(n=min(len(class_rows), n_class), random_state=seed))
    return pd.concat(parts)


def summarize_background(normalized_features: np.ndarray, k: int = 50, seed: int = 42):
    """Summarize a background sample via k-means.

    Args:
        normalized_features: (N, 5) normalized background features.
        k: Number of representative points to summarize down to.
        seed: Random seed.

    Returns:
        A summarized background, usable directly as an Explainer's background
        argument.
    """
    np.random.seed(seed)
    return shap.kmeans(normalized_features, min(k, len(normalized_features)))


def make_vib_predict_fn(model: MultimodalDefectClassifier, head: str) -> Callable[[np.ndarray], np.ndarray]:
    """Build a black-box prediction function for the vibration-only model.

    Args:
        model: A MultimodalDefectClassifier with modality="vibration".
        head: "defect" or "fault".

    Returns:
        A function (N, 5) normalized features -> (N,) or (N, 3) probabilities.
    """
    @torch.no_grad()
    def predict_fn(vib_features_np: np.ndarray) -> np.ndarray:
        vib_tensor = torch.tensor(vib_features_np, dtype=torch.float32)
        defect_logit, fault_logits = model(vib_features=vib_tensor)
        if head == "defect":
            return torch.sigmoid(defect_logit).squeeze(1).numpy()
        return torch.softmax(fault_logits, dim=1).numpy()

    return predict_fn


def make_fusion_predict_fn(model: MultimodalDefectClassifier, head: str,
                            image: torch.Tensor) -> Callable[[np.ndarray], np.ndarray]:
    """Build a black-box prediction function for the fusion model, with one
    specific instance's real image bound as the fixed input.

    Args:
        model: A MultimodalDefectClassifier with modality="both".
        head: "defect" or "fault".
        image: (3, H, W) real, paired image tensor for one specific test row.

    Returns:
        A function (N, 5) normalized features -> (N,) or (N, 3) probabilities,
        with the given image repeated to match N for every call.
    """
    @torch.no_grad()
    def predict_fn(vib_features_np: np.ndarray) -> np.ndarray:
        vib_tensor = torch.tensor(vib_features_np, dtype=torch.float32)
        image_batch = image.unsqueeze(0).repeat(len(vib_tensor), 1, 1, 1)
        defect_logit, fault_logits = model(image=image_batch, vib_features=vib_tensor)
        if head == "defect":
            return torch.sigmoid(defect_logit).squeeze(1).numpy()
        return torch.softmax(fault_logits, dim=1).numpy()

    return predict_fn


def _kernel_shap_to_explanation(raw_values, expected_value, test_features_raw: np.ndarray,
                                 n_samples: int) -> shap.Explanation:
    """Reconstruct a shap.Explanation from shap.KernelExplainer's raw output."""
    expected_value_arr = np.atleast_1d(np.asarray(expected_value, dtype=float))
    is_multiclass = expected_value_arr.size > 1

    if is_multiclass:
        values = np.stack(raw_values, axis=-1) if isinstance(raw_values, list) else np.asarray(raw_values)
        base_values = np.tile(expected_value_arr, (n_samples, 1))
    else:
        values = np.asarray(raw_values)
        base_values = np.full(n_samples, float(expected_value_arr[0]))

    return shap.Explanation(values=values, base_values=base_values,
                             data=test_features_raw, feature_names=FEATURE_NAMES)


def compute_shap_batched(predict_fn: Callable, background, test_features_normalized: np.ndarray,
                          test_features_raw: np.ndarray, nsamples="auto", l1_reg="auto") -> shap.Explanation:
    """Compute SHAP values in one batched call.

    Args:
        predict_fn: A prediction function.
        background: Output of summarize_background().
        test_features_normalized: (N, 5) normalized test features to explain.
        test_features_raw: (N, 5) raw test features, attached for display.
        nsamples: Passed to KernelExplainer.shap_values().
        l1_reg: Passed to KernelExplainer.shap_values().

    Returns:
        A shap.Explanation with raw feature values attached via `data`.
    """
    explainer = shap.KernelExplainer(predict_fn, background)
    raw_values = explainer.shap_values(test_features_normalized, nsamples=nsamples, l1_reg=l1_reg)
    return _kernel_shap_to_explanation(raw_values, explainer.expected_value, test_features_raw,
                                        len(test_features_normalized))


def compute_shap_per_instance(model: MultimodalDefectClassifier, head: str, background,
                               test_features_normalized: np.ndarray, test_features_raw: np.ndarray,
                               test_images: torch.Tensor, nsamples="auto", l1_reg="auto") -> shap.Explanation:
    """Compute SHAP values for the fusion model, one row at a time, each with
    that row's own paired image bound as the fixed input.

    Args:
        model: A MultimodalDefectClassifier with modality="both".
        head: "defect" or "fault".
        background: Output of summarize_background().
        test_features_normalized: (N, 5) normalized test features to explain.
        test_features_raw: (N, 5) raw test features, attached for display.
        test_images: (N, 3, H, W) real, paired image tensors, one per test row.
        nsamples: Passed to KernelExplainer.shap_values().
        l1_reg: Passed to KernelExplainer.shap_values().

    Returns:
        A shap.Explanation with all N rows' results stacked together, raw
        feature values attached via `data`.
    """
    per_row_values = []
    expected_value = None

    for i in range(len(test_features_normalized)):
        predict_fn = make_fusion_predict_fn(model, head, test_images[i])
        explainer = shap.KernelExplainer(predict_fn, background)
        raw = explainer.shap_values(test_features_normalized[i:i + 1], nsamples=nsamples, l1_reg=l1_reg)
        expected_value = explainer.expected_value

        expected_value_arr = np.atleast_1d(np.asarray(expected_value, dtype=float))
        if expected_value_arr.size > 1:
            row_values = np.stack(raw, axis=-1) if isinstance(raw, list) else np.asarray(raw)
        else:
            row_values = np.asarray(raw)
        per_row_values.append(row_values)

    stacked = np.concatenate(per_row_values, axis=0)
    n_samples = len(test_features_normalized)
    expected_value_arr = np.atleast_1d(np.asarray(expected_value, dtype=float))
    if expected_value_arr.size > 1:
        base_values = np.tile(expected_value_arr, (n_samples, 1))
    else:
        base_values = np.full(n_samples, float(expected_value_arr[0]))

    return shap.Explanation(values=stacked, base_values=base_values,
                             data=test_features_raw, feature_names=FEATURE_NAMES)


def load_images_for_df(df: pd.DataFrame, project_root) -> torch.Tensor:
    """Load and transform each row's real image, matching MultimodalDefectDataset's
    own image pipeline.

    Args:
        df: Manifest rows, with an 'image_path' column.
        project_root: Project root, as returned by find_project_root().

    Returns:
        (len(df), 3, H, W) stacked, transformed image tensor.
    """
    from defect_detection.data.augmentations import build_image_transform
    from PIL import Image

    transform = build_image_transform(training=False)
    images = [transform(Image.open(project_root / p).convert("RGB")) for p in df["image_path"]]
    return torch.stack(images)


def approximate_predictions_from_shap(shap_values: shap.Explanation) -> np.ndarray:
    """Reconstruct approximate model output from
    already-computed SHAP values via the additivity relationship
    (sum(values) + base_value).

    Args:
        shap_values: SHAP values for Head 1 (defect gate).

    Returns:
        (N,) array of approximate output probabilities.
    """
    values = cast(np.ndarray, shap_values.values)
    base_value = np.asarray(shap_values.base_values)
    if base_value.ndim == 0:
        base_value = np.full(values.shape[0], base_value)
    return values.sum(axis=tuple(range(1, values.ndim))) + base_value


def check_additivity_per_instance(model: MultimodalDefectClassifier, head: str,
                                   shap_values: shap.Explanation, test_features_normalized: np.ndarray,
                                   test_images: torch.Tensor, tol: float = 1e-3) -> dict:
    """Check SHAP additivity for the fusion model, one row at a time.

    Args:
        model: A MultimodalDefectClassifier with modality="both".
        head: "defect" or "fault".
        shap_values: Output of compute_shap_per_instance() for this model/head.
        test_features_normalized: The same features used to compute shap_values.
        test_images: The same per-row images used to compute shap_values.
        tol: Per-row tolerance passed to check_additivity.

    Returns:
        Dict with 'max_residual', 'mean_residual' (both across all rows'
        individual max residuals), 'n_within_tolerance', and 'n_total'.
    """
    n_samples = len(test_features_normalized)
    values = cast(np.ndarray, shap_values.values)
    base_values = np.asarray(shap_values.base_values)
    data = cast(np.ndarray, shap_values.data)

    row_max_residuals = []
    n_within_tolerance = 0

    for i in range(n_samples):
        predict_fn = make_fusion_predict_fn(model, head, test_images[i])
        row_shap_values = shap.Explanation(
            values=values[i:i + 1], base_values=base_values[i:i + 1],
            data=data[i:i + 1], feature_names=shap_values.feature_names,
        )
        result = check_additivity(row_shap_values, predict_fn, test_features_normalized[i:i + 1], tol=tol)
        row_max_residuals.append(result["max_residual"])
        if result["within_tolerance"]:
            n_within_tolerance += 1

    return {
        "max_residual": float(np.max(row_max_residuals)),
        "mean_residual": float(np.mean(row_max_residuals)),
        "n_within_tolerance": n_within_tolerance,
        "n_total": n_samples,
    }


def select_top_features(shap_values: shap.Explanation, k: int = 2) -> list[str]:
    """Select the k features with the highest mean absolute SHAP value.

    Args:
        shap_values: SHAP values with feature_names attached.
        k: Number of top features to return.

    Returns:
        A list of k feature names, in descending order of importance.
    """
    values = cast(np.ndarray, shap_values.values)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:k]
    feature_names = shap_values.feature_names
    return [feature_names[i] for i in order]


def check_additivity(shap_values: shap.Explanation, predict_fn: Callable,
                      test_features_normalized: np.ndarray, tol: float = 1e-3) -> dict:
    """Sanity-check SHAP's additivity property: sum(shap_values) + expected_value
    should approximately equal the model's actual output, for the given samples.

    Args:
        shap_values: Output of compute_shap_batched() or compute_shap_per_instance().
        predict_fn: The prediction function the SHAP values were computed against.
        test_features_normalized: The same features passed to compute the SHAP values.
        tol: Maximum acceptable mean absolute residual before flagging a warning.

    Returns:
        Dict with 'max_residual', 'mean_residual', 'within_tolerance'.
    """
    actual_output = predict_fn(test_features_normalized)
    values = cast(np.ndarray, shap_values.values)
    base_value = np.asarray(shap_values.base_values)
    if base_value.ndim == 0:
        base_value = np.full(len(test_features_normalized), base_value)

    predicted_from_shap = values.sum(axis=1) + base_value
    residuals = np.abs(actual_output.reshape(predicted_from_shap.shape) - predicted_from_shap)

    return {
        "max_residual": float(residuals.max()),
        "mean_residual": float(residuals.mean()),
        "within_tolerance": bool(residuals.mean() <= tol),
    }