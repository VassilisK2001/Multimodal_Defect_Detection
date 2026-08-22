import matplotlib.pyplot as plt
import numpy as np
import torch
import shap
from scipy.ndimage import zoom

from defect_detection.interpretability.gradcam import GradCAM

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Undo ImageNet normalization for display.

    Args:
        image: (3, H, W) normalized image tensor.

    Returns:
        (H, W, 3) array in [0, 1], ready for imshow.
    """
    denorm = image.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return denorm.clamp(0, 1).permute(1, 2, 0).numpy()


def resize_heatmap(heatmap: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a Grad-CAM heatmap to a target (H, W) shape via bilinear interpolation.

    Args:
        heatmap: (h, w) heatmap, as returned by GradCAM.generate().
        target_shape: (H, W) to resize to.

    Returns:
        (H, W) resized heatmap.
    """
    zoom_factors = (target_shape[0] / heatmap.shape[0], target_shape[1] / heatmap.shape[1])
    return zoom(heatmap, zoom_factors, order=1)


def plot_gradcam_overlay(image: np.ndarray, heatmap: np.ndarray, ax=None, title: str = "",
                          alpha: float = 0.4):
    """Plot an image with its Grad-CAM heatmap overlaid.

    Args:
        image: (H, W, 3) image, denormalized and scaled for display.
        heatmap: (H, W) heatmap, normalized to [0, 1], resized to match image.
        ax: Existing matplotlib Axes to draw onto, for use within a grid of
            subplots. If None, a new standalone Figure is created.
        title: Plot title.
        alpha: Heatmap overlay opacity.

    Returns:
        A matplotlib Figure if ax was None, otherwise None.
    """
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(image)
    ax.imshow(heatmap, cmap="jet", alpha=alpha)
    ax.set_title(title)
    ax.axis("off")

    if fig is not None:
        fig.tight_layout()
    return fig


def _render_gradcam_panel(model, dataset, row_index: int, ax, title: str,
                           target: str, target_class: int | None = None) -> None:
    """Generate one example's Grad-CAM heatmap and draw it onto ax."""
    image, vib_features, _, _, _ = dataset[row_index]
    image_batch = image.unsqueeze(0)
    vib_batch = vib_features.unsqueeze(0) if model.vibration_encoder is not None else None

    with GradCAM(model, model.image_encoder.layer4[-1]) as cam:
        heatmap = cam.generate(image_batch, vib_batch, target=target, target_class=target_class)

    heatmap_resized = resize_heatmap(heatmap, target_shape=tuple(image.shape[1:]))
    image_display = denormalize_image(image)
    plot_gradcam_overlay(image_display, heatmap_resized, ax=ax, title=title)


def _render_missing_panel(ax, title: str) -> None:
    """Draw a placeholder panel for a case with no matching example."""
    ax.set_title(f"{title}\n(no example found)")
    ax.axis("off")


def plot_defect_gate_gradcam_grid(model, dataset, defect_examples: dict, n: int = 3) -> plt.Figure:
    """Plot Grad-CAM for the defect gate: up to n correct-defective,
    false-negative, and false-positive examples, one row per case.

    Args:
        model: The MultimodalDefectClassifier to explain.
        dataset: MultimodalDefectDataset that the examples' row indices index into.
        defect_examples: Output of example_selection.find_defect_gate_examples().
        n: Number of example columns per case.

    Returns:
        A matplotlib Figure with len(defect_examples) rows x n columns.
    """
    case_names = list(defect_examples.keys())
    fig, axes = plt.subplots(len(case_names), n, figsize=(5 * n, 5 * len(case_names)), squeeze=False)

    for row_i, case_name in enumerate(case_names):
        row_indices = defect_examples[case_name]
        for col_i in range(n):
            ax = axes[row_i, col_i]
            title = case_name if col_i == 0 else ""
            if col_i < len(row_indices):
                _render_gradcam_panel(model, dataset, row_indices[col_i], ax, title, target="defect")
            else:
                _render_missing_panel(ax, title or case_name)

    fig.suptitle("Grad-CAM: Defect Gate", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_fault_type_gradcam_grid(model, dataset, fault_examples: dict,
                                  class_names: list[str], n: int = 3) -> plt.Figure:
    """Plot Grad-CAM for the fault-type head: up to n correct and n misclassified
    examples per class, both explaining the model's predicted class.

    Args:
        model: The MultimodalDefectClassifier to explain.
        dataset: MultimodalDefectDataset that the examples' row indices index into.
        fault_examples: Output of example_selection.find_fault_type_examples().
        class_names: Fault class names, in index order.
        n: Number of example columns per (class, case) row.

    Returns:
        A matplotlib Figure with (2 x len(class_names)) rows x n columns.
    """
    row_labels = [(class_name, case_name) for class_name in class_names for case_name in ("correct", "misclassified")]
    fig, axes = plt.subplots(len(row_labels), n, figsize=(5 * n, 5 * len(row_labels)), squeeze=False)

    for row_i, (class_name, case_name) in enumerate(row_labels):
        entries = fault_examples[class_name][case_name]
        row_title = f"{class_name} \u2014 {case_name}"
        for col_i in range(n):
            ax = axes[row_i, col_i]
            title = row_title if col_i == 0 else ""
            if col_i < len(entries):
                entry = entries[col_i]
                _render_gradcam_panel(
                    model, dataset, entry["row_index"], ax, title,
                    target="fault", target_class=entry["predicted_class"],
                )
            else:
                _render_missing_panel(ax, title or row_title)

    fig.suptitle("Grad-CAM: Fault Type (explaining the predicted class)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_beeswarm_comparison(vib_shap_values: shap.Explanation, fusion_shap_values: shap.Explanation,
                              title: str = "") -> plt.Figure:
    """Plot vibration-only and fusion beeswarm plots side by side.
 
    Args:
        vib_shap_values: SHAP values for the vibration-only model.
        fusion_shap_values: SHAP values for the fusion model.
        title: Overall figure title.
 
    Returns:
        A matplotlib Figure with 2 subplots.
    """
    fig, (ax_vib, ax_fusion) = plt.subplots(1, 2, figsize=(14, 5))
 
    plt.sca(ax_vib)
    shap.summary_plot(vib_shap_values, show=False, plot_size=None)
    ax_vib.set_title("Vibration-only")
 
    plt.sca(ax_fusion)
    shap.summary_plot(fusion_shap_values, show=False, plot_size=None)
    ax_fusion.set_title("Fusion (both)")
 
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
 
 
def plot_dependence(shap_values: shap.Explanation, feature_name: str, title: str = "") -> plt.Figure:
    """Plot a SHAP dependence plot for one feature, on raw (unnormalized) values.
 
    Args:
        shap_values: SHAP values with raw feature values attached via `data`.
        feature_name: Which feature to plot (must be in shap_values.feature_names).
        title: Plot title.
 
    Returns:
        A matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    shap.dependence_plot(feature_name, shap_values.values, shap_values.data,
                          feature_names=shap_values.feature_names, ax=ax, show=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig
 
 
def plot_waterfall(shap_values: shap.Explanation, row_index: int, title: str = "") -> plt.Figure:
    """Plot a SHAP waterfall plot for one specific test instance.
 
    Args:
        shap_values: SHAP values for a full test subset.
        row_index: Which row (within shap_values) to plot.
        title: Plot title.
 
    Returns:
        A matplotlib Figure.
    """
    fig = plt.figure(figsize=(8, 5))
    shap.waterfall_plot(shap_values[row_index], show=False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig