import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve


MODEL_DISPLAY_NAMES = {"image": "Image", "vibration": "Vibration", "both": "Fusion"}
MODEL_COLORS = {"image": "tab:orange", "vibration": "tab:green", "both": "tab:blue"}


def plot_defect_gate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """Plot a confusion matrix for the binary defect gate.

    Args:
        y_true: (N,) binary array, 1 if defective else 0.
        y_pred: (N,) binary array, thresholded predictions.

    Returns:
        A matplotlib Figure with a single heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["normal", "defect"], yticklabels=["normal", "defect"], ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Defect gate confusion matrix")
    fig.tight_layout()
    return fig


def plot_fault_type_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                                      class_names: list[str]) -> plt.Figure:
    """Plot a confusion matrix for the fault-type head.

    Args:
        y_true: (N,) integer class-index array, defective samples only.
        y_pred: (N,) integer class-index array, defective samples only.
        class_names: Class names in index order.

    Returns:
        A matplotlib Figure with a single heatmap.
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Fault-type confusion matrix")
    fig.tight_layout()
    return fig

def plot_defect_gate_global_curves(oof_results: dict, global_metrics: dict) -> plt.Figure:
    """Plot global OOF ROC and PR curves for the defect gate, all models overlaid.
 
    Args:
        oof_results: Output of run_oof_cross_validation().
        global_metrics: Output of compute_global_oof_metrics().
 
    Returns:
        A matplotlib Figure with two subplots: ROC curve (left), PR curve (right).
    """
    is_defect_true = oof_results["is_defect_true"]
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
 
    for modality, model_data in oof_results["models"].items():
        defect_proba = model_data["oof_defect_proba"]
        color = MODEL_COLORS.get(modality)
        label = MODEL_DISPLAY_NAMES.get(modality, modality)
        roc_auc = global_metrics[modality]["defect"]["roc_auc"]
        pr_auc = global_metrics[modality]["defect"]["pr_auc"]
 
        fpr, tpr, _ = roc_curve(is_defect_true, defect_proba)
        ax_roc.plot(fpr, tpr, color=color, label=f"{label} (AUC={roc_auc:.3f})")
 
        precision, recall, _ = precision_recall_curve(is_defect_true, defect_proba)
        ax_pr.plot(recall, precision, color=color, label=f"{label} (AUC={pr_auc:.3f})")
 
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Defect Gate \u2014 ROC Curve (Global OOF)")
    ax_roc.legend(fontsize=8)
 
    defect_rate = is_defect_true.mean()
    ax_pr.axhline(defect_rate, linestyle="--", color="gray", label=f"baseline ({defect_rate:.3f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Defect Gate \u2014 PR Curve (Global OOF)")
    ax_pr.legend(fontsize=8)
 
    fig.tight_layout()
    return fig
 
 
def plot_fault_type_global_pr_curves(oof_results: dict, global_metrics: dict,
                                      class_names: list[str],
                                      subplot_order: list[str] | None = None) -> plt.Figure:
    """Plot global OOF one-vs-rest PR curves per fault class, all models overlaid.
 
    Args:
        oof_results: Output of run_oof_cross_validation().
        global_metrics: Output of compute_global_oof_metrics().
        class_names: Fault class names, in index order.
        subplot_order: Class display order, left to right. Defaults to
            class_names' order if not given.
 
    Returns:
        A matplotlib Figure with one subplot per class, sharing the y-axis.
    """
    order = subplot_order or class_names
    class_to_idx = {name: i for i, name in enumerate(class_names)}
 
    is_defect_true = oof_results["is_defect_true"]
    fault_class_true = oof_results["fault_class_true"]
    defective_mask = is_defect_true == 1
    fault_true_binarized = label_binarize(
        fault_class_true[defective_mask], classes=range(len(class_names)),
    )
 
    fig, axes = plt.subplots(1, len(order), figsize=(5 * len(order), 5), sharey=True)
    if len(order) == 1:
        axes = [axes]
 
    for ax, class_name in zip(axes, order):
        class_idx = class_to_idx[class_name]
        y_true = fault_true_binarized[:, class_idx]
 
        for modality, model_data in oof_results["models"].items():
            fault_proba_defective = model_data["oof_fault_proba"][defective_mask]
            y_score = fault_proba_defective[:, class_idx]
            color = MODEL_COLORS.get(modality)
            label = MODEL_DISPLAY_NAMES.get(modality, modality)
            pr_auc = global_metrics[modality]["fault"][class_name]["pr_auc"]
 
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ax.plot(recall, precision, color=color, label=f"{label} (AUC={pr_auc:.3f})")
 
        ax.set_xlabel("Recall")
        ax.set_title(class_name)
        ax.legend(fontsize=8)
 
    axes[0].set_ylabel("Precision")
    fig.suptitle("Fault-Type PR Curves (Global OOF, one-vs-rest)")
    fig.tight_layout()
    return fig


def plot_bootstrap_distribution(values: np.ndarray, mean_value: float, std_value: float,
                                 xlabel: str = "", title: str = "") -> plt.Figure:
    """Plot a histogram of a bootstrap-resampled statistic's distribution, with
    the mean marked and a shaded +/- 1 std band.
 
    Args:
        values: (B,) array of the statistic computed on each bootstrap resample.
        mean_value: Mean across resamples.
        std_value: Standard deviation across resamples.
        xlabel: X-axis label.
        title: Plot title.
 
    Returns:
        A matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.hist(values, bins=30, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(mean_value, color="red", linestyle="-", linewidth=2,
               label=f"mean = {mean_value:.3f}")
    ax.axvspan(mean_value - std_value, mean_value + std_value, color="red", alpha=0.15,
               label=f"\u00b1 1 std = {std_value:.3f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig