
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


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