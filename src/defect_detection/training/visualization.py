
import matplotlib.pyplot as plt


def _plot_train_val_curve(train_values: list, val_values: list, ylabel: str, title: str) -> plt.Figure:
    """Shared helper: a single train/val line plot over epochs."""
    epochs = range(1, len(train_values) + 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_values, label=f"train {ylabel}")
    ax.plot(epochs, val_values, label=f"val {ylabel}")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_defect_loss_curve(history: dict, modality: str = "both") -> plt.Figure:
    """Plot train/val defect gate loss over epochs.

    Args:
        history: Dict with 'train_defect_loss' and 'val_defect_loss' lists.
        modality: Used only in the figure title.
    """
    return _plot_train_val_curve(
        history["train_defect_loss"], history["val_defect_loss"],
        ylabel="defect loss", title=f"Defect gate loss ({modality})",
    )


def plot_fault_type_loss_curve(history: dict, modality: str = "both") -> plt.Figure:
    """Plot train/val fault-type head loss over epochs.

    Args:
        history: Dict with 'train_fault_type_loss' and 'val_fault_type_loss' lists.
        modality: Used only in the figure title.
    """
    return _plot_train_val_curve(
        history["train_fault_type_loss"], history["val_fault_type_loss"],
        ylabel="fault-type loss", title=f"Fault-type head loss ({modality})",
    )


def plot_defect_accuracy_curve(history: dict, modality: str = "both") -> plt.Figure:
    """Plot train/val defect gate accuracy over epochs.

    Args:
        history: Dict with 'train_defect_accuracy' and 'val_defect_accuracy' lists.
        modality: Used only in the figure title.
    """
    fig = _plot_train_val_curve(
        history["train_defect_accuracy"], history["val_defect_accuracy"],
        ylabel="defect accuracy", title=f"Defect gate accuracy ({modality})",
    )
    fig.axes[0].set_ylim(0, 1)
    return fig


def plot_fault_type_accuracy_curve(history: dict, modality: str = "both") -> plt.Figure:
    """Plot train/val fault-type head accuracy over epochs.

    Args:
        history: Dict with 'train_fault_type_accuracy' and 'val_fault_type_accuracy' lists.
        modality: Used only in the figure title.
    """
    fig = _plot_train_val_curve(
        history["train_fault_type_accuracy"], history["val_fault_type_accuracy"],
        ylabel="fault-type accuracy", title=f"Fault-type head accuracy ({modality})",
    )
    fig.axes[0].set_ylim(0, 1)
    return fig