
import json
from pathlib import Path

def json_numpy_default(obj):
    """Fallback for json.dump: converts a numpy scalar to its native Python
    equivalent. Only called for values the encoder can't already handle."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")



def save_evaluation_results(modality: str, result: dict, output_dir: Path):
    """Save one modality's evaluation results to disk.

    Args:
        modality: "both", "image", or "vibration".
        result: The dict returned by evaluate_model().
        output_dir: Base reports directory.
    """
    modality_dir = output_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "defect_metrics": result["defect_metrics"],
        "fault_metrics": result["fault_metrics"],
    }
    with open(modality_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for name, fig in result["figures"].items():
        fig.savefig(modality_dir / f"{name}.png", dpi=150, bbox_inches="tight")


def save_cv_results(modality: str, fold_results: list[dict], aggregated: dict, output_dir) -> None:
    """Save k-fold CV results to output_dir/modality/: per-fold results and the
    aggregated mean/std summary, each as JSON.

    Args:
        modality: "both", "image", or "vibration".
        fold_results: The raw per-fold results.
        aggregated: fold results aggregated.
        output_dir: Base reports directory
    """
    modality_dir = output_dir / modality
    modality_dir.mkdir(parents=True, exist_ok=True)

    with open(modality_dir / "fold_results.json", "w", encoding="utf-8") as f:
        json.dump(fold_results, f, indent=2, default=json_numpy_default)

    with open(modality_dir / "aggregated.json", "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, default=json_numpy_default)