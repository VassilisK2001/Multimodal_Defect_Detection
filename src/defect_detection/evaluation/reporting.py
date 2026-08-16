import json
from pathlib import Path
import matplotlib.pyplot as plt

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


def save_modality_shuffle_results(results: dict, output_dir: Path, method: str = "shuffle") -> None:
    """Save modality shuffle test results to output_dir/results_<method>.json.
 
    Args:
        results: The dict returned by run_modality_shuffle_test().
        output_dir: Directory to save to.
        method: Corruption method used ("shuffle" or "zero").
    """
    output_dir.mkdir(parents=True, exist_ok=True)
 
    with open(output_dir / f"results_{method}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=json_numpy_default)

def save_oof_results(oof_results: dict, global_metrics: dict, figures: dict, output_dir: Path) -> None:
    """Save OOF cross-validation results to output_dir/<modality>/, one
    subdirectory per model, plus figures at output_dir's root.
 
    Args:
        oof_results: Output of run_oof_cross_validation(). Only each model's
            "fold_metrics" is saved.
        global_metrics: Output of compute_global_oof_metrics().
        figures: Dict mapping a filename stem) to a
            matplotlib Figure. Each is saved as "<stem>.png" and closed afterward.
        output_dir: Base directory. Writes output_dir/<modality>/fold_metrics.json
            and output_dir/<modality>/global_oof_metrics.json per model.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
 
    for modality, model_data in oof_results["models"].items():
        modality_dir = output_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)
 
        with open(modality_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
            json.dump(model_data["fold_metrics"], f, indent=2, default=json_numpy_default)
 
        with open(modality_dir / "global_oof_metrics.json", "w", encoding="utf-8") as f:
            json.dump(global_metrics[modality], f, indent=2, default=json_numpy_default)
 
    for stem, fig in figures.items():
        fig.savefig(output_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
 