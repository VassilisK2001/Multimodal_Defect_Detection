
import json
from pathlib import Path


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