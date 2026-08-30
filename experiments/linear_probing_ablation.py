"""
Diagnostic ablation: freezes the entire ResNet18 backbone (only the replaced
fc layer trainable) to test whether the image-only baseline's severe
training-time overfitting is capacity-driven.

Usage:
    python -m experiments.linear_probing_ablation --modality image
    python -m experiments.linear_probing_ablation --modality both
"""
import argparse

from defect_detection.training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["both", "image"], default="image")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        modality=args.modality,
        seed=args.seed,
        unfreeze_from="fc",
        run_name_suffix="_linear_probe",
        register_model=False,
    )