
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
    )