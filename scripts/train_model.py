"""
Trains a MultimodalDefectClassifier for the given modality 
and registers the resulting model in MLflow model registry.

Usage:
    python scripts/train_model.py --modality both
"""

import argparse
import logging
from typing import cast

from defect_detection.models.fusion_model import Modality
from defect_detection.training.train import train

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                         force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["both", "image", "vibration"], default="both")
    args = parser.parse_args()

    train(modality=cast(Modality, args.modality))