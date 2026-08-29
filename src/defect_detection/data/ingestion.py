import logging
import shutil
from pathlib import Path

import kagglehub

logger = logging.getLogger(__name__)


def download_dataset(slug: str, target_dir: Path, name: str) -> None:
    """Download a Kaggle dataset via kagglehub and copy it into target_dir,
    skipping the download if target_dir already has content.

    Args:
        slug: Kaggle dataset slug.
        target_dir: Directory to copy the dataset's contents into.
        name: Display name for log messages.
    """
    if target_dir.exists() and any(target_dir.iterdir()):
        logger.info("[%s] already present at %s, skipping download", name, target_dir)
        return

    logger.info("[%s] downloading via kagglehub...", name)
    cache_path = Path(kagglehub.dataset_download(slug))
    logger.info("[%s] downloaded to cache: %s", name, cache_path)

    target_dir.mkdir(parents=True, exist_ok=True)
    for item in cache_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    logger.info("[%s] copied into %s", name, target_dir)


def verify_structure(mvtec_dir: Path, cwru_dir: Path) -> dict:
    """Verify the downloaded datasets have the expected top-level structure.

    Args:
        mvtec_dir: MVTec AD root directory.
        cwru_dir: CWRU root directory.

    Returns:
        Dict with 'mvtec_categories' (list of category directory names) and
        'cwru_mat_file_count' (int).
    """
    mvtec_categories = [d.name for d in mvtec_dir.iterdir() if d.is_dir()] if mvtec_dir.exists() else []
    cwru_mat_files = list(cwru_dir.rglob("*.mat")) if cwru_dir.exists() else []

    logger.info("MVTec categories found: %s", mvtec_categories)
    logger.info("CWRU .mat files found: %d", len(cwru_mat_files))
    if cwru_mat_files:
        logger.info("  example: %s", cwru_mat_files[0])

    return {"mvtec_categories": mvtec_categories, "cwru_mat_file_count": len(cwru_mat_files)}