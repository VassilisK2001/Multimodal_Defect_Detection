
import shutil
from pathlib import Path
import kagglehub

RAW_DATA_DIR = Path("data/raw")
MVTEC_DIR = RAW_DATA_DIR / "mvtec"
CWRU_DIR = RAW_DATA_DIR / "cwru"

MVTEC_KAGGLE_SLUG = "ipythonx/mvtec-ad"      
CWRU_KAGGLE_SLUG = "astrollama/cwru-case-western-reserve-university-dataset" 


def download_dataset(slug: str, target_dir: Path, name: str) -> None:
    if target_dir.exists() and any(target_dir.iterdir()):
        print(f"[{name}] already present at {target_dir}, skipping download")
        return

    print(f"[{name}] downloading via kagglehub...")
    cache_path = Path(kagglehub.dataset_download(slug))
    print(f"[{name}] downloaded to cache: {cache_path}")

    target_dir.mkdir(parents=True, exist_ok=True)
    for item in cache_path.iterdir():
        dest = target_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)

    print(f"[{name}] copied into {target_dir}")


def verify_structure() -> None:
    print("\nVerifying dataset structure...")

    mvtec_categories = [d.name for d in MVTEC_DIR.iterdir() if d.is_dir()] if MVTEC_DIR.exists() else []
    print(f"MVTec categories found: {mvtec_categories}")

    cwru_files = list(CWRU_DIR.rglob("*.mat")) if CWRU_DIR.exists() else []
    print(f"CWRU .mat files found: {len(cwru_files)}")
    if cwru_files:
        print(f"  example: {cwru_files[0]}")


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_dataset(MVTEC_KAGGLE_SLUG, MVTEC_DIR, "MVTec AD")
    download_dataset(CWRU_KAGGLE_SLUG, CWRU_DIR, "CWRU")
    verify_structure()


if __name__ == "__main__":
    main()