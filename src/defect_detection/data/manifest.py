
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from scipy.io import loadmat

from defect_detection.utils import find_project_root


def build_fault_code_maps(config: dict) -> tuple[dict, dict]:
    """Derive code->name and name->code lookups from config's cwru.fault_types"""
    fault_types = config["cwru"]["fault_types"]
    code_to_name = {ft["code"]: ft["name"] for ft in fault_types}
    name_to_code = {ft["name"]: ft["code"] for ft in fault_types}
    return code_to_name, name_to_code


def load_config(config_path: str = "config/data_config.yaml") -> dict:
    project_root = find_project_root()
    with open(project_root / config_path) as f:
        return yaml.safe_load(f)


def get_defect_area_ratio(mask_path: Path) -> float:
    mask = np.array(Image.open(mask_path).convert("L"))
    return float(np.sum(mask > 127) / mask.size)


def get_signal_length(mat_path: Path) -> int:
    mat = loadmat(mat_path)
    de_key = [k for k in mat.keys() if "DE_time" in k][0]
    return mat[de_key].flatten().shape[0]


def build_cwru_inventory(cwru_dir: Path, config: dict, project_root: Path) -> pd.DataFrame:
    """Index every CWRU .mat file by condition/fault_type/severity, with its signal length"""
    window_size = config["window_size"]
    code_to_name, _ = build_fault_code_maps(config)
    records = []

    for path in sorted(cwru_dir.glob("*.mat")):
        name = path.stem
        n_samples = get_signal_length(path)
        n_windows = n_samples // window_size
        rel_path = path.relative_to(project_root).as_posix()

        if name.startswith(config["cwru"]["normal_prefix"]):
            records.append({
                "path": rel_path, "condition": "normal",
                "fault_type": None, "severity": None,
                "n_windows": n_windows,
            })
            continue

        match = re.match(r"(B|IR|OR)(\d{3})(\d)?_(\d)", name)
        if not match:
            print(f"Warning: skipping unrecognized CWRU filename: {name}")
            continue

        fault_code, diameter_code, _position, _idx = match.groups()
        fault_type = code_to_name[fault_code] 
        records.append({
            "path": rel_path, "condition": "fault",
            "fault_type": fault_type, "severity": diameter_code,
            "n_windows": n_windows,
        })

    return pd.DataFrame(records)


def build_mvtec_records(mvtec_dir: Path, config: dict, project_root: Path) -> list[dict]:
    """One record per MVTec image: category, defect_type, is_defect, fault_class"""
    records = []
    defect_to_fault_map = config["defect_to_fault_map"]

    for category in config["mvtec"]["categories"]:
        category_defect_map = defect_to_fault_map.get(category, {})

        # 'good' images from both train/ and test/ — pooled, since MVTec's official
        # train/test split is designed for unsupervised AD, not this project's use case
        for split_dir in ["train", "test"]:
            good_dir = mvtec_dir / category / split_dir / "good"
            if good_dir.exists():
                for img_path in sorted(good_dir.glob("*.png")):
                    records.append({
                        "image_path": img_path.relative_to(project_root).as_posix(),
                        "category": category,
                        "defect_type": "good",
                        "is_defect": 0,
                        "fault_class": "none",
                        "defect_area_ratio": 0.0,
                    })

        # Defective images only for defect types present in the mapping table
        test_dir = mvtec_dir / category / "test"
        gt_dir = mvtec_dir / category / "ground_truth"
        for defect_type, fault_class in category_defect_map.items():
            img_dir = test_dir / defect_type
            mask_dir = gt_dir / defect_type
            if not img_dir.exists():
                print(f"Warning: expected defect dir missing: {img_dir}")
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                mask_path = mask_dir / f"{img_path.stem}_mask.png"
                area_ratio = get_defect_area_ratio(mask_path) if mask_path.exists() else np.nan

                records.append({
                    "image_path": img_path.relative_to(project_root).as_posix(),
                    "category": category,
                    "defect_type": defect_type,
                    "is_defect": 1,
                    "fault_class": fault_class,
                    "defect_area_ratio": area_ratio,
                })

    return records


def assign_vibration_sample(row: pd.Series, cwru_inventory: pd.DataFrame,
                             config: dict, rng: random.Random) -> dict:
    """Pick a CWRU file + window index for one MVTec image record.
    Fault type is matched exactly; severity is chosen at random among available options"""
    if row["is_defect"] == 0:
        candidates = cwru_inventory[cwru_inventory.condition == "normal"]
    else:
        candidates = cwru_inventory[
            (cwru_inventory.condition == "fault") &
            (cwru_inventory.fault_type == row["fault_class"])
        ]

    if candidates.empty:
        raise ValueError(f"No CWRU files found for fault_class={row['fault_class']}")

    chosen_file = candidates.sample(n=1, random_state=rng.randint(0, 2**31)).iloc[0]
    window_idx = rng.randrange(chosen_file["n_windows"])

    return {
        "vibration_file": chosen_file["path"],
        "vibration_severity": chosen_file["severity"],
        "vibration_window_idx": window_idx,
    }


def build_manifest(config_path: str = "config/data_config.yaml", seed: int = 42) -> pd.DataFrame:
    config = load_config(config_path)
    project_root = find_project_root()
    mvtec_dir = project_root / config["paths"]["mvtec_dir"]
    cwru_dir = project_root / config["paths"]["cwru_dir"]

    rng = random.Random(seed)

    print("Indexing CWRU files...")
    cwru_inventory = build_cwru_inventory(cwru_dir, config, project_root)
    print(f"  {len(cwru_inventory)} CWRU files indexed "
          f"({(cwru_inventory.condition == 'normal').sum()} normal, "
          f"{(cwru_inventory.condition == 'fault').sum()} fault)")

    print("Indexing MVTec images...")
    mvtec_records = build_mvtec_records(mvtec_dir, config, project_root)
    print(f"  {len(mvtec_records)} MVTec images indexed")

    manifest_df = pd.DataFrame(mvtec_records)

    print("Assigning vibration samples...")
    vib_assignments = manifest_df.apply(
        lambda row: assign_vibration_sample(row, cwru_inventory, config, rng), axis=1
    )
    vib_df = pd.DataFrame(list(vib_assignments))
    manifest_df = pd.concat([manifest_df.reset_index(drop=True), vib_df], axis=1)

    fault_class_to_idx = {ft["name"]: i for i, ft in enumerate(config["cwru"]["fault_types"])}
    fault_class_to_idx["none"] = -1
    manifest_df["fault_class_idx"] = manifest_df["fault_class"].map(fault_class_to_idx)

    manifest_df["sample_id"] = [f"{i:05d}" for i in range(len(manifest_df))]

    return manifest_df


if __name__ == "__main__":
    config = load_config()
    df = build_manifest()

    project_root = find_project_root()
    out_dir = project_root / config["paths"]["manifest_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manifest.csv"
    df.to_csv(out_path, index=False)

    print(f"\nManifest written to {out_path} ({len(df)} rows)")
    print(df["is_defect"].value_counts())
    print(df["fault_class"].value_counts())