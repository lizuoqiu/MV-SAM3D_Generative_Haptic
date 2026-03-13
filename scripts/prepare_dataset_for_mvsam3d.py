#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def numeric_tail(name: str) -> int:
    nums = re.findall(r"\d+", name)
    if not nums:
        return -1
    return int(nums[-1])


def rgb_sort_key(path: Path):
    return (numeric_tail(path.stem), path.stem)


def thermal_sort_key(path: Path):
    return (numeric_tail(path.stem), path.stem)


def find_object_dirs(root: Path) -> list[Path]:
    out: list[Path] = []
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        subs = {p.name for p in d.iterdir() if p.is_dir()}
        if {"rgb", "thermal", "depth", "meta"}.issubset(subs):
            out.append(d)
    return sorted(out)


def ensure_images_link(obj_dir: Path) -> str:
    rgb_dir = obj_dir / "rgb"
    images_dir = obj_dir / "images"
    if images_dir.exists():
        return "exists"

    try:
        images_dir.symlink_to(rgb_dir, target_is_directory=True)
        return "symlink"
    except Exception:
        images_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(rgb_dir.iterdir(), key=rgb_sort_key):
            if f.is_file():
                target = images_dir / f.name
                if not target.exists():
                    target.hardlink_to(f)
        return "hardlink_copy"


def write_view_mapping(obj_dir: Path) -> Path:
    rgb_dir = obj_dir / "rgb"
    thermal_dir = obj_dir / "thermal"
    meta_dir = obj_dir / "meta"
    depth_dir = obj_dir / "depth"

    rgb_files = sorted(
        [p for p in rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=rgb_sort_key,
    )
    thermal_img_files = sorted(
        [p for p in thermal_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=thermal_sort_key,
    )
    thermal_csv_files = sorted([p for p in thermal_dir.iterdir() if p.suffix.lower() == ".csv"], key=thermal_sort_key)
    meta_files = sorted([p for p in meta_dir.iterdir() if p.suffix.lower() == ".json"], key=rgb_sort_key)
    depth_files = sorted([p for p in depth_dir.iterdir() if p.suffix.lower() == ".npy"], key=rgb_sort_key)

    n = min(len(rgb_files), len(thermal_img_files), len(thermal_csv_files), len(meta_files), len(depth_files))
    rows = []
    for i in range(n):
        rows.append(
            {
                "view_index": i,
                "rgb_image": rgb_files[i].name,
                "meta_json": meta_files[i].name,
                "depth_npy": depth_files[i].name,
                "thermal_image": thermal_img_files[i].name,
                "thermal_csv": thermal_csv_files[i].name,
            }
        )

    out = obj_dir / "view_mapping.json"
    out.write_text(json.dumps({"num_views": n, "pairs": rows}, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset layout for MV-SAM3D and write RGB/thermal mapping files.")
    parser.add_argument("--dataset-root", required=True)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    object_dirs = find_object_dirs(root)
    print(f"[prepare] object_dirs={len(object_dirs)}")

    for d in object_dirs:
        mode = ensure_images_link(d)
        mapping_path = write_view_mapping(d)
        print(f"[prepare] {d} images={mode} mapping={mapping_path.name}")


if __name__ == "__main__":
    main()
