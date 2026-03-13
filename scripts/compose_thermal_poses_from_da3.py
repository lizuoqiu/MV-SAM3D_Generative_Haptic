#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def to_4x4(matrix_like: Any) -> np.ndarray:
    arr = np.asarray(matrix_like, dtype=np.float64)
    if arr.shape == (4, 4):
        return arr
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        return out
    if arr.size == 16:
        return arr.reshape(4, 4)
    if arr.size == 12:
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr.reshape(3, 4)
        return out
    raise ValueError(f"Unsupported matrix shape/size: {arr.shape} / {arr.size}")


def load_rgb_to_thermal_transform(path: Path | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)

    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        if "T_thermal_from_rgb" in obj:
            return to_4x4(obj["T_thermal_from_rgb"])
        if "transform" in obj:
            return to_4x4(obj["transform"])
    return to_4x4(obj)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose per-view thermal world_to_camera poses from DA3 RGB extrinsics and "
            "an optional fixed RGB->thermal transform."
        )
    )
    parser.add_argument("--da3-output", required=True, help="Path to da3_output.npz")
    parser.add_argument("--view-mapping", required=True, help="Path to view_mapping.json")
    parser.add_argument("--output-json", required=True, help="Output thermal poses JSON")
    parser.add_argument(
        "--rgb-to-thermal-transform",
        default=None,
        help=(
            "Optional JSON path with 4x4 transform T_thermal_from_rgb. "
            "If omitted, identity is used."
        ),
    )
    args = parser.parse_args()

    da3_path = Path(args.da3_output).resolve()
    mapping_path = Path(args.view_mapping).resolve()
    out_path = Path(args.output_json).resolve()
    tf_path = Path(args.rgb_to_thermal_transform).resolve() if args.rgb_to_thermal_transform else None

    if not da3_path.exists():
        raise FileNotFoundError(da3_path)
    if not mapping_path.exists():
        raise FileNotFoundError(mapping_path)
    if tf_path is not None and not tf_path.exists():
        raise FileNotFoundError(tf_path)

    da3 = np.load(str(da3_path), allow_pickle=True)
    if "extrinsics" not in da3 or "image_files" not in da3:
        raise ValueError("da3_output.npz must contain 'extrinsics' and 'image_files'.")

    extrinsics = da3["extrinsics"]
    image_files = da3["image_files"]
    if extrinsics.ndim != 3:
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")

    image_name_to_idx: dict[str, int] = {}
    image_stem_to_idx: dict[str, int] = {}
    for i, p in enumerate(image_files):
        name = Path(str(p)).name
        stem = Path(name).stem
        image_name_to_idx[name] = i
        image_stem_to_idx[stem] = i

    mapping = json.loads(mapping_path.read_text())
    pairs = mapping.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("view_mapping.json format invalid: missing 'pairs' list.")

    T_thermal_from_rgb = load_rgb_to_thermal_transform(tf_path)

    thermal_poses: dict[str, list[list[float]]] = {}
    matched = 0
    for row in pairs:
        rgb_name = row.get("rgb_image")
        thermal_img = row.get("thermal_image")
        thermal_csv = row.get("thermal_csv")
        if not rgb_name or not thermal_img or not thermal_csv:
            continue

        idx = image_name_to_idx.get(rgb_name)
        if idx is None:
            idx = image_stem_to_idx.get(Path(rgb_name).stem)
        if idx is None:
            continue

        T_rgb = to_4x4(extrinsics[idx])
        T_thermal = T_thermal_from_rgb @ T_rgb

        # Write multiple keys so downstream matching is resilient.
        keys = {
            thermal_img,
            thermal_csv,
            Path(thermal_img).stem,
            Path(thermal_csv).stem,
        }
        for k in keys:
            thermal_poses[k] = T_thermal.tolist()

        matched += 1

    if matched == 0:
        raise RuntimeError("No view mappings matched between DA3 image files and view_mapping.json RGB names.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(thermal_poses, indent=2))

    print(
        json.dumps(
            {
                "output_json": str(out_path),
                "matched_views": matched,
                "total_pairs": len(pairs),
                "transform_source": str(tf_path) if tf_path else "identity",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
