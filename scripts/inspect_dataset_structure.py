#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any


def load_yaml_or_json(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"PyYAML is required to parse {path}. Install with: pip install pyyaml"
            ) from exc
        try:
            return yaml.safe_load(path.read_text())
        except Exception:
            # OpenCV YAML fallback (e.g. %YAML:1.0 + !!opencv-matrix).
            import cv2

            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise

            out: dict[str, Any] = {}
            scalar_keys = ["image_width", "image_height", "fx", "fy", "cx", "cy", "num_images_used"]
            matrix_keys = ["K", "camera_matrix", "intrinsic_matrix", "intrinsics", "dist", "per_view_rms"]
            for key in scalar_keys:
                node = fs.getNode(key)
                if not node.empty():
                    out[key] = node.real()
            for key in matrix_keys:
                node = fs.getNode(key)
                if not node.empty():
                    mat = node.mat()
                    if mat is not None:
                        out[key] = mat.tolist()
            fs.release()
            return out
    raise ValueError(f"Unsupported config file extension: {path.suffix}")


def summarize_obj(obj: Any, depth: int = 0, max_depth: int = 3) -> str:
    indent = "  " * depth
    if depth > max_depth:
        return f"{indent}..."

    if isinstance(obj, dict):
        lines = [f"{indent}dict(keys={list(obj.keys())[:12]})"]
        for idx, (k, v) in enumerate(obj.items()):
            if idx >= 8:
                lines.append(f"{indent}  ...")
                break
            lines.append(f"{indent}- {k}:")
            lines.append(summarize_obj(v, depth + 1, max_depth))
        return "\n".join(lines)

    if isinstance(obj, list):
        lines = [f"{indent}list(len={len(obj)})"]
        for i, item in enumerate(obj[:5]):
            lines.append(f"{indent}- [{i}]")
            lines.append(summarize_obj(item, depth + 1, max_depth))
        if len(obj) > 5:
            lines.append(f"{indent}  ...")
        return "\n".join(lines)

    return f"{indent}{type(obj).__name__}: {obj}"


def print_tree(root: Path, max_depth: int = 3, max_entries_per_dir: int = 20) -> None:
    root = root.resolve()
    print(f"[dataset] root: {root}")

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return
        rel = path.relative_to(root)
        prefix = "  " * depth
        label = "." if str(rel) == "." else str(rel)
        print(f"{prefix}{label}/")

        entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        shown = 0

        for d in dirs:
            if shown >= max_entries_per_dir:
                print(f"{prefix}  ...")
                break
            walk(d, depth + 1)
            shown += 1

        for f in files:
            if shown >= max_entries_per_dir:
                print(f"{prefix}  ...")
                break
            print(f"{prefix}  {f.name}")
            shown += 1

    walk(root, 0)


def find_intrinsic_candidates(root: Path) -> list[Path]:
    patterns = [
        "*intrin*.yaml",
        "*intrin*.yml",
        "*intrin*.json",
        "*camera*.yaml",
        "*camera*.yml",
        "*camera*.json",
        "*calib*.yaml",
        "*calib*.yml",
        "*calib*.json",
        "meta/*.json",
    ]
    found = set()
    for pattern in patterns:
        for p in root.rglob(pattern):
            if p.is_file():
                found.add(p.resolve())
    results = sorted(found)
    filtered: list[Path] = []
    for p in results:
        if p.name.startswith("meta_") and p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text())
                if "rgb_intrinsic" in obj or "depth_intrinsic" in obj:
                    filtered.append(p)
            except Exception:
                pass
        else:
            filtered.append(p)
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset structure and intrinsic files.")
    parser.add_argument("--dataset-root", required=True, help="Path to extracted dataset root.")
    parser.add_argument(
        "--thermal-intrinsic",
        default="thermal_intrinsics.yaml",
        help="Path to thermal intrinsic file in project.",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    thermal_intrinsic = Path(args.thermal_intrinsic).resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")
    if not thermal_intrinsic.exists():
        raise FileNotFoundError(f"Thermal intrinsic file not found: {thermal_intrinsic}")

    print_tree(dataset_root, max_depth=3, max_entries_per_dir=30)

    candidates = find_intrinsic_candidates(dataset_root)
    print(f"\n[intrinsics] Found {len(candidates)} candidate intrinsic/calibration files in dataset:")
    for p in candidates[:40]:
        print(f"- {p}")
    if len(candidates) > 40:
        print(f"- ... ({len(candidates) - 40} more)")

    print("\n[intrinsics] Thermal intrinsic structure (project file):")
    thermal_obj = load_yaml_or_json(thermal_intrinsic)
    print(summarize_obj(thermal_obj, max_depth=4))

    if candidates:
        display_candidates = candidates[:20]
        print("\n[intrinsics] Dataset intrinsic structures (each file parsed separately):")
        for p in display_candidates:
            print(f"\n--- {p}")
            try:
                obj = load_yaml_or_json(p)
                print(summarize_obj(obj, max_depth=4))
            except Exception as exc:
                print(f"Could not parse: {exc}")
        if len(candidates) > len(display_candidates):
            print(f"\n[intrinsics] ... skipped {len(candidates) - len(display_candidates)} additional files")


if __name__ == "__main__":
    main()
