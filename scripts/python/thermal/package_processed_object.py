#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path


def safe_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src, target_is_directory=src.is_dir())


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Package one processed object into processed_dataset layout.")
    parser.add_argument("--object-dir", required=True)
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--processed-root", required=True)
    parser.add_argument("--thermal-prefix", required=True, help="Prefix used by run_thermal_mapping.sh")
    parser.add_argument("--thermal-poses", required=True)
    parser.add_argument("--da3-output", required=True)
    parser.add_argument("--summary-image", required=True)
    args = parser.parse_args()

    obj_dir = Path(args.object_dir).resolve()
    recon_dir = Path(args.reconstruction_dir).resolve()
    processed_root = Path(args.processed_root).resolve()
    thermal_prefix = Path(args.thermal_prefix).resolve()
    thermal_poses = Path(args.thermal_poses).resolve()
    da3_output = Path(args.da3_output).resolve()
    summary_image = Path(args.summary_image).resolve()

    category = obj_dir.parent.name
    object_name = obj_dir.name
    out = processed_root / category / object_name

    src_dir = out / "source"
    rec_rgb_dir = out / "reconstruction" / "rgb_result"
    rec_th_dir = out / "reconstruction" / "thermal_result"
    rec_pose_dir = out / "reconstruction" / "poses"
    verify_dir = out / "verification"

    for d in (src_dir, rec_rgb_dir, rec_th_dir, rec_pose_dir, verify_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Keep raw data linked to avoid large duplication.
    safe_symlink(obj_dir / "rgb", src_dir / "rgb")
    safe_symlink(obj_dir / "thermal", src_dir / "thermal")
    safe_symlink(obj_dir / "meta", src_dir / "meta")
    safe_symlink(obj_dir / "depth", src_dir / "depth")
    copy_if_exists(obj_dir / "view_mapping.json", src_dir / "view_mapping.json")

    # RGB reconstruction artifacts.
    copy_if_exists(recon_dir / "result.glb", rec_rgb_dir / "result.glb")
    copy_if_exists(recon_dir / "result.ply", rec_rgb_dir / "result.ply")
    copy_if_exists(recon_dir / "params.npz", rec_rgb_dir / "params.npz")
    copy_if_exists(recon_dir / "inference.log", rec_rgb_dir / "inference.log")

    # Thermal mapping artifacts.
    thermal_files = [
        f"{thermal_prefix.name}_rgb.ply",
        f"{thermal_prefix.name}_thermal_avg.ply",
        f"{thermal_prefix.name}_thermal_max.ply",
        f"{thermal_prefix.name}_temperature_avg.npy",
        f"{thermal_prefix.name}_temperature_max.npy",
        f"{thermal_prefix.name}_temperature_avg.json",
        f"{thermal_prefix.name}_temperature_max.json",
        f"{thermal_prefix.name}_verify_avg.png",
        f"{thermal_prefix.name}_verify_max.png",
    ]
    for name in thermal_files:
        copy_if_exists(thermal_prefix.with_name(name), rec_th_dir / name)

    copy_if_exists(thermal_poses, rec_pose_dir / "thermal_poses.json")
    # DA3 output can be large; symlink to avoid duplication.
    safe_symlink(da3_output, rec_pose_dir / "da3_output.npz")
    copy_if_exists(summary_image, verify_dir / "task2_summary.png")

    manifest = {
        "object_dir": str(obj_dir),
        "reconstruction_dir": str(recon_dir),
        "packaged_at": str(out),
        "category": category,
        "object_name": object_name,
        "layout": {
            "source": str(src_dir),
            "rgb_result": str(rec_rgb_dir),
            "thermal_result": str(rec_th_dir),
            "poses": str(rec_pose_dir),
            "verification": str(verify_dir),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[package] saved: {out}")


if __name__ == "__main__":
    main()
