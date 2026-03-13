#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path


def find_object_dirs(dataset_root: Path) -> list[Path]:
    out: list[Path] = []
    for p in dataset_root.rglob("images"):
        # include real directories and symlinked images folders
        if not (p.is_dir() or p.is_symlink()):
            continue
        obj = p.parent
        if (obj / "thermal").is_dir() and (obj / "view_mapping.json").is_file():
            out.append(obj)
    # stable order
    return sorted(set(out))


def find_latest_recon_dir(visualization_root: Path, object_name: str, mask_name: str) -> Path | None:
    base = visualization_root / object_name / mask_name
    if not base.is_dir():
        return None
    candidates = [d for d in base.iterdir() if d.is_dir() and (d / "result.ply").is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Task-2 thermal mapping for all reconstructed objects.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--visualization-root", default="visualization")
    parser.add_argument("--thermal-intrinsics", required=True)
    parser.add_argument("--processed-root", default="processed_dataset")
    parser.add_argument("--mask-name", default="sam2_masks")
    parser.add_argument("--rgb-to-thermal-transform", default=None)
    parser.add_argument("--exclude-object", action="append", default=[])
    parser.add_argument("--report-json", default="processed_dataset/task2_batch_report.json")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    visualization_root = Path(args.visualization_root).resolve()
    thermal_intrinsics = Path(args.thermal_intrinsics).resolve()
    processed_root = Path(args.processed_root).resolve()
    report_json = Path(args.report_json).resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not visualization_root.is_dir():
        raise FileNotFoundError(visualization_root)
    if not thermal_intrinsics.is_file():
        raise FileNotFoundError(thermal_intrinsics)

    exclude = {str(Path(x).resolve()) for x in args.exclude_object}
    object_dirs = find_object_dirs(dataset_root)

    results: list[dict] = []
    for obj in object_dirs:
        obj_key = str(obj.resolve())
        if obj_key in exclude:
            results.append({"object_dir": obj_key, "status": "skipped_excluded"})
            print(f"[task2-batch] skip excluded: {obj}")
            continue

        recon = find_latest_recon_dir(visualization_root, obj.name, args.mask_name)
        if recon is None:
            results.append({"object_dir": obj_key, "status": "skipped_no_reconstruction"})
            print(f"[task2-batch] skip no reconstruction: {obj}")
            continue

        cmd = [
            "bash",
            "scripts/run_task2_for_object.sh",
            str(obj),
            str(recon),
            str(thermal_intrinsics),
            str(processed_root),
        ]
        if args.rgb_to_thermal_transform:
            cmd.append(str(Path(args.rgb_to_thermal_transform).resolve()))

        print(f"[task2-batch] run: {obj}")
        proc = subprocess.run(cmd)
        status = "ok" if proc.returncode == 0 else "failed"
        results.append(
            {
                "object_dir": obj_key,
                "reconstruction_dir": str(recon),
                "status": status,
                "returncode": int(proc.returncode),
            }
        )

    report_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_root": str(dataset_root),
        "visualization_root": str(visualization_root),
        "thermal_intrinsics": str(thermal_intrinsics),
        "processed_root": str(processed_root),
        "mask_name": args.mask_name,
        "num_total_objects": len(object_dirs),
        "num_ok": sum(1 for r in results if r["status"] == "ok"),
        "num_failed": sum(1 for r in results if r["status"] == "failed"),
        "num_skipped": sum(1 for r in results if r["status"].startswith("skipped_")),
        "results": results,
    }
    report_json.write_text(json.dumps(summary, indent=2))
    print(f"[task2-batch] report: {report_json}")

    if summary["num_failed"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
