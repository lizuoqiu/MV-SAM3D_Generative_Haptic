#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d
from PIL import Image


def parse_matrix_3x3(data: Any) -> np.ndarray | None:
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None


def parse_intrinsics_file(path: Path) -> np.ndarray:
    text = path.read_text()
    obj: Any

    if path.suffix.lower() == ".json":
        obj = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        try:
            obj = yaml.safe_load(text)
        except Exception:
            # OpenCV YAML (e.g. %YAML:1.0 with !!opencv-matrix) fallback.
            import cv2

            fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            if not fs.isOpened():
                raise

            for key in ("K", "camera_matrix", "intrinsic_matrix", "intrinsics"):
                node = fs.getNode(key)
                if not node.empty():
                    mat = node.mat()
                    fs.release()
                    if mat is not None:
                        mat = np.asarray(mat, dtype=np.float64)
                        if mat.shape == (3, 3):
                            return mat
                        if mat.size == 9:
                            return mat.reshape(3, 3)

            fx = fs.getNode("fx").real()
            fy = fs.getNode("fy").real()
            cx = fs.getNode("cx").real()
            cy = fs.getNode("cy").real()
            fs.release()
            if all(v != 0 for v in (fx, fy)):
                return np.array(
                    [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
            raise
    else:
        raise ValueError(f"Unsupported intrinsic file format: {path}")

    if isinstance(obj, dict):
        # Case 1: direct fx, fy, cx, cy keys.
        if all(k in obj for k in ("fx", "fy", "cx", "cy")):
            fx, fy, cx, cy = float(obj["fx"]), float(obj["fy"]), float(obj["cx"]), float(obj["cy"])
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        # Case 2: ROS CameraInfo-like camera_matrix.data.
        if "camera_matrix" in obj:
            cm = obj["camera_matrix"]
            if isinstance(cm, dict) and "data" in cm:
                mat = parse_matrix_3x3(cm["data"])
                if mat is not None:
                    return mat
            mat = parse_matrix_3x3(cm)
            if mat is not None:
                return mat

        # Case 3: generic matrix names.
        for key in ("K", "intrinsic_matrix", "intrinsics"):
            if key in obj:
                mat = parse_matrix_3x3(obj[key])
                if mat is not None:
                    return mat

    raise ValueError(f"Could not parse intrinsics from file: {path}")


def parse_poses_json(path: Path) -> dict[str, np.ndarray]:
    obj = json.loads(path.read_text())
    poses: dict[str, np.ndarray] = {}

    def to_pose_matrix(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
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
        raise ValueError(f"Unsupported pose shape: {arr.shape}")

    if isinstance(obj, dict):
        # Mapping: image_name -> matrix
        if all(isinstance(v, (list, dict)) for v in obj.values()):
            for k, v in obj.items():
                if isinstance(v, dict):
                    for pose_key in ("world_to_camera", "T_wc", "extrinsic"):
                        if pose_key in v:
                            poses[k] = to_pose_matrix(v[pose_key])
                            break
                else:
                    poses[k] = to_pose_matrix(v)
            if poses:
                return poses

        # Frame list style.
        for list_key in ("frames", "views", "images"):
            if list_key in obj and isinstance(obj[list_key], list):
                for item in obj[list_key]:
                    name = item.get("image") or item.get("file_name") or item.get("name")
                    pose = item.get("world_to_camera") or item.get("T_wc") or item.get("extrinsic")
                    if name is not None and pose is not None:
                        poses[str(name)] = to_pose_matrix(pose)
                if poses:
                    return poses

    if isinstance(obj, list):
        for item in obj:
            name = item.get("image") or item.get("file_name") or item.get("name")
            pose = item.get("world_to_camera") or item.get("T_wc") or item.get("extrinsic")
            if name is not None and pose is not None:
                poses[str(name)] = to_pose_matrix(pose)
        if poses:
            return poses

    raise ValueError(f"Could not parse poses file: {path}")


def load_thermal_image(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float64)


def load_thermal_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    lines = path.read_text(errors="ignore").splitlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Axis Y\\X"):
            start_idx = i + 1
            break
    if start_idx is None:
        raise ValueError(f"Could not locate thermal grid header in CSV: {path}")

    for line in lines[start_idx:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            int(parts[0])  # row index
        except Exception:
            continue

        vals: list[float] = []
        for p in parts[1:]:
            if p == "":
                continue
            try:
                vals.append(float(p))
            except Exception:
                pass
        if vals:
            rows.append(vals)

    if not rows:
        raise ValueError(f"No thermal numeric rows parsed from CSV: {path}")

    width = min(len(r) for r in rows)
    arr = np.asarray([r[:width] for r in rows], dtype=np.float64)
    return arr


def colorize_temperature(values: np.ndarray, valid: np.ndarray, tmin: float | None, tmax: float | None) -> np.ndarray:
    out = np.zeros((values.shape[0], 3), dtype=np.float64)
    if valid.sum() == 0:
        return out

    vals = values[valid]
    lo = float(np.min(vals) if tmin is None else tmin)
    hi = float(np.max(vals) if tmax is None else tmax)
    if hi <= lo:
        hi = lo + 1e-6

    x = (np.clip(values, lo, hi) - lo) / (hi - lo)
    # Lightweight "jet-like" colormap without matplotlib.
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    out[:, 0] = r
    out[:, 1] = g
    out[:, 2] = b
    out[~valid] = 0.2  # gray for vertices with no thermal data
    return out


def project_vertices(K: np.ndarray, T_wc: np.ndarray, verts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ones = np.ones((verts_world.shape[0], 1), dtype=np.float64)
    verts_h = np.concatenate([verts_world, ones], axis=1)
    cam = (T_wc @ verts_h.T).T[:, :3]
    z = cam[:, 2]
    valid_z = z > 1e-8

    uv = np.zeros((verts_world.shape[0], 2), dtype=np.float64)
    uv[valid_z, 0] = (K[0, 0] * cam[valid_z, 0] / z[valid_z]) + K[0, 2]
    uv[valid_z, 1] = (K[1, 1] * cam[valid_z, 1] / z[valid_z]) + K[1, 2]
    return uv, z, valid_z


def visible_vertex_indices_per_view(uv: np.ndarray, z: np.ndarray, valid_z: np.ndarray, h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.round(uv[:, 0]).astype(np.int64)
    v = np.round(uv[:, 1]).astype(np.int64)
    valid = valid_z & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if valid.sum() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    idx = np.where(valid)[0]
    uu = u[idx]
    vv = v[idx]
    zz = z[idx]
    pix = vv * w + uu

    # Z-buffer by sorting on (pix, depth) and picking the nearest per pixel.
    order = np.lexsort((zz, pix))
    pix_sorted = pix[order]
    first = np.ones_like(pix_sorted, dtype=bool)
    first[1:] = pix_sorted[1:] != pix_sorted[:-1]
    keep = order[first]

    return idx[keep], uu[keep], vv[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Map multi-view thermal values onto a reconstructed mesh.")
    parser.add_argument("--mesh", required=True, help="Input reconstructed mesh (ply/obj/glb).")
    parser.add_argument("--thermal-dir", required=True, help="Directory with thermal images.")
    parser.add_argument("--thermal-intrinsics", required=True, help="Thermal camera intrinsic file (.yaml/.yml/.json).")
    parser.add_argument("--thermal-poses", required=True, help="JSON file with thermal world_to_camera poses per image.")
    parser.add_argument(
        "--thermal-source",
        choices=["csv", "image"],
        default="csv",
        help="Use raw thermal CSV values or thermal image intensities.",
    )
    parser.add_argument("--aggregation", choices=["avg", "max"], default="avg")
    parser.add_argument("--output-prefix", required=True, help="Output prefix path (without extension).")
    parser.add_argument("--temp-min", type=float, default=None, help="Optional fixed colormap min.")
    parser.add_argument("--temp-max", type=float, default=None, help="Optional fixed colormap max.")
    parser.add_argument("--pose-image-suffix", default="", help="Optional suffix appended to pose keys when matching images.")
    args = parser.parse_args()

    mesh_path = Path(args.mesh).resolve()
    thermal_dir = Path(args.thermal_dir).resolve()
    thermal_intrinsic_path = Path(args.thermal_intrinsics).resolve()
    thermal_poses_path = Path(args.thermal_poses).resolve()
    output_prefix = Path(args.output_prefix).resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)
    if not thermal_dir.exists():
        raise FileNotFoundError(thermal_dir)
    if not thermal_intrinsic_path.exists():
        raise FileNotFoundError(thermal_intrinsic_path)
    if not thermal_poses_path.exists():
        raise FileNotFoundError(thermal_poses_path)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {mesh_path}")
    if not mesh.has_vertex_colors():
        # Preserve an RGB mesh output even if original has no colors.
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.full((len(mesh.vertices), 3), 0.7, dtype=np.float64))

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    orig_colors = np.asarray(mesh.vertex_colors, dtype=np.float64).copy()

    K = parse_intrinsics_file(thermal_intrinsic_path)
    poses = parse_poses_json(thermal_poses_path)

    if args.aggregation == "avg":
        agg_sum = np.zeros((verts.shape[0],), dtype=np.float64)
        agg_cnt = np.zeros((verts.shape[0],), dtype=np.int32)
    else:
        agg_max = np.full((verts.shape[0],), -np.inf, dtype=np.float64)
        agg_seen = np.zeros((verts.shape[0],), dtype=bool)

    if args.thermal_source == "csv":
        thermal_files = sorted([p for p in thermal_dir.iterdir() if p.suffix.lower() == ".csv"])
    else:
        thermal_files = sorted(
            [p for p in thermal_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
        )
    if not thermal_files:
        raise RuntimeError(f"No thermal images found in {thermal_dir}")

    used_views = 0
    for tpath in thermal_files:
        pose_key_candidates = [tpath.name, tpath.stem]
        if args.thermal_source == "csv":
            pose_key_candidates.extend(
                [f"{tpath.stem}.jpeg", f"{tpath.stem}.jpg", f"{tpath.stem}.png", f"{tpath.stem}.csv"]
            )
        if args.pose_image_suffix:
            pose_key_candidates.insert(0, f"{tpath.stem}{args.pose_image_suffix}")

        T_wc = None
        for k in pose_key_candidates:
            if k in poses:
                T_wc = poses[k]
                break
        if T_wc is None:
            print(f"[thermal] skip (pose missing): {tpath.name}")
            continue

        img = load_thermal_csv(tpath) if args.thermal_source == "csv" else load_thermal_image(tpath)
        h, w = img.shape[:2]

        uv, z, valid_z = project_vertices(K, T_wc, verts)
        vidx, uu, vv = visible_vertex_indices_per_view(uv, z, valid_z, h, w)
        if vidx.size == 0:
            print(f"[thermal] skip (no visible vertices): {tpath.name}")
            continue

        vals = img[vv, uu]

        if args.aggregation == "avg":
            np.add.at(agg_sum, vidx, vals)
            np.add.at(agg_cnt, vidx, 1)
        else:
            prev = agg_max[vidx]
            agg_max[vidx] = np.maximum(prev, vals)
            agg_seen[vidx] = True

        used_views += 1
        print(f"[thermal] view={tpath.name} visible_vertices={vidx.size}")

    if used_views == 0:
        raise RuntimeError("No thermal views were matched with poses.")

    if args.aggregation == "avg":
        valid = agg_cnt > 0
        temp = np.zeros((verts.shape[0],), dtype=np.float64)
        temp[valid] = agg_sum[valid] / agg_cnt[valid]
    else:
        valid = agg_seen
        temp = np.zeros((verts.shape[0],), dtype=np.float64)
        temp[valid] = agg_max[valid]

    thermal_colors = colorize_temperature(temp, valid, args.temp_min, args.temp_max)

    rgb_mesh = o3d.geometry.TriangleMesh(mesh)
    rgb_mesh.vertex_colors = o3d.utility.Vector3dVector(orig_colors)

    thermal_mesh = o3d.geometry.TriangleMesh(mesh)
    thermal_mesh.vertex_colors = o3d.utility.Vector3dVector(thermal_colors)

    rgb_out = output_prefix.with_name(output_prefix.name + "_rgb.ply")
    thermal_out = output_prefix.with_name(output_prefix.name + f"_thermal_{args.aggregation}.ply")
    temp_out = output_prefix.with_name(output_prefix.name + f"_temperature_{args.aggregation}.npy")
    meta_out = output_prefix.with_name(output_prefix.name + f"_temperature_{args.aggregation}.json")

    o3d.io.write_triangle_mesh(str(rgb_out), rgb_mesh)
    o3d.io.write_triangle_mesh(str(thermal_out), thermal_mesh)
    np.save(str(temp_out), temp)

    summary = {
        "mesh": str(mesh_path),
        "thermal_dir": str(thermal_dir),
        "thermal_intrinsics": str(thermal_intrinsic_path),
        "thermal_poses": str(thermal_poses_path),
        "aggregation": args.aggregation,
        "used_views": used_views,
        "num_vertices": int(verts.shape[0]),
        "num_vertices_with_temperature": int(valid.sum()),
        "coverage_ratio": float(valid.mean()),
        "temperature_min": float(temp[valid].min()) if valid.any() else None,
        "temperature_max": float(temp[valid].max()) if valid.any() else None,
        "outputs": {
            "rgb_mesh": str(rgb_out),
            "thermal_mesh": str(thermal_out),
            "temperature_npy": str(temp_out),
        },
    }
    meta_out.write_text(json.dumps(summary, indent=2))

    print("[thermal] Mapping complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
