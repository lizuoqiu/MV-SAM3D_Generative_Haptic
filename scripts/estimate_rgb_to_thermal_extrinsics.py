#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_matrix_3x3(data: Any) -> np.ndarray | None:
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None


def parse_intrinsics(path: Path) -> np.ndarray:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        try:
            obj = yaml.safe_load(text)
        except Exception:
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
        raise ValueError(f"Unsupported intrinsic file extension: {path}")

    if isinstance(obj, dict):
        if all(k in obj for k in ("fx", "fy", "cx", "cy")):
            fx, fy, cx, cy = float(obj["fx"]), float(obj["fy"]), float(obj["cx"]), float(obj["cy"])
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        if "camera_matrix" in obj:
            cm = obj["camera_matrix"]
            if isinstance(cm, dict) and "data" in cm:
                m = parse_matrix_3x3(cm["data"])
                if m is not None:
                    return m
            m = parse_matrix_3x3(cm)
            if m is not None:
                return m

        for key in ("K", "intrinsic_matrix", "intrinsics"):
            if key in obj:
                m = parse_matrix_3x3(obj[key])
                if m is not None:
                    return m

    raise ValueError(f"Could not parse intrinsics from {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate rigid transform from RGB camera coordinates to thermal camera coordinates."
    )
    parser.add_argument("--rgb-intrinsics", required=True)
    parser.add_argument("--thermal-intrinsics", required=True)
    parser.add_argument(
        "--correspondences",
        required=True,
        help=(
            "JSON list of points. Each item needs thermal_uv [u,v] and either "
            "rgb_xyz [x,y,z] or (rgb_uv [u,v] + depth)."
        ),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--use-ransac", action="store_true", help="Use solvePnPRansac instead of solvePnP.")
    args = parser.parse_args()

    K_rgb = parse_intrinsics(Path(args.rgb_intrinsics).resolve())
    K_th = parse_intrinsics(Path(args.thermal_intrinsics).resolve())

    corr = json.loads(Path(args.correspondences).read_text())
    if not isinstance(corr, list) or len(corr) < 6:
        raise ValueError("Need at least 6 correspondences.")

    obj_pts = []
    img_pts = []

    fx, fy, cx, cy = K_rgb[0, 0], K_rgb[1, 1], K_rgb[0, 2], K_rgb[1, 2]

    for item in corr:
        th_uv = item.get("thermal_uv")
        if th_uv is None:
            continue

        if "rgb_xyz" in item:
            xyz = np.asarray(item["rgb_xyz"], dtype=np.float64)
            if xyz.shape != (3,):
                continue
        else:
            rgb_uv = item.get("rgb_uv")
            depth = item.get("depth")
            if rgb_uv is None or depth is None:
                continue
            u, v = float(rgb_uv[0]), float(rgb_uv[1])
            z = float(depth)
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            xyz = np.array([x, y, z], dtype=np.float64)

        obj_pts.append(xyz)
        img_pts.append(np.asarray(th_uv, dtype=np.float64))

    if len(obj_pts) < 6:
        raise ValueError(f"Valid correspondences after parsing: {len(obj_pts)} (need >= 6).")

    obj = np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3)
    img = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
    dist = np.zeros((5, 1), dtype=np.float64)

    if args.use_ransac:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=obj,
            imagePoints=img,
            cameraMatrix=K_th,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    else:
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=obj,
            imagePoints=img,
            cameraMatrix=K_th,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        inliers = None

    if not ok:
        raise RuntimeError("solvePnP failed.")

    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)

    # Reprojection error report.
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K_th, dist)
    proj = proj.reshape(-1, 2)
    err = np.linalg.norm(proj - img.reshape(-1, 2), axis=1)

    output = {
        "num_correspondences": int(len(obj_pts)),
        "num_inliers": int(len(inliers)) if inliers is not None else None,
        "mean_reprojection_error_px": float(err.mean()),
        "median_reprojection_error_px": float(np.median(err)),
        "T_thermal_from_rgb": T.tolist(),
        "R_thermal_from_rgb": R.tolist(),
        "t_thermal_from_rgb": tvec.reshape(3).tolist(),
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
