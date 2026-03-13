#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def natural_sort_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    try:
        return (0, int(stem), stem)
    except ValueError:
        return (1, 0, stem)


def parse_matrix_3x3(data: Any) -> np.ndarray | None:
    arr = np.asarray(data, dtype=np.float64)
    if arr.shape == (3, 3):
        return arr
    if arr.size == 9:
        return arr.reshape(3, 3)
    return None


@dataclass
class CameraIntrinsics:
    K: np.ndarray
    dist: np.ndarray


def parse_intrinsics_file(path: Path) -> CameraIntrinsics:
    text = path.read_text()
    obj: Any

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

            K = None
            for key in ("K", "camera_matrix", "intrinsic_matrix", "intrinsics"):
                node = fs.getNode(key)
                if not node.empty():
                    mat = node.mat()
                    if mat is not None:
                        mat = np.asarray(mat, dtype=np.float64)
                        if mat.shape == (3, 3):
                            K = mat
                            break
                        if mat.size == 9:
                            K = mat.reshape(3, 3)
                            break

            if K is None:
                fx = fs.getNode("fx").real()
                fy = fs.getNode("fy").real()
                cx = fs.getNode("cx").real()
                cy = fs.getNode("cy").real()
                if all(v != 0 for v in (fx, fy)):
                    K = np.array(
                        [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )

            if K is None:
                fs.release()
                raise ValueError(f"Could not parse K from {path}")

            dist = None
            for key in ("dist", "distortion", "dist_coeffs", "distCoeffs"):
                node = fs.getNode(key)
                if not node.empty():
                    d = node.mat()
                    if d is not None:
                        dist = np.asarray(d, dtype=np.float64).reshape(-1, 1)
                        break
            fs.release()
            if dist is None:
                dist = np.zeros((5, 1), dtype=np.float64)
            return CameraIntrinsics(K=K, dist=dist)
    else:
        raise ValueError(f"Unsupported intrinsics format: {path}")

    if isinstance(obj, dict):
        K = None
        for key in ("K", "intrinsic_matrix", "intrinsics"):
            if key in obj:
                K = parse_matrix_3x3(obj[key])
                if K is not None:
                    break
        if K is None and "camera_matrix" in obj:
            cm = obj["camera_matrix"]
            if isinstance(cm, dict) and "data" in cm:
                K = parse_matrix_3x3(cm["data"])
            if K is None:
                K = parse_matrix_3x3(cm)
        if K is None and all(k in obj for k in ("fx", "fy", "cx", "cy")):
            fx, fy, cx, cy = float(obj["fx"]), float(obj["fy"]), float(obj["cx"]), float(obj["cy"])
            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        if K is None:
            raise ValueError(f"Could not parse K from {path}")

        dist = None
        for key in ("dist", "distortion", "dist_coeffs", "distCoeffs"):
            if key in obj:
                raw = obj[key]
                if isinstance(raw, dict) and "data" in raw:
                    raw = raw["data"]
                try:
                    dist = np.asarray(raw, dtype=np.float64).reshape(-1, 1)
                    break
                except Exception:
                    pass
        if dist is None:
            if "distortion" in obj and isinstance(obj["distortion"], dict):
                d = obj["distortion"]
                vals = [d.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")]
                dist = np.asarray(vals, dtype=np.float64).reshape(-1, 1)
            else:
                dist = np.zeros((5, 1), dtype=np.float64)
        return CameraIntrinsics(K=K, dist=dist)

    raise ValueError(f"Could not parse intrinsics from {path}")


def find_csv_header_line(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.startswith("Axis Y\\X"):
            return i + 1
    return -1


def load_thermal_csv(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    lines = path.read_text(errors="ignore").splitlines()
    start = find_csv_header_line(lines)
    if start < 0:
        raise ValueError(f"Could not parse thermal csv: {path}")

    for line in lines[start:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            int(parts[0])
        except Exception:
            continue

        vals = []
        for p in parts[1:]:
            if not p:
                continue
            try:
                vals.append(float(p))
            except Exception:
                pass
        if vals:
            rows.append(vals)

    if not rows:
        raise ValueError(f"No numeric thermal rows in csv: {path}")

    w = min(len(r) for r in rows)
    arr = np.asarray([r[:w] for r in rows], dtype=np.float64)
    return arr


def normalize_to_uint8(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.uint8)
    lo = float(np.percentile(x[finite], 1.0))
    hi = float(np.percentile(x[finite], 99.0))
    if hi <= lo:
        lo = float(np.min(x[finite]))
        hi = float(np.max(x[finite]))
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (y * 255.0).astype(np.uint8)


def load_gray_image(path: Path, thermal_csv: bool = False) -> np.ndarray:
    if thermal_csv:
        arr = load_thermal_csv(path)
        return normalize_to_uint8(arr)

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_corners(gray: np.ndarray, pattern_size: tuple[int, int], use_sb: bool) -> tuple[bool, np.ndarray | None]:
    flags_std = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE

    if use_sb:
        sb_flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=sb_flags)
        if ok:
            return True, corners.astype(np.float32)

    ok, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags_std)
    if not ok or corners is None:
        return False, None

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    corners = cv2.cornerSubPix(gray, corners, winSize=(5, 5), zeroZone=(-1, -1), criteria=crit)
    return True, corners.astype(np.float32)


def create_object_points(cols: int, rows: int, square_size: float) -> np.ndarray:
    obj = np.zeros((rows * cols, 3), dtype=np.float32)
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj[:, :2] = grid * float(square_size)
    return obj


def read_pairs_from_json(path: Path) -> list[tuple[Path, Path]]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, list):
        raise ValueError("pairs-json must be a list")
    out = []
    for row in obj:
        if not isinstance(row, dict):
            continue
        rgb = row.get("rgb")
        thermal = row.get("thermal")
        if rgb and thermal:
            out.append((Path(str(rgb)), Path(str(thermal))))
    return out


def auto_pair_files(rgb_dir: Path, thermal_dir: Path, rgb_glob: str, thermal_glob: str) -> list[tuple[Path, Path]]:
    rgb_files = sorted(rgb_dir.glob(rgb_glob), key=natural_sort_key)
    thermal_files = sorted(thermal_dir.glob(thermal_glob), key=natural_sort_key)
    n = min(len(rgb_files), len(thermal_files))
    return [(rgb_files[i], thermal_files[i]) for i in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standard OpenCV checkerboard stereo calibration for RGB-thermal camera pair."
    )
    parser.add_argument("--rgb-dir", required=True, help="RGB calibration image directory.")
    parser.add_argument("--thermal-dir", required=True, help="Thermal calibration image directory.")
    parser.add_argument("--board-cols", type=int, required=True, help="Inner-corner columns of checkerboard.")
    parser.add_argument("--board-rows", type=int, required=True, help="Inner-corner rows of checkerboard.")
    parser.add_argument("--square-size", type=float, required=True, help="Checkerboard square size in meters.")
    parser.add_argument("--output-json", required=True, help="Output calibration json path.")
    parser.add_argument("--pairs-json", default=None, help="Optional explicit pair list json [{rgb,thermal}, ...].")
    parser.add_argument("--rgb-glob", default="*.jpg", help="Glob for RGB files when auto pairing.")
    parser.add_argument("--thermal-glob", default="*.jpeg", help="Glob for thermal files when auto pairing.")
    parser.add_argument(
        "--thermal-csv",
        action="store_true",
        help="Set if thermal inputs are CSV grids instead of image files.",
    )
    parser.add_argument("--rgb-intrinsics", default=None, help="Optional RGB intrinsics file (json/yaml).")
    parser.add_argument("--thermal-intrinsics", default=None, help="Optional thermal intrinsics file (json/yaml).")
    parser.add_argument(
        "--fix-intrinsics",
        action="store_true",
        help="Fix intrinsics during stereoCalibrate (recommended when intrinsics are known).",
    )
    parser.add_argument("--use-sb", action="store_true", help="Use findChessboardCornersSB first.")
    parser.add_argument("--min-valid-pairs", type=int, default=10, help="Minimum usable pairs required.")
    parser.add_argument("--save-debug-dir", default=None, help="Optional directory to save corner debug images.")
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir).resolve()
    thermal_dir = Path(args.thermal_dir).resolve()
    out_path = Path(args.output_json).resolve()
    debug_dir = Path(args.save_debug_dir).resolve() if args.save_debug_dir else None
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_dir.exists() or not thermal_dir.exists():
        raise FileNotFoundError("rgb-dir or thermal-dir does not exist.")

    if args.pairs_json:
        raw_pairs = read_pairs_from_json(Path(args.pairs_json).resolve())
        pairs = []
        for rgb_rel, th_rel in raw_pairs:
            rgb_path = rgb_rel if rgb_rel.is_absolute() else rgb_dir / rgb_rel
            th_path = th_rel if th_rel.is_absolute() else thermal_dir / th_rel
            pairs.append((rgb_path, th_path))
    else:
        pairs = auto_pair_files(rgb_dir, thermal_dir, args.rgb_glob, args.thermal_glob)

    if not pairs:
        raise RuntimeError("No image pairs found.")

    pattern_size = (int(args.board_cols), int(args.board_rows))
    obj_template = create_object_points(args.board_cols, args.board_rows, args.square_size)

    object_points: list[np.ndarray] = []
    rgb_points: list[np.ndarray] = []
    thermal_points: list[np.ndarray] = []
    used_pairs: list[dict[str, str]] = []
    skipped_pairs: list[dict[str, str]] = []

    rgb_size = None
    thermal_size = None

    for i, (rgb_path, th_path) in enumerate(pairs):
        if not rgb_path.exists() or not th_path.exists():
            skipped_pairs.append({"rgb": str(rgb_path), "thermal": str(th_path), "reason": "missing_file"})
            continue

        rgb_gray = load_gray_image(rgb_path, thermal_csv=False)
        th_gray = load_gray_image(th_path, thermal_csv=args.thermal_csv)

        ok_rgb, c_rgb = detect_corners(rgb_gray, pattern_size, use_sb=args.use_sb)
        ok_th, c_th = detect_corners(th_gray, pattern_size, use_sb=args.use_sb)
        if not ok_rgb or not ok_th or c_rgb is None or c_th is None:
            skipped_pairs.append(
                {
                    "rgb": str(rgb_path),
                    "thermal": str(th_path),
                    "reason": "chessboard_not_detected",
                    "ok_rgb": bool(ok_rgb),
                    "ok_thermal": bool(ok_th),
                }
            )
            continue

        object_points.append(obj_template.copy())
        rgb_points.append(c_rgb)
        thermal_points.append(c_th)
        used_pairs.append({"rgb": str(rgb_path), "thermal": str(th_path)})

        if rgb_size is None:
            rgb_size = (rgb_gray.shape[1], rgb_gray.shape[0])
        if thermal_size is None:
            thermal_size = (th_gray.shape[1], th_gray.shape[0])

        if debug_dir is not None:
            rgb_dbg = cv2.cvtColor(rgb_gray, cv2.COLOR_GRAY2BGR)
            th_dbg = cv2.cvtColor(th_gray, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(rgb_dbg, pattern_size, c_rgb, True)
            cv2.drawChessboardCorners(th_dbg, pattern_size, c_th, True)
            cv2.imwrite(str(debug_dir / f"{i:04d}_rgb.png"), rgb_dbg)
            cv2.imwrite(str(debug_dir / f"{i:04d}_thermal.png"), th_dbg)

    if len(object_points) < int(args.min_valid_pairs):
        raise RuntimeError(
            f"Not enough valid pairs: {len(object_points)} < min-valid-pairs={args.min_valid_pairs}"
        )
    if rgb_size is None or thermal_size is None:
        raise RuntimeError("No valid image size found.")

    if args.rgb_intrinsics:
        rgb_intr = parse_intrinsics_file(Path(args.rgb_intrinsics).resolve())
        K_rgb = rgb_intr.K.copy()
        d_rgb = rgb_intr.dist.copy()
    else:
        K_rgb = np.eye(3, dtype=np.float64)
        d_rgb = np.zeros((8, 1), dtype=np.float64)
        rms_rgb, K_rgb, d_rgb, _r, _t = cv2.calibrateCamera(
            object_points, rgb_points, rgb_size, K_rgb, d_rgb
        )
        print(f"[calib] RGB calibrateCamera RMS: {rms_rgb:.6f}")

    if args.thermal_intrinsics:
        th_intr = parse_intrinsics_file(Path(args.thermal_intrinsics).resolve())
        K_th = th_intr.K.copy()
        d_th = th_intr.dist.copy()
    else:
        K_th = np.eye(3, dtype=np.float64)
        d_th = np.zeros((8, 1), dtype=np.float64)
        rms_th, K_th, d_th, _r, _t = cv2.calibrateCamera(
            object_points, thermal_points, thermal_size, K_th, d_th
        )
        print(f"[calib] Thermal calibrateCamera RMS: {rms_th:.6f}")

    stereo_flags = 0
    if args.fix_intrinsics:
        stereo_flags |= cv2.CALIB_FIX_INTRINSIC
    else:
        stereo_flags |= cv2.CALIB_USE_INTRINSIC_GUESS

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 120, 1e-7)
    rms_stereo, K_rgb_out, d_rgb_out, K_th_out, d_th_out, R, t, E, F = cv2.stereoCalibrate(
        object_points,
        rgb_points,
        thermal_points,
        K_rgb,
        d_rgb,
        K_th,
        d_th,
        imageSize=rgb_size,
        criteria=criteria,
        flags=stereo_flags,
    )

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)

    out = {
        "num_total_pairs": len(pairs),
        "num_used_pairs": len(object_points),
        "num_skipped_pairs": len(skipped_pairs),
        "board": {
            "cols": int(args.board_cols),
            "rows": int(args.board_rows),
            "square_size_m": float(args.square_size),
        },
        "rgb_image_size_wh": [int(rgb_size[0]), int(rgb_size[1])],
        "thermal_image_size_wh": [int(thermal_size[0]), int(thermal_size[1])],
        "rms_stereo": float(rms_stereo),
        "stereo_flags": int(stereo_flags),
        "T_thermal_from_rgb": T.tolist(),
        "R_thermal_from_rgb": R.tolist(),
        "t_thermal_from_rgb": t.reshape(3).tolist(),
        "rgb_intrinsics": {
            "K": K_rgb_out.tolist(),
            "dist": np.asarray(d_rgb_out).reshape(-1).tolist(),
        },
        "thermal_intrinsics": {
            "K": K_th_out.tolist(),
            "dist": np.asarray(d_th_out).reshape(-1).tolist(),
        },
        "used_pairs": used_pairs,
        "skipped_pairs": skipped_pairs,
        "output_json": str(out_path),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
