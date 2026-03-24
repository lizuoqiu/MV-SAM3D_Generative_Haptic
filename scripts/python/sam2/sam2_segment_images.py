#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def natural_sort_key(path: Path) -> tuple[int, int, str]:
    stem = path.stem
    try:
        return (0, int(stem), stem)
    except ValueError:
        return (1, 0, stem)


def image_files_in_dir(image_dir: Path) -> list[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files: list[Path] = []
    for ext in exts:
        files.extend(image_dir.glob(ext))
    return sorted(files, key=natural_sort_key)


def border_touch_ratio(mask: np.ndarray) -> float:
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    border_pixels = np.logical_and(mask, border).sum()
    total_border = border.sum()
    return float(border_pixels) / float(max(total_border, 1))


def center_distance(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return 1.0
    cx = float(xs.mean())
    cy = float(ys.mean())
    h, w = mask.shape
    nx = (cx - (w / 2.0)) / max(w / 2.0, 1.0)
    ny = (cy - (h / 2.0)) / max(h / 2.0, 1.0)
    return float(np.sqrt(nx * nx + ny * ny))


def score_mask(mask_dict: dict[str, Any], image_hw: tuple[int, int], min_area_ratio: float, max_area_ratio: float) -> float:
    seg = np.asarray(mask_dict["segmentation"], dtype=bool)
    h, w = image_hw
    area_ratio = float(seg.sum()) / float(max(h * w, 1))
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return float("-inf")

    iou = float(mask_dict.get("predicted_iou", 0.0))
    stability = float(mask_dict.get("stability_score", 0.0))
    b_ratio = border_touch_ratio(seg)
    c_dist = center_distance(seg)

    # Single-object turntable bias:
    # prefer center-ish, non-border-dominant, stable masks with reasonable area.
    return (1.2 * area_ratio) + (0.6 * iou) + (0.6 * stability) - (1.2 * b_ratio) - (0.8 * c_dist)


def pick_best_mask(
    mask_candidates: list[dict[str, Any]],
    image_hw: tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    best_score = float("-inf")
    best_mask: np.ndarray | None = None
    best_meta: dict[str, Any] = {}

    for idx, candidate in enumerate(mask_candidates):
        seg = np.asarray(candidate["segmentation"], dtype=bool)
        score = score_mask(candidate, image_hw, min_area_ratio=min_area_ratio, max_area_ratio=max_area_ratio)
        if score > best_score:
            best_score = score
            best_mask = seg
            best_meta = {
                "index": idx,
                "score": float(score),
                "area": int(seg.sum()),
                "predicted_iou": float(candidate.get("predicted_iou", 0.0)),
                "stability_score": float(candidate.get("stability_score", 0.0)),
                "bbox_xywh": candidate.get("bbox"),
            }

    if best_mask is not None and np.isfinite(best_score):
        return best_mask, best_meta

    # Fallback: largest mask.
    if not mask_candidates:
        return np.zeros(image_hw, dtype=bool), {"index": -1, "score": float("-inf"), "area": 0}

    areas = [int(np.asarray(c["segmentation"], dtype=bool).sum()) for c in mask_candidates]
    max_idx = int(np.argmax(areas))
    largest = np.asarray(mask_candidates[max_idx]["segmentation"], dtype=bool)
    return largest, {
        "index": max_idx,
        "score": None,
        "area": int(largest.sum()),
        "predicted_iou": float(mask_candidates[max_idx].get("predicted_iou", 0.0)),
        "stability_score": float(mask_candidates[max_idx].get("stability_score", 0.0)),
        "bbox_xywh": mask_candidates[max_idx].get("bbox"),
        "fallback": "largest_mask",
    }


def save_rgba_mask(mask: np.ndarray, out_path: Path) -> None:
    h, w = mask.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(out_path)


def save_cutout(rgb: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    cut = rgb.copy()
    cut[~mask] = 0
    Image.fromarray(cut).save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate object masks from RGB images using SAM2.")
    parser.add_argument("--image-dir", required=True, help="Input directory with RGB images.")
    parser.add_argument("--output-mask-dir", required=True, help="Output directory for RGBA masks.")
    parser.add_argument(
        "--output-cutout-dir",
        default=None,
        help="Optional output directory for RGB cutouts with background removed.",
    )
    parser.add_argument("--sam2-checkpoint", required=True, help="Path to SAM2 checkpoint (.pt).")
    parser.add_argument("--sam2-config", required=True, help="SAM2 model config path (relative to sam2 repo).")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.86)
    parser.add_argument("--stability-score-thresh", type=float, default=0.92)
    parser.add_argument("--min-mask-region-area", type=int, default=500)
    parser.add_argument("--min-area-ratio", type=float, default=0.01)
    parser.add_argument("--max-area-ratio", type=float, default=0.95)
    parser.add_argument("--metadata-json", default=None, help="Optional output metadata JSON path.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    output_mask_dir = Path(args.output_mask_dir).resolve()
    output_cutout_dir = Path(args.output_cutout_dir).resolve() if args.output_cutout_dir else None
    ckpt = Path(args.sam2_checkpoint).resolve()

    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir}")
    if not ckpt.exists():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt}")

    output_mask_dir.mkdir(parents=True, exist_ok=True)
    if output_cutout_dir is not None:
        output_cutout_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = build_sam2(args.sam2_config, str(ckpt), device=device, apply_postprocessing=False)
    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
    )

    files = image_files_in_dir(image_dir)
    if not files:
        raise RuntimeError(f"No images found in {image_dir}")

    metadata: dict[str, Any] = {
        "image_dir": str(image_dir),
        "output_mask_dir": str(output_mask_dir),
        "output_cutout_dir": str(output_cutout_dir) if output_cutout_dir else None,
        "sam2_checkpoint": str(ckpt),
        "sam2_config": args.sam2_config,
        "device": device,
        "images": [],
    }

    for image_path in files:
        rgb = np.array(Image.open(image_path).convert("RGB"))
        h, w = rgb.shape[:2]

        if device == "cuda":
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                candidates = mask_generator.generate(rgb)
        else:
            with torch.inference_mode():
                candidates = mask_generator.generate(rgb)

        best_mask, best_meta = pick_best_mask(
            candidates,
            image_hw=(h, w),
            min_area_ratio=args.min_area_ratio,
            max_area_ratio=args.max_area_ratio,
        )

        out_mask_path = output_mask_dir / f"{image_path.stem}.png"
        save_rgba_mask(best_mask, out_mask_path)

        out_cutout_path = None
        if output_cutout_dir is not None:
            out_cutout_path = output_cutout_dir / f"{image_path.stem}.png"
            save_cutout(rgb, best_mask, out_cutout_path)

        metadata["images"].append(
            {
                "image": str(image_path),
                "num_candidates": len(candidates),
                "selected": best_meta,
                "mask_pixels": int(best_mask.sum()),
                "mask_ratio": float(best_mask.mean()),
                "mask_path": str(out_mask_path),
                "cutout_path": str(out_cutout_path) if out_cutout_path else None,
            }
        )

        print(
            f"[sam2] {image_path.name}: candidates={len(candidates)} "
            f"selected_idx={best_meta.get('index')} area={int(best_mask.sum())}"
        )

    if args.metadata_json:
        out_meta = Path(args.metadata_json).resolve()
        out_meta.parent.mkdir(parents=True, exist_ok=True)
        out_meta.write_text(json.dumps(metadata, indent=2))
        print(f"[sam2] metadata written: {out_meta}")

    print(f"[sam2] Completed {len(files)} images")


if __name__ == "__main__":
    main()
