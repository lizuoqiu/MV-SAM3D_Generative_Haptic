#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize mesh temperature mapping for quick verification.")
    parser.add_argument("--mesh", required=True, help="Mesh file path.")
    parser.add_argument("--temperature", required=True, help="Per-vertex temperature .npy file.")
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--max-points", type=int, default=80000, help="Max points sampled for plotting.")
    parser.add_argument(
        "--background-points",
        type=int,
        default=120000,
        help="Max background mesh points to plot in gray so full shape remains visible.",
    )
    parser.add_argument("--temp-min", type=float, default=None)
    parser.add_argument("--temp-max", type=float, default=None)
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(str(Path(args.mesh).resolve()))
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {args.mesh}")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    temp = np.load(str(Path(args.temperature).resolve()))
    if temp.shape[0] != verts.shape[0]:
        raise ValueError(
            f"Temperature length ({temp.shape[0]}) does not match vertices ({verts.shape[0]})"
        )

    rng = np.random.default_rng(0)

    valid = np.isfinite(temp) & (temp != 0.0)
    if valid.sum() == 0:
        valid = np.isfinite(temp)
    if valid.sum() == 0:
        raise RuntimeError("No valid temperature values to visualize.")

    valid_idx = np.where(valid)[0]
    if valid_idx.size > args.max_points:
        valid_idx = rng.choice(valid_idx, size=args.max_points, replace=False)

    bg_idx = np.where(np.isfinite(temp))[0]
    if bg_idx.size > args.background_points:
        bg_idx = rng.choice(bg_idx, size=args.background_points, replace=False)

    p_valid = verts[valid_idx]
    t_valid = temp[valid_idx]
    p_bg = verts[bg_idx]

    tmin = float(np.min(t_valid) if args.temp_min is None else args.temp_min)
    tmax = float(np.max(t_valid) if args.temp_max is None else args.temp_max)
    if tmax <= tmin:
        tmax = tmin + 1e-6

    fig = plt.figure(figsize=(14, 10))
    views = [
        (20, 45, "View A"),
        (20, 135, "View B"),
        (20, 225, "View C"),
        (80, 0, "Top"),
    ]

    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        ax.scatter(
            p_bg[:, 0], p_bg[:, 1], p_bg[:, 2],
            c="#9a9a9a",
            s=0.10,
            alpha=0.20,
        )
        sc = ax.scatter(
            p_valid[:, 0], p_valid[:, 1], p_valid[:, 2],
            c=t_valid,
            cmap="inferno",
            s=0.35,
            vmin=tmin,
            vmax=tmax,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_axis_off()

    cbar = fig.colorbar(sc, ax=fig.axes, fraction=0.02, pad=0.02)
    cbar.set_label("Temperature")
    fig.suptitle(
        f"Temperature Mapping Verification\n"
        f"valid_vertices={valid.sum()}/{len(valid)} ({100.0 * valid.mean():.1f}%)",
        fontsize=12,
    )
    fig.tight_layout()

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220)
    print(f"[viz] saved: {out}")
    print(f"[viz] temp_range=({tmin:.6f}, {tmax:.6f})")
    print(f"[viz] valid_coverage={valid.sum()}/{len(valid)} ({100.0 * valid.mean():.2f}%)")


if __name__ == "__main__":
    main()
