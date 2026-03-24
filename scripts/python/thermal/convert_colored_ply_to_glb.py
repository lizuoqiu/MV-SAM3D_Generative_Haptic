#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


def read_binary_rgba_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read binary little-endian PLY with vertex (x,y,z,rgba) and no faces."""
    with path.open("rb") as f:
        header_lines: list[str] = []
        while True:
            line_b = f.readline()
            if not line_b:
                raise ValueError("Unexpected EOF before end_header")
            line = line_b.decode("utf-8", "ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break
        data_offset = f.tell()

    vertex_count = None
    for line in header_lines:
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
            break
    if vertex_count is None:
        raise ValueError("PLY missing 'element vertex' declaration")

    # This file is known to be x,y,z,red,green,blue,alpha.
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("alpha", "u1"),
        ]
    )
    arr = np.memmap(path, mode="r", offset=data_offset, dtype=dtype, shape=(vertex_count,))

    xyz = np.column_stack([arr["x"], arr["y"], arr["z"]]).astype(np.float32)
    rgba = np.column_stack([arr["red"], arr["green"], arr["blue"], arr["alpha"]]).astype(np.uint8)
    return xyz, rgba


def infer_grid_shape(n: int, width: int | None, height: int | None) -> tuple[int, int]:
    if width is not None and height is not None:
        if width * height != n:
            raise ValueError(f"width*height={width*height} does not match vertex count {n}")
        return height, width
    if width is not None:
        if n % width != 0:
            raise ValueError(f"vertex count {n} not divisible by width={width}")
        return n // width, width
    if height is not None:
        if n % height != 0:
            raise ValueError(f"vertex count {n} not divisible by height={height}")
        return height, n // height

    preferred_widths = [1280, 1920, 640, 1024, 960, 1440, 1600, 2048, 3840]
    for w in preferred_widths:
        if n % w == 0:
            return n // w, w

    # Fallback: use nearest factor to sqrt(n)
    root = int(np.sqrt(n))
    for d in range(root, 0, -1):
        if n % d == 0:
            h = d
            w = n // d
            return h, w
    raise ValueError(f"Could not infer grid shape for n={n}")


def jet_like_colormap(norm: np.ndarray) -> np.ndarray:
    x = np.clip(norm, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def build_temperature_texture(temp_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    finite_valid = valid_mask & np.isfinite(temp_map)
    if not np.any(finite_valid):
        raise ValueError("No finite temperature values found in valid region")

    vals = temp_map[finite_valid]
    lo = float(np.percentile(vals, 1.0))
    hi = float(np.percentile(vals, 99.0))
    if hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))
    if hi <= lo:
        hi = lo + 1e-6

    norm = np.clip((temp_map - lo) / (hi - lo), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    rgb = (jet_like_colormap(norm) * 255.0).astype(np.uint8)
    rgb[~valid_mask] = 0
    return rgb


def build_valid_points_and_uv(xyz_grid: np.ndarray, valid_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w, _ = xyz_grid.shape
    flat_xyz = xyz_grid.reshape(-1, 3)
    flat_valid = valid_grid.reshape(-1)
    yy, xx = np.indices((h, w), dtype=np.float32)
    u = xx / max(w - 1, 1)
    v = 1.0 - (yy / max(h - 1, 1))
    flat_uv = np.stack([u, v], axis=-1).reshape(-1, 2)
    return flat_xyz[flat_valid], flat_uv[flat_valid]


def auto_edge_threshold(
    xyz_grid: np.ndarray,
    valid_grid: np.ndarray,
    mult: float,
    percentile: float,
) -> float:
    dx = np.linalg.norm(xyz_grid[:, 1:, :] - xyz_grid[:, :-1, :], axis=-1)
    mx = valid_grid[:, 1:] & valid_grid[:, :-1]
    dy = np.linalg.norm(xyz_grid[1:, :, :] - xyz_grid[:-1, :, :], axis=-1)
    my = valid_grid[1:, :] & valid_grid[:-1, :]

    d = np.concatenate([dx[mx], dy[my]])
    if d.size == 0:
        raise ValueError("No valid neighbor distances for edge threshold estimation")
    base = float(np.percentile(d, percentile))
    return base * mult


def filter_connected_components(
    vertices: np.ndarray,
    uv: np.ndarray,
    faces: np.ndarray,
    min_component_faces: int,
    keep_largest_components: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if min_component_faces <= 0 and keep_largest_components <= 0:
        return vertices, uv, faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    face_nodes = np.arange(len(faces), dtype=np.int64)
    components = trimesh.graph.connected_components(
        edges=mesh.face_adjacency,
        nodes=face_nodes,
        min_len=1,
    )
    if not components:
        return vertices, uv, faces

    comp_arrays = [np.asarray(c, dtype=np.int64) for c in components]
    comp_sizes = np.array([len(c) for c in comp_arrays], dtype=np.int64)
    order = np.argsort(comp_sizes)[::-1]

    selected: list[np.ndarray] = []
    for idx in order:
        comp = comp_arrays[int(idx)]
        if min_component_faces > 0 and len(comp) < min_component_faces:
            continue
        selected.append(comp)
        if keep_largest_components > 0 and len(selected) >= keep_largest_components:
            break

    if not selected:
        selected = [comp_arrays[int(order[0])]]

    kept_face_idx = np.sort(np.concatenate(selected))
    kept_faces = faces[kept_face_idx]
    used_vertices, inverse = np.unique(kept_faces.reshape(-1), return_inverse=True)
    new_vertices = vertices[used_vertices]
    new_uv = uv[used_vertices]
    new_faces = inverse.reshape(-1, 3).astype(np.int64)
    return new_vertices, new_uv, new_faces


def taubin_smooth_vertices(vertices: np.ndarray, faces: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return vertices

    v = vertices.astype(np.float64, copy=True)
    f = faces.astype(np.int64, copy=False)
    n = v.shape[0]

    if len(f) == 0 or n == 0:
        return vertices

    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    src = edges[:, 0]
    dst = edges[:, 1]

    deg = np.bincount(src, minlength=n).astype(np.float64)
    has_nbr = deg > 0
    deg_safe = np.where(has_nbr, deg, 1.0)

    def laplacian(curr_v: np.ndarray) -> np.ndarray:
        sum_x = np.bincount(src, weights=curr_v[dst, 0], minlength=n)
        sum_y = np.bincount(src, weights=curr_v[dst, 1], minlength=n)
        sum_z = np.bincount(src, weights=curr_v[dst, 2], minlength=n)
        mean = np.column_stack([sum_x / deg_safe, sum_y / deg_safe, sum_z / deg_safe])
        mean[~has_nbr] = curr_v[~has_nbr]
        return mean - curr_v

    lamb = 0.5
    mu = -0.53
    for _ in range(iterations):
        v = v + lamb * laplacian(v)
        v = v + mu * laplacian(v)
    return v.astype(vertices.dtype, copy=False)


def nearest_uv_from_source_points(
    query_vertices: np.ndarray,
    source_vertices: np.ndarray,
    source_uv: np.ndarray,
    device_preference: str,
) -> np.ndarray:
    if len(query_vertices) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(source_vertices) == 0:
        raise ValueError("Cannot project UV from empty source vertices")

    try:
        import open3d as o3d
    except Exception as e:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Poisson mode requires open3d for nearest-neighbor UV projection. "
            "Install open3d (e.g. `pip install open3d`)."
        ) from e

    if device_preference == "auto":
        use_cuda = bool(o3d.core.cuda.is_available())
    elif device_preference == "cuda":
        if not o3d.core.cuda.is_available():
            raise RuntimeError("Requested --poisson-device cuda, but Open3D CUDA is not available")
        use_cuda = True
    elif device_preference == "cpu":
        use_cuda = False
    else:
        raise ValueError(f"Unsupported device_preference: {device_preference}")

    device = o3d.core.Device("CUDA:0" if use_cuda else "CPU:0")
    src_tensor = o3d.core.Tensor(source_vertices.astype(np.float32, copy=False), device=device)
    qry_tensor = o3d.core.Tensor(query_vertices.astype(np.float32, copy=False), device=device)
    nns = o3d.core.nns.NearestNeighborSearch(src_tensor)
    nns.knn_index()
    idx, _ = nns.knn_search(qry_tensor, 1)
    idx_np = idx.cpu().numpy().reshape(-1).astype(np.int64, copy=False)
    return source_uv[idx_np]


def reconstruct_mesh_poisson(
    source_vertices: np.ndarray,
    source_uv: np.ndarray,
    depth: int,
    scale: float,
    linear_fit: bool,
    density_trim: float,
    normal_radius: float,
    normal_max_nn: int,
    orient_k: int,
    voxel_size: float,
    max_input_points: int,
    sample_seed: int,
    target_faces: int,
    device_preference: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import open3d as o3d
    except Exception as e:  # pragma: no cover - exercised when dependency missing
        raise RuntimeError(
            "Poisson reconstruction requires open3d. Install it first (e.g. `pip install open3d`)."
        ) from e

    points = source_vertices.astype(np.float64, copy=False)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    if max_input_points > 0 and len(pcd.points) > max_input_points:
        rng = np.random.default_rng(sample_seed)
        idx = rng.choice(len(pcd.points), size=max_input_points, replace=False)
        pcd = pcd.select_by_index(idx.tolist())
    if len(pcd.points) < 10:
        raise ValueError("Too few points after voxel downsample for Poisson reconstruction")

    if device_preference == "auto":
        use_cuda = bool(o3d.core.cuda.is_available())
    elif device_preference == "cuda":
        if not o3d.core.cuda.is_available():
            raise RuntimeError("Requested --poisson-device cuda, but Open3D CUDA is not available")
        use_cuda = True
    elif device_preference == "cpu":
        use_cuda = False
    else:
        raise ValueError(f"Unsupported device_preference: {device_preference}")

    if normal_radius <= 0.0:
        extent = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent(), dtype=np.float64)
        diag = float(np.linalg.norm(extent))
        normal_radius = max(diag * 0.01, 1e-4)

    if use_cuda:
        points_tensor = o3d.core.Tensor(
            np.asarray(pcd.points, dtype=np.float32),
            device=o3d.core.Device("CUDA:0"),
        )
        tpcd = o3d.t.geometry.PointCloud(points_tensor)
        tpcd.estimate_normals(max_nn=normal_max_nn, radius=normal_radius)
        pcd = tpcd.to_legacy()
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=normal_max_nn,
            )
        )
    if orient_k >= 3:
        try:
            pcd.orient_normals_consistent_tangent_plane(orient_k)
        except RuntimeError:
            # Disconnected point sets can fail orientation; Poisson can still run.
            pass

    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=depth,
        scale=scale,
        linear_fit=linear_fit,
    )
    dens = np.asarray(densities, dtype=np.float64)
    if density_trim > 0.0:
        q = float(np.clip(density_trim, 0.0, 0.95))
        dens_cut = float(np.quantile(dens, q))
        mesh_o3d.remove_vertices_by_mask(dens < dens_cut)
    else:
        dens_cut = float(np.min(dens)) if dens.size > 0 else 0.0

    mesh_o3d.remove_degenerate_triangles()
    mesh_o3d.remove_duplicated_triangles()
    mesh_o3d.remove_duplicated_vertices()
    mesh_o3d.remove_non_manifold_edges()

    if target_faces > 0 and len(mesh_o3d.triangles) > target_faces:
        mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_faces)
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_duplicated_triangles()
        mesh_o3d.remove_duplicated_vertices()
        mesh_o3d.remove_non_manifold_edges()

    vertices = np.asarray(mesh_o3d.vertices).astype(np.float32, copy=False)
    faces = np.asarray(mesh_o3d.triangles).astype(np.int64, copy=False)
    if len(faces) == 0 or len(vertices) == 0:
        raise ValueError("Poisson reconstruction produced an empty mesh")

    uv = nearest_uv_from_source_points(
        query_vertices=vertices,
        source_vertices=source_vertices,
        source_uv=source_uv,
        device_preference=device_preference,
    )
    stats = {
        "poisson_input_points": int(len(pcd.points)),
        "poisson_depth": int(depth),
        "poisson_scale": float(scale),
        "poisson_density_trim": float(density_trim),
        "poisson_density_cut": float(dens_cut),
        "poisson_normal_radius": float(normal_radius),
        "poisson_normal_max_nn": int(normal_max_nn),
        "poisson_orient_k": int(orient_k),
        "poisson_max_input_points": int(max_input_points),
        "poisson_target_faces": int(target_faces),
        "poisson_device": "cuda" if use_cuda else "cpu",
        "poisson_raw_vertices": int(len(vertices)),
        "poisson_raw_faces": int(len(faces)),
    }
    return vertices, uv, faces, stats


def build_mesh_from_grid(
    xyz_grid: np.ndarray,
    valid_grid: np.ndarray,
    edge_thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w, _ = xyz_grid.shape
    flat_xyz = xyz_grid.reshape(-1, 3)
    flat_valid = valid_grid.reshape(-1)

    # Vertex compaction map
    flat_to_compact = np.full(flat_xyz.shape[0], -1, dtype=np.int64)
    flat_to_compact[flat_valid] = np.arange(int(flat_valid.sum()), dtype=np.int64)

    # UV by grid coordinate
    yy, xx = np.indices((h, w), dtype=np.float32)
    u = xx / max(w - 1, 1)
    v = 1.0 - (yy / max(h - 1, 1))
    flat_uv = np.stack([u, v], axis=-1).reshape(-1, 2)

    vertices = flat_xyz[flat_valid]
    uv = flat_uv[flat_valid]

    # Cell corner indices in flattened grid
    a = (yy[:-1, :-1] * w + xx[:-1, :-1]).astype(np.int64).ravel()
    b = (yy[1:, :-1] * w + xx[1:, :-1]).astype(np.int64).ravel()
    c = (yy[:-1, 1:] * w + xx[:-1, 1:]).astype(np.int64).ravel()
    d = (yy[1:, 1:] * w + xx[1:, 1:]).astype(np.int64).ravel()

    # Triangles: (a,b,c) and (b,d,c)
    tri1 = np.stack([a, b, c], axis=1)
    tri2 = np.stack([b, d, c], axis=1)

    def filter_and_map(tri: np.ndarray) -> np.ndarray:
        v0 = flat_valid[tri[:, 0]]
        v1 = flat_valid[tri[:, 1]]
        v2 = flat_valid[tri[:, 2]]
        mask_valid = v0 & v1 & v2
        tri = tri[mask_valid]
        if tri.size == 0:
            return np.empty((0, 3), dtype=np.int64)

        p0 = flat_xyz[tri[:, 0]]
        p1 = flat_xyz[tri[:, 1]]
        p2 = flat_xyz[tri[:, 2]]
        e01 = np.linalg.norm(p0 - p1, axis=1)
        e12 = np.linalg.norm(p1 - p2, axis=1)
        e20 = np.linalg.norm(p2 - p0, axis=1)
        mask_edge = (e01 <= edge_thresh) & (e12 <= edge_thresh) & (e20 <= edge_thresh)
        tri = tri[mask_edge]
        if tri.size == 0:
            return np.empty((0, 3), dtype=np.int64)

        return np.stack(
            [
                flat_to_compact[tri[:, 0]],
                flat_to_compact[tri[:, 1]],
                flat_to_compact[tri[:, 2]],
            ],
            axis=1,
        ).astype(np.int64)

    faces1 = filter_and_map(tri1)
    faces2 = filter_and_map(tri2)
    faces = np.vstack([faces1, faces2]) if (faces1.size or faces2.size) else np.empty((0, 3), dtype=np.int64)
    if faces.size == 0:
        raise ValueError("No faces generated. Try lower --edge-multiplier or smaller --stride.")

    return vertices, uv, faces


def make_mesh(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray, tex_rgb: np.ndarray, emissive_rgb: np.ndarray | None = None) -> trimesh.Trimesh:
    tex_img = Image.fromarray(tex_rgb, mode="RGB")
    mat_kwargs = dict(
        baseColorTexture=tex_img,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        emissiveFactor=[1.0, 1.0, 1.0],
    )
    if emissive_rgb is not None:
        mat_kwargs["emissiveTexture"] = Image.fromarray(emissive_rgb, mode="RGB")

    material = PBRMaterial(**mat_kwargs)
    visual = TextureVisuals(uv=uv, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    return mesh


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert organized colored PLY + temperature NPY to textured GLB mesh")
    p.add_argument("--input-ply", default="colored_preview.ply", help="Input colored point cloud PLY")
    p.add_argument("--temperature-npy", default="temperature_fused.npy", help="Per-point temperature array")
    p.add_argument("--output-dir", default="exports/thermal_glb", help="Output directory")
    p.add_argument("--basename", default="kettle", help="Output file basename")
    p.add_argument(
        "--reconstruct-method",
        choices=["grid", "poisson"],
        default="poisson",
        help="Geometry reconstruction mode: grid triangulation or Poisson surface reconstruction",
    )
    p.add_argument("--width", type=int, default=None, help="Grid width (optional, auto-infer if omitted)")
    p.add_argument("--height", type=int, default=None, help="Grid height (optional, auto-infer if omitted)")
    p.add_argument("--stride", type=int, default=2, help="Grid decimation stride (1=full, 2=half)")
    p.add_argument(
        "--edge-multiplier",
        type=float,
        default=4.0,
        help="Max triangle edge as multiplier of the selected neighbor-distance percentile",
    )
    p.add_argument(
        "--edge-percentile",
        type=float,
        default=50.0,
        help="Neighbor-distance percentile used as base for edge threshold (50=median)",
    )
    p.add_argument(
        "--min-component-faces",
        type=int,
        default=0,
        help="Drop connected components smaller than this face count (0 disables)",
    )
    p.add_argument(
        "--keep-largest-components",
        type=int,
        default=0,
        help="Keep only N largest connected components (0 keeps all)",
    )
    p.add_argument(
        "--taubin-iterations",
        type=int,
        default=0,
        help="Apply Taubin smoothing iterations after topology filtering (0 disables)",
    )
    p.add_argument("--poisson-depth", type=int, default=9, help="Poisson octree depth (poisson mode)")
    p.add_argument("--poisson-scale", type=float, default=1.1, help="Poisson scale factor (poisson mode)")
    p.add_argument(
        "--poisson-linear-fit",
        action="store_true",
        help="Enable linear fit in Poisson reconstruction (poisson mode)",
    )
    p.add_argument(
        "--poisson-density-trim",
        type=float,
        default=0.02,
        help="Trim lowest vertex-density quantile after Poisson (0 disables, poisson mode)",
    )
    p.add_argument(
        "--poisson-normal-radius",
        type=float,
        default=0.0,
        help="Normal-estimation radius (<=0 auto from bbox diag, poisson mode)",
    )
    p.add_argument(
        "--poisson-normal-max-nn",
        type=int,
        default=30,
        help="Max NN for normal estimation (poisson mode)",
    )
    p.add_argument(
        "--poisson-orient-k",
        type=int,
        default=20,
        help="K for consistent normal orientation (poisson mode, <3 disables)",
    )
    p.add_argument(
        "--poisson-voxel-size",
        type=float,
        default=0.01,
        help="Optional voxel downsample size before Poisson (poisson mode)",
    )
    p.add_argument(
        "--poisson-max-input-points",
        type=int,
        default=120000,
        help="Maximum input points for Poisson after voxel downsample (0 disables)",
    )
    p.add_argument(
        "--poisson-sample-seed",
        type=int,
        default=0,
        help="Random seed for Poisson input sampling",
    )
    p.add_argument(
        "--poisson-target-faces",
        type=int,
        default=0,
        help="Optional quadric decimation target face count after Poisson (0 disables)",
    )
    p.add_argument(
        "--poisson-device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for Poisson preprocessing and UV NN projection",
    )
    args = p.parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if not (0.0 < args.edge_percentile <= 100.0):
        raise ValueError("--edge-percentile must be in (0, 100]")
    if args.min_component_faces < 0:
        raise ValueError("--min-component-faces must be >= 0")
    if args.keep_largest_components < 0:
        raise ValueError("--keep-largest-components must be >= 0")
    if args.taubin_iterations < 0:
        raise ValueError("--taubin-iterations must be >= 0")
    if args.poisson_depth < 6:
        raise ValueError("--poisson-depth should be >= 6")
    if args.poisson_scale <= 0:
        raise ValueError("--poisson-scale must be > 0")
    if not (0.0 <= args.poisson_density_trim < 1.0):
        raise ValueError("--poisson-density-trim must be in [0, 1)")
    if args.poisson_normal_max_nn < 3:
        raise ValueError("--poisson-normal-max-nn must be >= 3")
    if args.poisson_orient_k < 0:
        raise ValueError("--poisson-orient-k must be >= 0")
    if args.poisson_voxel_size < 0:
        raise ValueError("--poisson-voxel-size must be >= 0")
    if args.poisson_max_input_points < 0:
        raise ValueError("--poisson-max-input-points must be >= 0")
    if args.poisson_target_faces < 0:
        raise ValueError("--poisson-target-faces must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    input_ply = Path(args.input_ply).resolve()
    temp_npy = Path(args.temperature_npy).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz, rgba = read_binary_rgba_ply(input_ply)
    n = xyz.shape[0]
    h, w = infer_grid_shape(n, args.width, args.height)

    if h * w != n:
        raise ValueError(f"Inferred grid {h}x{w} does not match vertex count {n}")

    temp = np.load(temp_npy, allow_pickle=False)
    if temp.ndim != 1 or temp.shape[0] != n:
        raise ValueError(f"temperature array shape {temp.shape} does not match vertex count {n}")

    xyz_grid = xyz.reshape(h, w, 3)[:: args.stride, :: args.stride, :]
    rgba_grid = rgba.reshape(h, w, 4)[:: args.stride, :: args.stride, :]
    temp_grid = temp.reshape(h, w)[:: args.stride, :: args.stride]

    valid_grid = (rgba_grid[..., 3] > 0) & np.isfinite(xyz_grid).all(axis=-1)
    source_vertices, source_uv = build_valid_points_and_uv(xyz_grid, valid_grid)
    if len(source_vertices) == 0:
        raise ValueError("No valid input points after stride/mask filtering")

    edge_thresh: float | None = None
    poisson_stats: dict[str, Any] | None = None
    if args.reconstruct_method == "grid":
        edge_thresh = auto_edge_threshold(
            xyz_grid=xyz_grid,
            valid_grid=valid_grid,
            mult=args.edge_multiplier,
            percentile=args.edge_percentile,
        )
        vertices, uv, faces = build_mesh_from_grid(xyz_grid, valid_grid, edge_thresh)
    elif args.reconstruct_method == "poisson":
        vertices, uv, faces, poisson_stats = reconstruct_mesh_poisson(
            source_vertices=source_vertices,
            source_uv=source_uv,
            depth=args.poisson_depth,
            scale=args.poisson_scale,
            linear_fit=args.poisson_linear_fit,
            density_trim=args.poisson_density_trim,
            normal_radius=args.poisson_normal_radius,
            normal_max_nn=args.poisson_normal_max_nn,
            orient_k=args.poisson_orient_k,
            voxel_size=args.poisson_voxel_size,
            max_input_points=args.poisson_max_input_points,
            sample_seed=args.poisson_sample_seed,
            target_faces=args.poisson_target_faces,
            device_preference=args.poisson_device,
        )
    else:
        raise ValueError(f"Unsupported reconstruct method: {args.reconstruct_method}")

    vertices, uv, faces = filter_connected_components(
        vertices=vertices,
        uv=uv,
        faces=faces,
        min_component_faces=args.min_component_faces,
        keep_largest_components=args.keep_largest_components,
    )
    vertices = taubin_smooth_vertices(vertices, faces, iterations=args.taubin_iterations)

    object_tex = rgba_grid[..., :3].copy()
    object_tex[~valid_grid] = 0
    temp_tex = build_temperature_texture(temp_grid, valid_grid)

    base = args.basename
    object_tex_path = out_dir / f"{base}_object_texture.png"
    temp_tex_path = out_dir / f"{base}_temperature_texture.png"
    glb_object_path = out_dir / f"{base}_object_color.glb"
    glb_temp_path = out_dir / f"{base}_temperature.glb"
    glb_combined_path = out_dir / f"{base}_combined.glb"

    Image.fromarray(object_tex, mode="RGB").save(object_tex_path)
    Image.fromarray(temp_tex, mode="RGB").save(temp_tex_path)

    mesh_object = make_mesh(vertices, faces, uv, object_tex)
    mesh_temp = make_mesh(vertices, faces, uv, temp_tex)
    mesh_combined = make_mesh(vertices, faces, uv, object_tex, emissive_rgb=temp_tex)

    mesh_object.export(glb_object_path)
    mesh_temp.export(glb_temp_path)
    mesh_combined.export(glb_combined_path)

    print("[convert] done")
    print(f"[convert] input_ply={input_ply}")
    print(f"[convert] temp_npy={temp_npy}")
    print(f"[convert] grid={h}x{w} stride={args.stride}")
    print(f"[convert] reconstruct_method={args.reconstruct_method}")
    print(f"[convert] valid_vertices_ratio={valid_grid.mean():.4f}")
    if edge_thresh is not None:
        print(f"[convert] edge_percentile={args.edge_percentile:.1f}")
        print(f"[convert] edge_threshold={edge_thresh:.6f}")
    if poisson_stats is not None:
        print(f"[convert] poisson_input_points={poisson_stats['poisson_input_points']}")
        print(f"[convert] poisson_depth={poisson_stats['poisson_depth']}")
        print(f"[convert] poisson_scale={poisson_stats['poisson_scale']:.3f}")
        print(f"[convert] poisson_density_trim={poisson_stats['poisson_density_trim']:.4f}")
        print(f"[convert] poisson_density_cut={poisson_stats['poisson_density_cut']:.6f}")
        print(f"[convert] poisson_normal_radius={poisson_stats['poisson_normal_radius']:.6f}")
        print(f"[convert] poisson_max_input_points={poisson_stats['poisson_max_input_points']}")
        print(f"[convert] poisson_target_faces={poisson_stats['poisson_target_faces']}")
        print(f"[convert] poisson_device={poisson_stats['poisson_device']}")
    print(f"[convert] mesh_vertices={len(vertices)} mesh_faces={len(faces)}")
    print(f"[convert] object_texture={object_tex_path}")
    print(f"[convert] temperature_texture={temp_tex_path}")
    print(f"[convert] glb_object={glb_object_path}")
    print(f"[convert] glb_temperature={glb_temp_path}")
    print(f"[convert] glb_combined={glb_combined_path}")


if __name__ == "__main__":
    main()
