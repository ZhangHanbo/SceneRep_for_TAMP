#!/usr/bin/env python3
"""Inspect the EKF tracker's VoxelObservability grid in 3D.

Replays a trajectory's depth frames through ``VoxelObservability.integrate_depth``
and renders the per-voxel state (occupied / empty / unobserved) as colored
Open3D point clouds. Designed as a preflight check for the gravity-predict fix:
before changing predict_landing_pose, verify the grid the predictor reads is
correct.

Usage::

    # static — integrate all frames first, then open viewer
    python scripts/visualize_voxel_obs.py

    # animated — open viewer, watch the grid grow as frames are integrated
    python scripts/visualize_voxel_obs.py --animate-every 5

    # add the unseen voxels (heavy):
    python scripts/visualize_voxel_obs.py --animate-every 5 --show-unseen

    # overlay scene-wide TSDF mesh from the heuristic tracker:
    python scripts/visualize_voxel_obs.py --animate-every 10 --with-tsdf

    # inspect a specific column (apple's actual landing at fr 396):
    python scripts/visualize_voxel_obs.py --column 0.10,0.65

The viewer prints ``voxel_obs.stats()`` and the column raycast-down result
(``column_state``, ``surface_z``, ``first_unseen_z``) for each highlighted
column at the end of integration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from perception.voxel_observability import VoxelObservability  # noqa: E402

DATASET_DIR = os.path.join(SCENEREP_ROOT, "datasets")

# Fetch head-camera intrinsics (matches scripts/visualize_ekf_tracking.py).
K_DEFAULT = np.array([
    [554.3827, 0.0, 320.5],
    [0.0, 554.3827, 240.5],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

# Apple landmarks for default column inspection on apple_drop:
#   release  ≈ (-0.06, +0.33, +0.94)  at fr 267 (just before release)
#   landing  ≈ (+0.10, +0.65, +0.08)  at fr 396 (re-acquired after the drop)
DEFAULT_APPLE_DROP_COLUMNS = [
    (-0.06, 0.33, "apple-release"),
    (0.10, 0.65, "apple-landing"),
]

COLOR_OCCUPIED = [0.85, 0.15, 0.15]
COLOR_EMPTY = [0.55, 0.70, 0.85]
COLOR_UNSEEN = [0.55, 0.55, 0.55]
COLOR_TSDF = [0.30, 0.85, 0.30]


# ─────────────────────────────────────────────────────────────────────
#  Loaders (mirrors scripts/visualize_ekf_tracking.py)
# ─────────────────────────────────────────────────────────────────────

def _load_amcl_poses(path: str) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out.append(T)
    return out


def _load_T_bc_poses(path: str) -> Optional[Dict[int, np.ndarray]]:
    if not os.path.exists(path):
        return None
    out: Dict[int, np.ndarray] = {}
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            try:
                idx = int(arr[0])
                tx, ty, tz, qx, qy, qz, qw = map(float, arr[1:])
            except ValueError:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out[idx] = T
    return out or None


def _load_depth(path: str) -> Optional[np.ndarray]:
    if not os.path.exists(path):
        return None
    return np.load(path)


# ─────────────────────────────────────────────────────────────────────
#  Voxel-state extraction (vectorised)
# ─────────────────────────────────────────────────────────────────────

def _state_masks(vo: VoxelObservability) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    occ = vo._n_hit >= vo.n_min_hit
    emp = (~occ) & (vo._n_pass >= vo.n_min_pass)
    unsn = ~(occ | emp)
    return occ, emp, unsn


def _voxel_centres(vo: VoxelObservability, mask: np.ndarray) -> np.ndarray:
    ijk = np.argwhere(mask)
    if ijk.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return vo.aabb_min[None, :] + (ijk + 0.5) * vo.voxel_size


def _fit_aabb(slam_poses: List[np.ndarray],
               T_bc_map: Optional[Dict[int, np.ndarray]],
               K: np.ndarray,
               depth_dir: str,
               vo_kwargs: Dict,
               integrate_kwargs: Dict,
               *,
               start: int = 0, end: Optional[int] = None,
               step: int = 1, pad_voxels: int = 4,
               require_occupied: bool = True,
               ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a tight AABB around the trajectory's actually-touched voxels via
    a probe-pass integration.

    Strategy: integrate every frame into a throwaway VoxelObservability built
    with the YAML's full AABB, then take the world-frame AABB of every voxel
    where ``_n_hit > 0`` (any depth termination). Padded by
    ``pad_voxels * voxel_size_m`` so the second pass has slack for boundary
    rays.

    Why occupied-only and not occupied-or-passed: the pass-touched volume
    fans out along rays all the way to the FOV edges (3 m at min_range)
    which makes the AABB explode. Occupied voxels mark *where surfaces
    actually are* — that's the scene volume the user wants to inspect.

    Caveat: this runs ``integrate_depth`` once for the probe pass, so the
    full inspection is 2× the integration cost.
    """
    if end is None:
        end = len(slam_poses)
    voxel_size_m = float(vo_kwargs["voxel_size_m"])

    # Probe pass with the YAML AABB.
    probe = VoxelObservability(**vo_kwargs)
    for fr in range(start, min(end, len(slam_poses)), step):
        T_wb = slam_poses[fr]
        T_bc = (T_bc_map.get(fr) if T_bc_map is not None else None)
        if T_bc is None:
            T_bc = np.eye(4, dtype=np.float64)
        depth_path = os.path.join(depth_dir, f"depth_{fr:06d}.npy")
        if not os.path.exists(depth_path):
            continue
        try:
            depth = np.load(depth_path).astype(np.float32)
        except Exception:
            continue
        T_cw = np.asarray(T_wb, dtype=np.float64) @ np.asarray(T_bc, dtype=np.float64)
        try:
            probe.integrate_depth(depth=depth, K=K, T_cw=T_cw, **integrate_kwargs)
        except Exception:
            continue

    if require_occupied:
        touched = probe._n_hit > 0
    else:
        touched = (probe._n_hit > 0) | (probe._n_pass > 0)
    if not touched.any():
        # Fallback: keep the original AABB.
        return probe.aabb_min.copy(), probe.aabb_max.copy()
    ijk = np.argwhere(touched)
    pts_w = probe.aabb_min[None, :] + (ijk + 0.5) * voxel_size_m
    pad = pad_voxels * voxel_size_m
    return pts_w.min(axis=0) - pad, pts_w.max(axis=0) + pad


# ─────────────────────────────────────────────────────────────────────
#  TSDF helpers (heuristic-style scene-wide volume)
# ─────────────────────────────────────────────────────────────────────

def _build_tsdf(voxel_size_m: float, max_range_m: float):
    from heuristic_tracker.tsdf_o3d import TSDFVolume
    return TSDFVolume(voxel_size=voxel_size_m, depth_max=max_range_m)


def _integrate_tsdf(tsdf, depth: np.ndarray, K: np.ndarray, T_cw: np.ndarray) -> None:
    h, w = depth.shape
    color = np.full((h, w, 3), 0.6, dtype=np.float32)
    T_wc = np.linalg.inv(T_cw)
    tsdf.integrate(color=color, depth=depth.astype(np.float32),
                   K=K, T=T_wc, depth_scale=1.0)


def _tsdf_mesh(o3d, tsdf):
    try:
        V, F, _N, _C = tsdf.get_mesh(weight_th=0.1)
    except Exception:
        return None
    if not V.size or not F.size:
        return None
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(V.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(F.astype(np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(COLOR_TSDF)
    return mesh


# ─────────────────────────────────────────────────────────────────────
#  Geometry builders
# ─────────────────────────────────────────────────────────────────────

def _build_dynamic_geoms(o3d, vo: VoxelObservability, args,
                          columns: List[Tuple[float, float, str]],
                          rng: np.random.Generator) -> List:
    """Build the geometries that change with the voxel grid state.

    Returns the list of Open3D geometries to (re-)add to the viewer:
    occupied/empty/unseen point clouds + per-column highlight spheres.
    """
    geoms: List = []
    occ_mask, emp_mask, unsn_mask = _state_masks(vo)

    occ_pts = _voxel_centres(vo, occ_mask)
    if occ_pts.size:
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(occ_pts))
        pcd.paint_uniform_color(COLOR_OCCUPIED)
        geoms.append(pcd)

    if args.show_empty:
        emp_pts = _voxel_centres(vo, emp_mask)
        if emp_pts.size:
            sub = max(1, int(args.point_subsample))
            if sub > 1:
                idx = rng.permutation(emp_pts.shape[0])[::sub]
                emp_pts = emp_pts[idx]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(emp_pts))
            pcd.paint_uniform_color(COLOR_EMPTY)
            geoms.append(pcd)

    if not args.hide_unseen:
        unsn_pts = _voxel_centres(vo, unsn_mask)
        if unsn_pts.size:
            # Unseen typically dominates (~80% of grid); aggressive subsample.
            sub = max(int(args.point_subsample), 8)
            idx = rng.permutation(unsn_pts.shape[0])[::sub]
            unsn_pts = unsn_pts[idx]
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unsn_pts))
            pcd.paint_uniform_color(COLOR_UNSEEN)
            geoms.append(pcd)

    state_colors = {0: [0.50, 0.50, 0.50],
                     1: [0.30, 0.55, 0.95],
                     2: [1.00, 0.20, 0.20]}
    for x, y, _name in columns:
        z_top = float(vo.aabb_max[2])
        result = vo.raycast_down((x, y, z_top))
        for z_centre, state_code in result.states:
            sph = o3d.geometry.TriangleMesh.create_sphere(
                radius=vo.voxel_size * 0.55)
            sph.translate([x, y, z_centre])
            sph.paint_uniform_color(state_colors.get(state_code, [1.0, 1.0, 0.0]))
            geoms.append(sph)
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=vo.voxel_size * 0.7, height=vo.voxel_size * 1.5)
        cone.translate([x, y, z_top + vo.voxel_size])
        cone.paint_uniform_color([1.0, 0.85, 0.0])
        geoms.append(cone)
    return geoms


def _build_static_geoms(o3d, vo: VoxelObservability) -> List:
    aabb = o3d.geometry.AxisAlignedBoundingBox(
        min_bound=vo.aabb_min, max_bound=vo.aabb_max)
    aabb.color = (0.0, 0.0, 0.0)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    return [aabb, axes]


# ─────────────────────────────────────────────────────────────────────
#  Per-frame integration helper
# ─────────────────────────────────────────────────────────────────────

def _integrate_frame(vo, depth_dir, fr, slam_poses, T_bc_map,
                      integrate_kwargs, tsdf,
                      arm_builder=None, joints_map=None) -> bool:
    depth = _load_depth(os.path.join(depth_dir, f"depth_{fr:06d}.npy"))
    if depth is None:
        return False
    T_wb = slam_poses[fr]
    T_bc = (T_bc_map.get(fr) if T_bc_map is not None else np.eye(4))
    if T_bc is None:
        T_bc = np.eye(4)
    T_cw = np.asarray(T_wb, dtype=np.float64) @ np.asarray(T_bc, dtype=np.float64)

    # Optional arm mask: zero out depth pixels under the projected arm so
    # they get rejected by the (min_depth_m, max_range_m) gate inside
    # integrate_depth. The same depth is forwarded to TSDF below — keeping
    # the two views consistent.
    depth = depth.astype(np.float32)
    if arm_builder is not None and joints_map is not None and fr in joints_map:
        try:
            mask = arm_builder.build(joints_map[fr], T_bc)
            depth = depth.copy()
            depth[mask > 0] = 0.0
        except Exception as e:
            print(f"[warn] arm_mask build failed at fr {fr}: {e}")

    try:
        vo.integrate_depth(depth=depth, K=K_DEFAULT, T_cw=T_cw,
                            **integrate_kwargs)
    except Exception as e:
        print(f"[warn] integrate_depth failed at fr {fr}: {e}")
        return False
    if tsdf is not None:
        try:
            _integrate_tsdf(tsdf, depth, K_DEFAULT, T_cw)
        except Exception as e:
            print(f"[warn] tsdf.integrate failed at fr {fr}: {e}")
    return True


# ─────────────────────────────────────────────────────────────────────
#  Animated viewer path
# ─────────────────────────────────────────────────────────────────────

def _run_animated(o3d, vo, integrate_kwargs, slam_poses, T_bc_map, depth_dir,
                   args, columns, tsdf, total_frames: int,
                   arm_builder=None, joints_map=None) -> None:
    """Open viewer, integrate frame-by-frame, refresh geometry every
    ``args.animate_every`` frames, keep window open after integration finishes.
    """
    rng = np.random.default_rng(0)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"voxel_obs animation — {args.trajectory}",
                       width=1280, height=860)

    # Static scaffolding (added once, never removed).
    static_geoms = _build_static_geoms(o3d, vo)
    for g in static_geoms:
        vis.add_geometry(g, reset_bounding_box=True)

    dynamic_geoms: List = []
    tsdf_mesh = None
    saved_view = None
    n_done = 0
    end = min(args.max_frame, total_frames)
    refresh_every = max(1, int(args.animate_every))

    print(f"[viewer] animated mode: refresh every {refresh_every} frames. "
          f"Drag to rotate, scroll to zoom; window stays open after integration.")

    def _refresh():
        nonlocal dynamic_geoms, tsdf_mesh, saved_view
        # Save camera state so the view doesn't snap on geometry swap.
        try:
            saved_view = (vis.get_view_control()
                            .convert_to_pinhole_camera_parameters())
        except Exception:
            saved_view = None
        # Remove old dynamic geoms.
        for g in dynamic_geoms:
            vis.remove_geometry(g, reset_bounding_box=False)
        dynamic_geoms = _build_dynamic_geoms(o3d, vo, args, columns, rng)
        for g in dynamic_geoms:
            vis.add_geometry(g, reset_bounding_box=False)
        if tsdf is not None:
            if tsdf_mesh is not None:
                vis.remove_geometry(tsdf_mesh, reset_bounding_box=False)
                tsdf_mesh = None
            new_mesh = _tsdf_mesh(o3d, tsdf)
            if new_mesh is not None:
                vis.add_geometry(new_mesh, reset_bounding_box=False)
                tsdf_mesh = new_mesh
        if saved_view is not None:
            try:
                (vis.get_view_control()
                    .convert_from_pinhole_camera_parameters(
                        saved_view, allow_arbitrary=True))
            except Exception:
                pass

    for fr in range(args.start, end, args.step):
        if not _integrate_frame(vo, depth_dir, fr, slam_poses, T_bc_map,
                                  integrate_kwargs, tsdf,
                                  arm_builder=arm_builder, joints_map=joints_map):
            continue
        n_done += 1
        if n_done % refresh_every == 0 or fr == end - 1:
            _refresh()
            occ_mask, emp_mask, unsn_mask = _state_masks(vo)
            print(f"  fr {fr:>4}: occ={int(occ_mask.sum()):>6}  "
                  f"emp={int(emp_mask.sum()):>6}  "
                  f"unsn={int(unsn_mask.sum()):>6}")
        vis.poll_events()
        vis.update_renderer()

    print("[viewer] integration complete. Window stays open until you close it.")
    # Final refresh to make sure the last frame is reflected.
    _refresh()
    vis.run()
    vis.destroy_window()


# ─────────────────────────────────────────────────────────────────────
#  Static viewer path (default)
# ─────────────────────────────────────────────────────────────────────

def _run_static(o3d, vo, args, columns, tsdf) -> None:
    rng = np.random.default_rng(0)
    geometries: List = []
    geometries.extend(_build_static_geoms(o3d, vo))
    geometries.extend(_build_dynamic_geoms(o3d, vo, args, columns, rng))

    if tsdf is not None:
        mesh = _tsdf_mesh(o3d, tsdf)
        if mesh is not None:
            geometries.append(mesh)
            print(f"[tsdf] mesh: V={len(mesh.vertices)}  F={len(mesh.triangles)}  "
                  f"(green; should align with the red occupied points)")
        else:
            print("[tsdf] empty mesh (no surfaces extracted at weight_th=0.1)")

    print("[viewer] opening Open3D window. Drag to rotate, scroll to zoom, "
          "press Q to quit.")
    legend = "occupied=red"
    if not args.hide_unseen:
        legend += ", unseen=gray"
    if args.show_empty:
        legend += ", empty=blue"
    if tsdf is not None:
        legend += ", tsdf=green"
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"voxel_obs — {args.trajectory}  ({legend})")


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", default="apple_drop")
    ap.add_argument("--max-frame", type=int, default=10**6,
                    help="exclusive upper bound on frame index (default: all frames)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--config-path", dest="config_path", default=None,
                    help="customization YAML (defaults to ekf_tracker/configs/default.yaml)")
    ap.add_argument("--show-empty", action="store_true",
                    help="render the empty (free-space) voxels in pale blue. "
                         "Off by default so the focus stays on occupied + unseen.")
    ap.add_argument("--hide-unseen", action="store_true",
                    help="hide the gray unseen voxels (default: shown). "
                         "Sub-sampled aggressively; turn off if perf is a problem.")
    ap.add_argument("--point-subsample", type=int, default=4,
                    help="render every Nth point in the empty/unseen clouds "
                         "(default 4; unseen forced to ≥8).")
    ap.add_argument("--with-tsdf", action="store_true",
                    help="also build a scene-wide heuristic TSDF mesh and overlay it")
    ap.add_argument("--column", action="append", default=[],
                    metavar="X,Y",
                    help="world-frame (x,y) to highlight via raycast_down. "
                         "Repeatable. Default on apple_drop: release + landing landmarks.")
    ap.add_argument("--animate-every", type=int, default=0,
                    metavar="N",
                    help="if N > 0, open the viewer FIRST and refresh the "
                         "geometry every N integrated frames so you can watch "
                         "the grid grow. 0 (default) = integrate all frames "
                         "first, then open a static viewer.")
    ap.add_argument("--auto-aabb", action="store_true",
                    help="crop the voxel-grid AABB to a tight box around the "
                         "camera's actually-reachable region for THIS trajectory. "
                         "Removes the geometrically-unreachable 'unseen' bulk so "
                         "the viz reflects the observable scene only. The default "
                         "YAML AABB is unaffected (live tracker keeps it).")
    ap.add_argument("--mask-arm", action="store_true",
                    help="zero out depth pixels under the projected Fetch arm "
                         "before integrating into the voxel grid, so the robot "
                         "itself doesn't pollute the occupancy. Uses "
                         "utils.fetch_arm_mask.ArmMaskBuilder.")
    ap.add_argument("--arm-dilate-px", type=int, default=8,
                    help="dilation kernel for the arm mask (only with --mask-arm; "
                         "default 8 px to cover convex-hull-edge slack).")
    ap.add_argument("--arm-urdf-path", default=None,
                    help="override URDF path for --mask-arm (default: utils.fetch_arm_mask.DEFAULT_URDF)")
    ap.add_argument("--arm-mesh-root", default=None,
                    help="override mesh root for --mask-arm (default: utils.fetch_arm_mask.DEFAULT_MESH_ROOT)")
    ap.add_argument("--no-viewer", action="store_true",
                    help="skip the interactive Open3D window (still prints stats)")
    args = ap.parse_args()

    # ── load config ──
    from ekf_tracker.configs import (
        load_config,
        build_voxel_observability_kwargs,
        build_voxel_integrate_kwargs,
    )
    cfg = load_config(args.config_path)
    vo_kwargs = build_voxel_observability_kwargs(cfg)
    integrate_kwargs = build_voxel_integrate_kwargs(cfg)

    # ── load trajectory data (needed before vo so --auto-aabb can fit) ──
    traj = args.trajectory
    ds_root = os.path.join(DATASET_DIR, traj)
    pose_path = os.path.join(ds_root, "pose_txt", "amcl_pose.txt")
    T_bc_path = os.path.join(ds_root, "pose_txt", "T_bc.txt")
    depth_dir = os.path.join(ds_root, "depth")

    slam_poses = _load_amcl_poses(pose_path)
    if not slam_poses:
        print(f"[err] no AMCL poses at {pose_path}", file=sys.stderr)
        return 2
    T_bc_map = _load_T_bc_poses(T_bc_path)
    if T_bc_map is None:
        print(f"[warn] no {T_bc_path} — using identity T_bc (camera == base, "
              f"voxel coverage will be wrong on Fetch).")

    n_frames = len(slam_poses)
    end = min(args.max_frame, n_frames)
    print(f"[traj] {traj}: integrating frames {args.start}..{end - 1} (step {args.step})")

    # ── optional --auto-aabb: crop the AABB to the camera's reach ──
    if args.auto_aabb:
        print("[auto-aabb] running probe-pass integration to fit AABB ...")
        new_min, new_max = _fit_aabb(
            slam_poses, T_bc_map, K_DEFAULT,
            depth_dir=depth_dir,
            vo_kwargs=vo_kwargs,
            integrate_kwargs=integrate_kwargs,
            start=args.start, end=end, step=args.step,
        )
        vo_kwargs = dict(vo_kwargs)
        vo_kwargs["workspace_aabb"] = (
            tuple(float(x) for x in new_min),
            tuple(float(x) for x in new_max),
        )
        print(f"[auto-aabb] fitted AABB to "
              f"{new_min.tolist()} .. {new_max.tolist()}  "
              f"(extent {(new_max - new_min).tolist()})")

    vo = VoxelObservability(**vo_kwargs)
    print(f"[voxel] aabb={vo.aabb_min.tolist()}..{vo.aabb_max.tolist()}  "
          f"voxel_size={vo.voxel_size}m  shape={vo.shape}  "
          f"n_min_hit={vo.n_min_hit}  n_min_pass={vo.n_min_pass}")

    # ── optional TSDF ──
    tsdf = None
    if args.with_tsdf:
        try:
            tsdf = _build_tsdf(vo.voxel_size,
                                max_range_m=integrate_kwargs.get("max_range_m", 5.0))
        except Exception as e:
            print(f"[warn] TSDF construction failed ({e}); skipping --with-tsdf overlay")
            tsdf = None

    # ── optional arm mask ──
    arm_builder = None
    joints_map = None
    if args.mask_arm:
        from utils.fetch_arm_mask import (
            ArmMaskBuilder, DEFAULT_URDF as _ARM_URDF,
            DEFAULT_MESH_ROOT as _ARM_MESH_ROOT,
        )
        joints_path = os.path.join(ds_root, "pose_txt", "joints_pose.json")
        if not os.path.exists(joints_path):
            print(f"[err] --mask-arm needs {joints_path}", file=sys.stderr)
            return 5
        with open(joints_path) as f:
            raw = json.load(f)
        joints_map = {int(k): v for k, v in raw.items()}
        urdf_path = args.arm_urdf_path or _ARM_URDF
        mesh_root = args.arm_mesh_root or _ARM_MESH_ROOT
        try:
            arm_builder = ArmMaskBuilder(
                urdf_path=urdf_path, mesh_root=mesh_root,
                K=K_DEFAULT, image_shape=(480, 640),
                dilate_px=args.arm_dilate_px,
            )
        except Exception as e:
            print(f"[err] ArmMaskBuilder construction failed: {e}", file=sys.stderr)
            return 5
        print(f"[mask-arm] enabled  ({len(arm_builder.meshes)} link meshes, "
              f"dilate_px={args.arm_dilate_px})")

    # ── columns to highlight ──
    columns: List[Tuple[float, float, str]] = []
    if args.column:
        for s in args.column:
            try:
                x, y = (float(t) for t in s.split(","))
                columns.append((x, y, f"col@({x:.2f},{y:.2f})"))
            except ValueError:
                print(f"[warn] could not parse --column {s!r}; expected 'x,y'")
    elif traj == "apple_drop":
        columns = list(DEFAULT_APPLE_DROP_COLUMNS)

    # ── viewer dispatch ──
    if args.no_viewer:
        # Headless: integrate, print stats + columns, return.
        n_done = 0
        for fr in range(args.start, end, args.step):
            if _integrate_frame(vo, depth_dir, fr, slam_poses, T_bc_map,
                                  integrate_kwargs, tsdf,
                                  arm_builder=arm_builder, joints_map=joints_map):
                n_done += 1
                if n_done % 100 == 0:
                    print(f"  ... {n_done} frames integrated")
        print(f"[traj] integrated {n_done} frames")
        _print_stats_and_columns(vo, columns)
        return 0

    try:
        import open3d as o3d
    except ImportError:
        print("[err] open3d not installed; install or pass --no-viewer", file=sys.stderr)
        return 3

    if args.animate_every > 0:
        _run_animated(o3d, vo, integrate_kwargs, slam_poses, T_bc_map,
                       depth_dir, args, columns, tsdf, n_frames,
                       arm_builder=arm_builder, joints_map=joints_map)
        _print_stats_and_columns(vo, columns)
        return 0

    # Static path: integrate all, then render once.
    n_done = 0
    for fr in range(args.start, end, args.step):
        if _integrate_frame(vo, depth_dir, fr, slam_poses, T_bc_map,
                              integrate_kwargs, tsdf,
                              arm_builder=arm_builder, joints_map=joints_map):
            n_done += 1
            if n_done % 100 == 0:
                print(f"  ... {n_done} frames integrated")
    print(f"[traj] integrated {n_done} frames")
    _print_stats_and_columns(vo, columns)
    _run_static(o3d, vo, args, columns, tsdf)
    return 0


def _print_stats_and_columns(vo, columns) -> None:
    stats = vo.stats()
    n_total = stats["n_total"]
    print(f"[stats] occupied={stats['n_occupied']:>9}  "
          f"empty={stats['n_empty']:>9}  "
          f"unseen={stats['n_unseen']:>9}  "
          f"total={n_total:>9}  "
          f"({stats['bytes_used'] / 1e6:.1f} MB)")
    print(f"[stats] occupied %={100 * stats['n_occupied'] / n_total:.2f}  "
          f"empty %={100 * stats['n_empty'] / n_total:.2f}  "
          f"unseen %={100 * stats['n_unseen'] / n_total:.2f}")
    for x, y, name in columns:
        z_top = float(vo.aabb_max[2])
        result = vo.raycast_down((x, y, z_top))
        print(f"[column] {name:<18}  state={result.column_state:<13}  "
              f"surface_z={result.surface_z!s:<8}  "
              f"first_unseen_z={result.first_unseen_z!s:<8}  "
              f"floor_z={result.floor_z:.3f}  n_states={len(result.states)}")


if __name__ == "__main__":
    sys.exit(main())
