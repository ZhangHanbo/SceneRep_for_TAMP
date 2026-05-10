"""Gravity-aware one-shot predict for the EKF on object release.

Called once when the gripper releases an object: simulates the
free-fall + bounce + roll dispersion using a parametric model and the
voxel observability grid (`perception.voxel_observability`) to find
the supporting surface below the object.

The output is an updated SE(3) mean and 6×6 covariance (in `[v, ω]`
tangent ordering matching `pose_update/ekf_se3.py`). Subsequent frames
revert to the standard static-predict — this function is invoked at
the release-transition edge only.

Parametric model (full derivation in
`docs/ekf_tracker/latex/bernoulli_ekf.tex`, "Gravity-aware predict at release"):

    σ_xy² = σ_bounce² + σ_release_v² + σ_shape²
        σ_bounce    = ε_roughness · sqrt(2 g h) / max(0.1, 1 - e)
        σ_release_v = 0.5 · ‖v_xy‖ · t_fall
        σ_shape     = r_obj · shape_footprint_factor(shape)
    σ_z       = 0.5 · r_obj · (1 - e/2)              (clean surface)
    σ_yaw     = π · (1 - exp(-h · (1 - e²) / r_obj))
    σ_roll    = σ_pitch = arctan(r_obj / max(h_stable, 0.01))

Voxel-driven adjustment to (h, σ_z), in priority order:

    release_visible:    the release pose itself projects to a valid depth
                        pixel and the depth matches the release-pose z_cam
                        within `tol_release_visible_m`. The apple is still
                        in view; perception will handle it on the next
                        frame. We return identity (no covariance inflation).
    landing_visible:    the gravity-predicted landing XY projects to a
                        valid depth pixel. We snap landing_z to the world-z
                        of the visible surface and tighten σ_z to a few cm.
    neighbourhood_median: column under the predicted XY is `all_unseen`,
                        but ≥ `min_neighbour_surfaces_for_median` columns
                        within `r_neighbourhood_m` have a known surface_z.
                        We use the median surface_z + a spread-driven σ_z.
    hit_occupied:       column hit a surface cleanly.
    mixed_unseen:       column hit a surface, but with unseen voxels above.
    all_unseen:         column never observed (fallback to midpoint).
    all_empty:          column observed empty all the way down.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Tuple

import numpy as np

from utils.object_dynamics import (
    ObjectDynamicsProperty,
    shape_footprint_factor,
)
from perception.voxel_observability import VoxelObservability


# Default surface-roughness coefficient in σ_bounce. Units: m / (m·s⁻¹) →
# i.e. a 1 m/s impact velocity produces ~5 mm of horizontal scatter per
# bounce. Order-of-magnitude; tune empirically on the apple_drop trajectory.
EPS_ROUGHNESS_DEFAULT = 5.0e-3


@dataclass
class GravityPredictInfo:
    """Diagnostic record returned by `predict_landing_pose`.

    Surfaced via the orchestrator's gravity-predict log so visualizers
    and integration tests can inspect the parametric components.
    """
    column_state: str
    drop_height_m: float
    surface_z: Optional[float]
    first_unseen_z: Optional[float]
    floor_z: float
    sigma_xy: float
    sigma_z: float
    sigma_yaw: float
    sigma_roll: float
    sigma_pitch: float
    sigma_bounce: float
    sigma_release_v: float
    sigma_shape: float
    landing_z: float
    skipped: bool = False
    skip_reason: str = ""
    # Fix A diagnostic: how many neighbourhood columns yielded a surface_z
    # for the all_unseen → neighbourhood_median override.
    n_neighbour_surfaces: int = 0
    # Fix B diagnostic: depth read at the projected landing pixel (None if
    # Fix B did not apply or the pixel was out of FOV / invalid).
    landing_visible_depth_m: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "column_state": self.column_state,
            "drop_height_m": float(self.drop_height_m),
            "surface_z": (None if self.surface_z is None
                          else float(self.surface_z)),
            "first_unseen_z": (None if self.first_unseen_z is None
                                else float(self.first_unseen_z)),
            "floor_z": float(self.floor_z),
            "sigma_xy": float(self.sigma_xy),
            "sigma_z": float(self.sigma_z),
            "sigma_yaw": float(self.sigma_yaw),
            "sigma_roll": float(self.sigma_roll),
            "sigma_pitch": float(self.sigma_pitch),
            "sigma_bounce": float(self.sigma_bounce),
            "sigma_release_v": float(self.sigma_release_v),
            "sigma_shape": float(self.sigma_shape),
            "landing_z": float(self.landing_z),
            "skipped": bool(self.skipped),
            "skip_reason": str(self.skip_reason),
            "n_neighbour_surfaces": int(self.n_neighbour_surfaces),
            "landing_visible_depth_m": (
                None if self.landing_visible_depth_m is None
                else float(self.landing_visible_depth_m)),
        }


# ─────────────────────────────────────────────────────────────────────
#  Visibility / neighbourhood helpers (Fix A + Fix B)
# ─────────────────────────────────────────────────────────────────────

def _surface_world_at_xyz(p_world: np.ndarray,
                            T_cw: np.ndarray,
                            K: np.ndarray,
                            depth: np.ndarray,
                            image_shape: Tuple[int, int],
                            *,
                            min_d: float,
                            max_d: float
                            ) -> Optional[Tuple[np.ndarray, float, float]]:
    """Project world-frame point ``p_world`` into the camera and look up the
    visible surface at that pixel.

    Returns ``(p_surface_world, depth_value, p_query_z_cam)`` if the
    projection is in-bounds with valid depth, else ``None``.

    ``p_surface_world`` is the world-frame position of whatever surface the
    pixel ray hits (back-projected from the depth sample). ``p_query_z_cam``
    is the camera-frame z of the *query* point (so the caller can decide
    whether the depth matches the query, e.g. for release-pose visibility).
    """
    H, W = image_shape
    p_h = np.asarray([p_world[0], p_world[1], p_world[2], 1.0],
                      dtype=np.float64)
    T_cb = np.linalg.inv(np.asarray(T_cw, dtype=np.float64))
    p_cam = (T_cb @ p_h)[:3]
    if not np.isfinite(p_cam[2]) or p_cam[2] < min_d:
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * float(p_cam[0]) / float(p_cam[2]) + cx
    v = fy * float(p_cam[1]) / float(p_cam[2]) + cy
    ui = int(round(u))
    vi = int(round(v))
    if not (0 <= ui < W and 0 <= vi < H):
        return None
    d = float(depth[vi, ui])
    if not (min_d < d < max_d) or not np.isfinite(d):
        return None
    # Back-project the depth pixel (ui, vi, d) to camera frame, then world.
    x_cam = (float(ui) - cx) * d / fx
    y_cam = (float(vi) - cy) * d / fy
    p_surface_cam = np.asarray([x_cam, y_cam, d, 1.0], dtype=np.float64)
    p_surface_world = (np.asarray(T_cw, dtype=np.float64)
                        @ p_surface_cam)[:3]
    return p_surface_world, d, float(p_cam[2])


def _neighbourhood_surfaces(
    voxel_obs: VoxelObservability,
    centre_xy: Tuple[float, float],
    z_top: float,
    *,
    radius_m: float,
    n_samples: int,
    max_distance_m: float,
    floor_z: float,
) -> Tuple[int, Optional[float], float]:
    """Raycast down from N points on a circle around ``centre_xy``.

    Returns ``(n_with_surface, median_surface_z, surface_spread_m)``. If
    no neighbour found a surface, returns ``(0, None, 0.0)``.

    ``surface_spread_m`` is half the IQR of the surface_z values (quick
    approximation of one σ for the column z-uncertainty).
    """
    if n_samples <= 0 or radius_m <= 0.0:
        return 0, None, 0.0
    surfs = []
    for i in range(int(n_samples)):
        theta = 2.0 * np.pi * (i / float(n_samples))
        x = float(centre_xy[0]) + radius_m * float(np.cos(theta))
        y = float(centre_xy[1]) + radius_m * float(np.sin(theta))
        cast = voxel_obs.raycast_down(
            (x, y, z_top),
            max_distance_m=float(max_distance_m),
            floor_z=float(floor_z),
        )
        if cast.surface_z is not None:
            surfs.append(float(cast.surface_z))
    if not surfs:
        return 0, None, 0.0
    arr = np.asarray(surfs, dtype=np.float64)
    median = float(np.median(arr))
    if arr.size >= 4:
        q75 = float(np.quantile(arr, 0.75))
        q25 = float(np.quantile(arr, 0.25))
        spread = 0.5 * max(0.0, q75 - q25)
    else:
        spread = float(arr.max() - arr.min()) * 0.5
    return len(surfs), median, spread


# ─────────────────────────────────────────────────────────────────────
#  Main entry
# ─────────────────────────────────────────────────────────────────────

def predict_landing_pose(
    T_release: np.ndarray,
    P_release: np.ndarray,
    voxel_obs: Optional[VoxelObservability],
    dyn: ObjectDynamicsProperty,
    *,
    gravity: float,
    workspace_floor_z: float,
    eps_roughness: float,
    max_drop_m: float,
    shape_footprint_factors: Mapping[str, float],
    v_release_world: Optional[np.ndarray] = None,
    live_object_voxels: Iterable[Tuple[float, float, float, float]] = (),
    # Fix A — neighbourhood-median for all_unseen.
    r_neighbourhood_m: float = 0.30,
    n_neighbourhood_samples: int = 8,
    min_neighbour_surfaces_for_median: int = 3,
    # Fix B — visibility-based override.
    K: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    T_cw: Optional[np.ndarray] = None,
    image_shape: Optional[Tuple[int, int]] = None,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
    tol_release_visible_m: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, GravityPredictInfo]:
    """One-shot release-time predict: returns settled pose + covariance.

    Args:
        T_release: (4, 4) world-frame pose at release (the object is
            still at the gripper, falling has not started yet).
        P_release: (6, 6) covariance at release in `[v, ω]` tangent
            ordering (translation first then rotation).
        voxel_obs: scene voxel-observability grid. None → skip and
            return identity update (caller falls back to standard
            static predict).
        dyn: dynamics property (resolved via
            `utils.object_dynamics.lookup_dynamics(label)`).
        gravity: gravitational acceleration (m/s²).
        workspace_floor_z: world-frame z below which the column is
            assumed to be void.
        eps_roughness: surface-roughness coefficient in the bounce
            model (m / (m·s⁻¹)).
        max_drop_m: maximum drop distance to consider for the raycast.
        v_release_world: (3,) world-frame velocity at release; defaults
            to zero. Only the horizontal component is used.
        live_object_voxels: iterable of (x, y, z, radius) for other
            tracked objects to overlay as OCCUPIED for this query.

        # ─── Fix A: neighbourhood-median fallback for all_unseen ───
        r_neighbourhood_m: XY radius for neighbouring columns.
        n_neighbourhood_samples: number of evenly-spaced angular samples.
        min_neighbour_surfaces_for_median: minimum hit-count to trust
            the median (else fall through to original midpoint logic).

        # ─── Fix B: visibility-based override ───
        K, depth, T_cw, image_shape: per-frame camera intrinsics + depth
            map + camera-to-world extrinsic + (H, W). All four are
            required to enable the override; if any is None, Fix B is
            skipped (Fix A still applies).
        min_depth_m, max_depth_m: range for "valid" depth pixel.
        tol_release_visible_m: depth-vs-release-z_cam tolerance for
            declaring the release pose still in view.

    Returns:
        T_land: (4, 4) predicted post-settling pose.
        P_land: (6, 6) inflated covariance.
        info: `GravityPredictInfo` with diagnostics.
    """
    T_release = np.asarray(T_release, dtype=np.float64)
    P_release = np.asarray(P_release, dtype=np.float64)
    if T_release.shape != (4, 4) or P_release.shape != (6, 6):
        raise ValueError("T_release must be (4, 4) and P_release (6, 6)")

    # Skip safely when no voxel grid is available.
    if voxel_obs is None:
        info = GravityPredictInfo(
            column_state="skipped", drop_height_m=0.0,
            surface_z=None, first_unseen_z=None,
            floor_z=workspace_floor_z,
            sigma_xy=0.0, sigma_z=0.0, sigma_yaw=0.0,
            sigma_roll=0.0, sigma_pitch=0.0,
            sigma_bounce=0.0, sigma_release_v=0.0, sigma_shape=0.0,
            landing_z=float(T_release[2, 3]),
            skipped=True, skip_reason="voxel_obs is None",
        )
        return T_release.copy(), P_release.copy(), info

    fix_b_available = (K is not None and depth is not None
                       and T_cw is not None and image_shape is not None)

    # ── Fix B (1/2): release-pose visible? ──
    # If the release pose itself currently projects onto a valid depth
    # pixel whose depth matches the release-pose camera-z within tolerance,
    # the apple is still in view — perception will update it normally.
    # Skip the gravity prior; return identity.
    if fix_b_available:
        rel_xyz = T_release[:3, 3]
        probe = _surface_world_at_xyz(
            rel_xyz, T_cw, K, depth, image_shape,
            min_d=min_depth_m, max_d=max_depth_m,
        )
        if probe is not None:
            _, d_release_pixel, p_release_z_cam = probe
            if abs(d_release_pixel - p_release_z_cam) < tol_release_visible_m:
                info = GravityPredictInfo(
                    column_state="release_visible",
                    drop_height_m=0.0,
                    surface_z=None, first_unseen_z=None,
                    floor_z=float(workspace_floor_z),
                    sigma_xy=0.0, sigma_z=0.0, sigma_yaw=0.0,
                    sigma_roll=0.0, sigma_pitch=0.0,
                    sigma_bounce=0.0, sigma_release_v=0.0, sigma_shape=0.0,
                    landing_z=float(rel_xyz[2]),
                    skipped=True, skip_reason="release pose visible",
                    n_neighbour_surfaces=0,
                    landing_visible_depth_m=float(d_release_pixel),
                )
                return T_release.copy(), P_release.copy(), info

    # Object bottom in world frame: T_release applied to (0, 0, -r) so
    # we raycast from below the centre of mass.
    bottom_local = np.array([0.0, 0.0, -float(dyn.radius_m), 1.0])
    p_bottom = (T_release @ bottom_local)[:3]

    cast = voxel_obs.raycast_down(
        p_bottom,
        max_distance_m=float(max_drop_m),
        floor_z=float(workspace_floor_z),
        live_object_voxels=tuple(live_object_voxels),
    )

    # Resolve drop height h, landing z, and column-driven σ_z floor.
    column_state = cast.column_state
    h: float
    sigma_z_floor: float
    landing_z: float
    n_neighbour_hits = 0
    landing_visible_depth: Optional[float] = None

    if column_state == "hit_occupied":
        landing_z = float(cast.surface_z) + dyn.radius_m
        h = max(0.0, float(p_bottom[2]) - float(cast.surface_z))
        sigma_z_floor = 0.0
    elif column_state == "mixed_unseen":
        landing_z = float(cast.surface_z) + dyn.radius_m
        h = max(0.0, float(p_bottom[2]) - float(cast.surface_z))
        sigma_z_floor = max(0.0, (float(cast.first_unseen_z)
                                  - float(cast.surface_z)) / 2.0)
    elif column_state == "all_unseen":
        # ── Fix A: try neighbourhood-median surface ──
        n_neighbour_hits, neigh_median, neigh_spread = _neighbourhood_surfaces(
            voxel_obs,
            centre_xy=(float(p_bottom[0]), float(p_bottom[1])),
            z_top=float(p_bottom[2]),
            radius_m=float(r_neighbourhood_m),
            n_samples=int(n_neighbourhood_samples),
            max_distance_m=float(max_drop_m),
            floor_z=float(workspace_floor_z),
        )
        if (neigh_median is not None
                and n_neighbour_hits >= int(min_neighbour_surfaces_for_median)):
            column_state = "neighbourhood_median"
            landing_z = float(neigh_median) + dyn.radius_m
            h = max(0.0, float(p_bottom[2]) - float(neigh_median))
            # σ_z absorbs the IQR-half spread plus a voxel-scale floor.
            sigma_z_floor = max(float(neigh_spread),
                                 2.0 * float(voxel_obs.voxel_size))
        elif fix_b_available:
            # ── Fix B (2/2): visibility at predicted-landing XY ──
            # Probe the camera at the apple-bottom XY column. If the
            # camera currently sees a surface there, snap to it.
            probe = _surface_world_at_xyz(
                np.array([p_bottom[0], p_bottom[1], float(p_bottom[2])]),
                T_cw, K, depth, image_shape,
                min_d=min_depth_m, max_d=max_depth_m,
            )
            if probe is not None:
                p_surface_w, d_pixel, _ = probe
                landing_visible_depth = float(d_pixel)
                column_state = "landing_visible"
                landing_z = float(p_surface_w[2]) + dyn.radius_m
                h = max(0.0, float(p_bottom[2]) - float(p_surface_w[2]))
                sigma_z_floor = max(2.0 * float(voxel_obs.voxel_size), 0.02)
            else:
                # Final fallback: original midpoint logic.
                first_unseen = (float(cast.first_unseen_z)
                                if cast.first_unseen_z is not None
                                else float(p_bottom[2]))
                midpoint = 0.5 * (first_unseen + float(cast.floor_z))
                landing_z = midpoint + dyn.radius_m
                h = max(0.0, float(p_bottom[2]) - midpoint)
                sigma_z_floor = max(0.0,
                                     (first_unseen - float(cast.floor_z)) / 2.0)
        else:
            # Original midpoint logic (preserved for back-compat callers
            # that don't pass K/depth/T_cw).
            first_unseen = (float(cast.first_unseen_z)
                            if cast.first_unseen_z is not None
                            else float(p_bottom[2]))
            midpoint = 0.5 * (first_unseen + float(cast.floor_z))
            landing_z = midpoint + dyn.radius_m
            h = max(0.0, float(p_bottom[2]) - midpoint)
            sigma_z_floor = max(0.0,
                                 (first_unseen - float(cast.floor_z)) / 2.0)
    else:  # 'all_empty' — the column is observed empty all the way down.
        landing_z = float(cast.floor_z) + dyn.radius_m
        h = max(0.0, float(p_bottom[2]) - float(cast.floor_z))
        sigma_z_floor = dyn.radius_m

    # Already-settled fast path: object started on/below the surface.
    already_settled = h <= 1e-3

    # Parametric bounce/roll/shape model.
    e = float(dyn.e)
    one_minus_e = max(0.1, 1.0 - e)  # numerical guard
    v_impact = np.sqrt(max(0.0, 2.0 * gravity * h))
    sigma_bounce = eps_roughness * v_impact / one_minus_e
    sigma_shape = dyn.radius_m * shape_footprint_factor(
        dyn.shape, factors=shape_footprint_factors)
    if v_release_world is None:
        v_release_xy_norm = 0.0
    else:
        v = np.asarray(v_release_world, dtype=np.float64).ravel()
        v_release_xy_norm = float(np.linalg.norm(v[:2]))
    t_fall = np.sqrt(max(0.0, 2.0 * h / gravity))
    sigma_release_v = 0.5 * v_release_xy_norm * t_fall

    sigma_xy = float(np.sqrt(
        sigma_bounce * sigma_bounce
        + sigma_release_v * sigma_release_v
        + sigma_shape * sigma_shape
    ))
    if already_settled:
        sigma_xy = sigma_shape  # tiny settling jitter only.
    sigma_z = float(np.sqrt(
        (0.5 * dyn.radius_m * (1.0 - e * 0.5)) ** 2
        + sigma_z_floor ** 2
    ))
    if already_settled:
        sigma_z = 0.5 * dyn.radius_m
    sigma_yaw = float(np.pi * (1.0 - np.exp(
        -h * (1.0 - e * e) / max(dyn.radius_m, 1e-3))))
    if already_settled:
        sigma_yaw = 0.05  # ~3°
    h_stable = max(landing_z - cast.floor_z, 0.01)
    sigma_roll = sigma_pitch = float(np.arctan(dyn.radius_m / h_stable))

    # Build T_land: keep xy and rotation; replace z.
    T_land = T_release.copy()
    if not already_settled:
        T_land[2, 3] = landing_z

    # Inflate covariance additively. Tangent ordering is [v, ω] per
    # pose_update/ekf_se3.py: Σ block-diag(σ_x², σ_y², σ_z², σ_roll²,
    # σ_pitch², σ_yaw²).
    extra = np.diag([
        sigma_xy ** 2, sigma_xy ** 2, sigma_z ** 2,
        sigma_roll ** 2, sigma_pitch ** 2, sigma_yaw ** 2,
    ])
    P_land = P_release + extra

    info = GravityPredictInfo(
        column_state=column_state,
        drop_height_m=float(h),
        surface_z=cast.surface_z,
        first_unseen_z=cast.first_unseen_z,
        floor_z=float(cast.floor_z),
        sigma_xy=float(sigma_xy),
        sigma_z=float(sigma_z),
        sigma_yaw=float(sigma_yaw),
        sigma_roll=float(sigma_roll),
        sigma_pitch=float(sigma_pitch),
        sigma_bounce=float(sigma_bounce),
        sigma_release_v=float(sigma_release_v),
        sigma_shape=float(sigma_shape),
        landing_z=float(landing_z),
        n_neighbour_surfaces=int(n_neighbour_hits),
        landing_visible_depth_m=landing_visible_depth,
    )
    return T_land, P_land, info
