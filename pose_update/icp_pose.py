"""
Per-object pose estimator with three switchable backends.

Methods
═══════

1. **centroid**  — translation-only back-projected centroid.
   Cheapest, no rotation, fixed R_icp. What the original integration
   test used. Useful as a lower-bound reference.

2. **icp_chain** — ICP against a first-frame reference cloud, with the
   ICP initialization taken from the PREVIOUS frame's result (rotation
   warmstart) and the current centroid (translation reset). This is the
   default "continuous-tracking" mode. Can drift because each frame's
   ICP error feeds the next frame's initialization.

3. **icp_anchor** — ICP against the first-frame reference cloud, with
   a fully state-free, fully camera-frame init:
       rotation:    identity.
       translation: current centroid of masked depth.
   No T_wb anywhere. No prev_T_co. Each frame's ICP is independent of
   both localization and past ICP results.

   Semantics: this is the strict interpretation of the uncertainty
   decomposition — localization (Σ_wb) lives in the filter's world-
   frame state; the OBSERVATION is purely a camera-frame quantity
   produced from the CURRENT observation alone. We only care about
   the camera-frame TRANSLATION; any rotation-drift behaviour is the
   filter's problem, not the observation's.


Shared interface
────────────────

    est = PoseEstimator(K, method="icp_chain")   # or "centroid", "icp_anchor"
    for each detection d:
        T_co, R_icp, fitness, rmse = est.estimate(
            oid=d["id"], mask=d["mask"], depth=depth)
        if T_co is None:
            continue                             # dropped (ICP fitness gate)
        d["T_co"], d["R_icp"], d["fitness"], d["rmse"] = ...

None of the three methods reads T_wb / camera-to-world. Layer 1 (world-
frame localization) and Layer 2 (camera-frame object pose from ICP) are
strictly independent — composition happens in the filter, not here.

All three methods share the same fitness gate: if ICP fitness < MIN_FITNESS
or RMSE > MAX_RMSE, return `(None, None, fitness, rmse)` so the caller
drops the detection. (centroid doesn't run ICP and is never rejected.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import open3d as o3d


METHODS = (
    "centroid",
    "icp_chain",
    "icp_anchor",
    "icp_chain_strict",   # NEW: prev_T_co full init, no centroid reset
    "icp_anchor_strict",  # NEW: first-frame T_co full init, no centroid reset
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _back_project(mask: np.ndarray,
                  depth: np.ndarray,
                  K: np.ndarray,
                  min_depth: float = 0.1,
                  max_depth: float = 5.0,
                  min_points: int = 30) -> Optional[np.ndarray]:
    """Back-project masked, valid-depth pixels to (N, 3) camera-frame points."""
    ys, xs = np.where(mask > 0)
    if len(xs) < min_points:
        return None
    ds = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(ds) & (ds > min_depth) & (ds < max_depth)
    if valid.sum() < min_points:
        return None
    xs, ys, ds = xs[valid], ys[valid], ds[valid]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    X = (xs - cx) * ds / fx
    Y = (ys - cy) * ds / fy
    Z = ds
    return np.stack([X, Y, Z], axis=1)


def _voxelize(pts: np.ndarray, voxel: float) -> o3d.geometry.PointCloud:
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc.voxel_down_sample(voxel)


# ─────────────────────────────────────────────────────────────────────
# Per-object state
# ─────────────────────────────────────────────────────────────────────

@dataclass
class _ObjRef:
    """Per-object state — strictly camera-frame. No world-frame info."""
    ref_points: np.ndarray        # (M, 3) in object-local frame (centroid-centered)
    prev_T_co: np.ndarray         # (4, 4) last successful ICP result (chain variants)
    first_T_co: np.ndarray        # (4, 4) first-frame T_co (anchor_strict init)
    obj_radius: float             # metres — for rotation uncertainty scaling
    n_frames_tracked: int


# ─────────────────────────────────────────────────────────────────────
# Unified estimator
# ─────────────────────────────────────────────────────────────────────

class PoseEstimator:
    """Per-object T_co estimator with three switchable backends.

    Args:
        K:        (3, 3) camera intrinsics.
        method:   one of "centroid", "icp_chain", "icp_anchor".
    """

    # Open3D ICP parameters.
    VOXEL_SIZE = 0.005       # 5 mm
    ICP_THRESHOLD = 0.020    # 2 cm correspondence threshold
    ICP_MAX_ITER = 30

    # Fitness gate (strict).  Below this the observation is dropped.
    MIN_FITNESS = 0.90
    MAX_RMSE = 0.015         # 15 mm safety net

    # Covariance floors (variances).
    TRANS_VAR_FLOOR = 1e-6
    ROT_VAR_FLOOR = 1e-4

    # Hand-chosen constant covariance for the centroid method (no ICP
    # diagnostics available).
    CENTROID_R_ICP = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3])

    def __init__(self, K: np.ndarray, method: str = "icp_chain"):
        if method not in METHODS:
            raise ValueError(
                f"method must be one of {METHODS}, got {method!r}")
        self.K = np.asarray(K, dtype=np.float64)
        self.method = method
        self._refs: Dict[int, _ObjRef] = {}

    # --------------------------------------------------------------- #
    #  Public entry point
    # --------------------------------------------------------------- #

    def estimate(self,
                 oid: int,
                 mask: np.ndarray,
                 depth: np.ndarray,
                 T_cw: Optional[np.ndarray] = None,
                 ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray],
                            float, float]:
        """Estimate (T_co, R_icp) for one object detection at one frame.

        Args:
            oid:   object id (used to key per-object reference state).
            mask:  (H, W) bool detection mask.
            depth: (H, W) float32 depth in metres.
            T_cw:  kept in the signature for API parity; NONE of the
                   three methods currently consume it. The anchor's
                   decomposition premise says ICP is strictly
                   camera-frame and independent of localization.

        Returns:
            T_co:    (4, 4) or None (None → drop this detection).
            R_icp:   (6, 6) or None.
            fitness: ICP inlier fraction or 1.0 (centroid).
            rmse:    ICP inlier RMSE or 0.0 (centroid).
        """
        if self.method == "centroid":
            return self._estimate_centroid(mask, depth)
        return self._estimate_icp(oid, mask, depth, init_policy=self.method)

    # --------------------------------------------------------------- #
    #  Method 1: centroid
    # --------------------------------------------------------------- #

    def _estimate_centroid(self,
                           mask: np.ndarray,
                           depth: np.ndarray,
                           ) -> Tuple[Optional[np.ndarray],
                                      Optional[np.ndarray], float, float]:
        pts = _back_project(mask, depth, self.K)
        if pts is None:
            return None, None, 0.0, 0.0
        T_co = np.eye(4, dtype=np.float64)
        T_co[:3, 3] = pts.mean(axis=0)
        return T_co, self.CENTROID_R_ICP.copy(), 1.0, 0.0

    # --------------------------------------------------------------- #
    #  Method 2 + 3: ICP (chain or anchor init)
    # --------------------------------------------------------------- #

    def _estimate_icp(self,
                      oid: int,
                      mask: np.ndarray,
                      depth: np.ndarray,
                      init_policy: str,
                      ) -> Tuple[Optional[np.ndarray],
                                 Optional[np.ndarray], float, float]:
        pts_cam = _back_project(mask, depth, self.K)
        if pts_cam is None:
            return None, None, 0.0, 0.0

        centroid_now = pts_cam.mean(axis=0)

        # ── First observation: anchor object-local frame at the centroid ─
        if oid not in self._refs:
            ref_points = pts_cam - centroid_now
            obj_radius = max(float(np.linalg.norm(ref_points, axis=1).max()),
                              0.03)
            T_co_init = np.eye(4, dtype=np.float64)
            T_co_init[:3, 3] = centroid_now
            self._refs[oid] = _ObjRef(
                ref_points=ref_points,
                prev_T_co=T_co_init.copy(),
                first_T_co=T_co_init.copy(),
                obj_radius=obj_radius,
                n_frames_tracked=0,
            )
            R_icp = np.diag([5e-5]*3 + [1e-2]*3)
            return T_co_init, R_icp, 1.0, 0.0

        ref = self._refs[oid]

        # ── Initial guess per method ───────────────────────────────────
        # No T_wb in any branch: localization lives in the filter, visual
        # observation is strictly camera-frame.
        if init_policy == "icp_chain":
            # Previous rotation, current-centroid translation.
            init_T = ref.prev_T_co.copy()
            init_T[:3, 3] = centroid_now
        elif init_policy == "icp_anchor":
            # Stateless: identity rotation, current-centroid translation.
            init_T = np.eye(4, dtype=np.float64)
            init_T[:3, 3] = centroid_now
        elif init_policy == "icp_chain_strict":
            # Pure chain: previous T_co (BOTH rotation AND translation).
            # No centroid reset — translation also chains forward.
            init_T = ref.prev_T_co.copy()
        elif init_policy == "icp_anchor_strict":
            # Pure anchor: first-frame T_co (BOTH rotation AND translation).
            # Init never updates from the first observation. Truly stateless
            # across all later frames — the SAME init is used every time.
            init_T = ref.first_T_co.copy()
        else:
            raise ValueError(f"unknown init_policy {init_policy!r}")

        # ── Run ICP ────────────────────────────────────────────────────
        src = _voxelize(ref.ref_points, self.VOXEL_SIZE)
        tgt = _voxelize(pts_cam, self.VOXEL_SIZE)
        result = o3d.pipelines.registration.registration_icp(
            src, tgt,
            self.ICP_THRESHOLD,
            init_T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=self.ICP_MAX_ITER),
        )
        T_co = np.asarray(result.transformation, dtype=np.float64)
        fitness = float(result.fitness)
        rmse = float(result.inlier_rmse)

        icp_ok = (fitness >= self.MIN_FITNESS
                  and rmse <= self.MAX_RMSE
                  and np.isfinite(T_co).all())
        if not icp_ok:
            # Refresh chain init so the next frame has a better seed;
            # anchor init is state-free so this is a no-op for it.
            refreshed = ref.prev_T_co.copy()
            refreshed[:3, 3] = centroid_now
            ref.prev_T_co = refreshed
            return None, None, fitness, rmse

        # ── Data-driven R_icp ──────────────────────────────────────────
        fitness_scale = min(5.0, 0.5 / max(fitness, 1e-3))
        trans_var = max(rmse**2, self.TRANS_VAR_FLOOR) * fitness_scale
        rot_var = max((rmse / ref.obj_radius)**2,
                       self.ROT_VAR_FLOOR) * fitness_scale
        R_icp = np.diag([trans_var]*3 + [rot_var]*3)

        ref.prev_T_co = T_co.copy()
        ref.n_frames_tracked += 1
        return T_co, R_icp, fitness, rmse

    # --------------------------------------------------------------- #

    def reset(self, oid: Optional[int] = None) -> None:
        """Drop per-object cache."""
        if oid is None:
            self._refs.clear()
        else:
            self._refs.pop(oid, None)


# ─────────────────────────────────────────────────────────────────────
# Backward-compatibility alias
# ─────────────────────────────────────────────────────────────────────

# Keep the old import path working while we transition visualize_pipeline.
ICPPoseEstimator = PoseEstimator
