"""
Gaussian-backend state for Layer 2.

When the low-level SLAM returns a single-Gaussian posterior
`PoseEstimate(T_wb, Σ_wb)`, this module tracks objects in the ROBOT BASE
FRAME, keeping the filter state independent of Σ_wb.

Why base frame (derivation)
───────────────────────────
The observation model in world frame is
    z = h(x, o) + v,   h(x, o) = T_wb⁻¹ · T_wo,   v ~ N(0, R_icp).
Marginalizing over x = T_wb ~ N(μ_wb, Σ_wb) and folding the linearized
contribution into R (the "Paper 1" recipe) gives
    R_eff = H_x Σ_wb H_xᵀ + R_icp.
That is correct for a SINGLE frame. Across frames it treats Σ_wb as
i.i.d., which is false — x_t and x_{t+1} are nearly the same random
variable. Recursive fusion under that fiction drives Σ_wo below Σ_wb,
which is physically impossible for a static object.

Base-frame storage removes Σ_wb from the recursion. With a known fixed
camera-to-base T_bc from kinematics, the observation becomes
    T_bo_meas = T_bc · T_co_meas,   noise Ad(T_bc) · R_icp · Ad(T_bc)ᵀ,
which is deterministic in x. The EKF state (μ_bo, Σ_bo) evolves without
ever touching Σ_wb. Σ_wb enters only once, at output time, via the
composition
    T_wo = T_wb · T_bo,   Σ_wo = Ad(T_bo⁻¹) · Σ_wb · Ad(·)ᵀ + Σ_bo,
which lower-bounds Σ_wo by the projected Σ_wb. Correct physics.

Regimes covered
───────────────
* Static object: base-frame pose changes as the base moves, but its
  world-frame pose is constant. We propagate μ_bo(t) = inv(ΔT_wb) · μ_bo(t−1)
  each frame (ΔT_wb from consecutive SLAM means). Σ_bo inflates by a
  phase-aware Q. Uncertainty in ΔT_wb is left for the output composition
  to absorb (it never enters the recursion).

* Manipulated object: rigidly attached to the gripper.
  T_bo(t) = ΔT_bg · T_bo(t−1), with ΔT_bg = T_bg(t)·inv(T_bg(t−1)) from
  proprioception. This predict is *local* — no T_wb coupling — which is
  cleaner than the particle version's T_wb(t)·ΔT_bg·inv(T_wb(t−1)) form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np

from pose_update.ekf_se3 import (
    se3_exp, se3_log, se3_adjoint, ekf_predict,
)
from pose_update.slam_interface import (
    PoseEstimate, ParticlePose, as_gaussian,
)


# ─────────────────────────────────────────────────────────────────────
# Per-object belief (single Gaussian, base frame)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GaussianObjectBelief:
    """Gaussian SE(3) belief about one object, in ROBOT BASE frame.

    Attributes:
        mu_bo:  (4, 4) SE(3) mean  (object-in-base).
        cov_bo: (6, 6) covariance in se(3) tangent at mu_bo, [v, ω] ordering.
    """
    mu_bo: np.ndarray
    cov_bo: np.ndarray


# ─────────────────────────────────────────────────────────────────────
# GaussianState
# ─────────────────────────────────────────────────────────────────────

class GaussianState:
    """Per-object EKFs in base frame. Σ_wb enters only at output time.

    The state holds the *current* SLAM posterior (μ_wb, Σ_wb), the
    *previous* one (used by `predict_static` to propagate base motion),
    and a dict of per-object base-frame beliefs. Object-level metadata
    (labels, phase state) is an orchestrator concern — this class is
    the pure filter engine.

    If the low-level SLAM returns a ParticlePose, we collapse it to a
    Gaussian via `to_gaussian()` at ingest — this is the Gaussian
    pipeline, so multimodality is intentionally dropped. If you want
    multimodality preserved, use `RBPFState` instead.
    """

    _LOG_EPS = -1e18

    def __init__(self, T_bc: Optional[np.ndarray] = None):
        """Args:
            T_bc: (4, 4) camera-in-base transform. If None, assumed to be
                  identity (camera frame == base frame). For a Fetch-style
                  head-mounted camera this is a fixed constant derived from
                  the URDF.
        """
        self.objects: Dict[int, GaussianObjectBelief] = {}

        # Current & previous SLAM posterior
        self.T_wb: Optional[np.ndarray] = None
        self.Sigma_wb: Optional[np.ndarray] = None
        self.prev_T_wb: Optional[np.ndarray] = None

        # Camera-in-base (constant). Used to transform T_co into base frame.
        self.T_bc: np.ndarray = (
            np.eye(4, dtype=np.float64) if T_bc is None
            else np.asarray(T_bc, dtype=np.float64).copy()
        )
        self._Ad_bc: np.ndarray = se3_adjoint(self.T_bc)

    # --------------------------------------------------------------- #
    #  Inspection
    # --------------------------------------------------------------- #

    @property
    def initialized(self) -> bool:
        return self.T_wb is not None

    # --------------------------------------------------------------- #
    #  SLAM ingestion (collapses ParticlePose → PoseEstimate)
    # --------------------------------------------------------------- #

    def ingest_slam(self, slam_result) -> None:
        """Update (T_wb, Σ_wb) from the latest SLAM result.

        Cached `prev_T_wb` is used by `predict_static` to form ΔT_wb.
        If the backend returns a `ParticlePose`, we moment-match to a
        Gaussian here (the Gaussian pipeline cannot represent
        multimodality).
        """
        pe = as_gaussian(slam_result)
        self.prev_T_wb = None if self.T_wb is None else self.T_wb.copy()
        self.T_wb = pe.T.copy()
        self.Sigma_wb = pe.cov.copy()

    # --------------------------------------------------------------- #
    #  Object initialization (base-frame)
    # --------------------------------------------------------------- #

    def ensure_object(self,
                      oid: int,
                      T_co_meas: np.ndarray,
                      init_cov: np.ndarray) -> bool:
        """Create an entry for `oid` from a camera-frame observation.

        `T_bo_init = T_bc · T_co_meas` — deterministic from known
        kinematics, so no Σ_wb enters at init. Returns True if this
        was a new object.
        """
        if oid in self.objects:
            return False
        T_bo = self.T_bc @ np.asarray(T_co_meas, dtype=np.float64)
        self.objects[oid] = GaussianObjectBelief(
            mu_bo=T_bo,
            cov_bo=np.asarray(init_cov, dtype=np.float64).copy(),
        )
        return True

    # --------------------------------------------------------------- #
    #  Predict steps
    # --------------------------------------------------------------- #

    def predict_static(self, Q_fn: Callable[[int], np.ndarray]) -> None:
        """Propagate base-frame belief under BASE motion.

        For a truly world-frame-static object, the base-frame pose is
        no longer constant when the robot moves; it transforms by
        `inv(ΔT_wb) = T_wb(t-1)·inv(T_wb(t))`. Covariance propagates
        via the adjoint plus a phase-aware Q supplied by the caller.

        The uncertainty in ΔT_wb is intentionally NOT added to Σ_bo:
        that's handled at output time by the world-frame composition.
        Folding it in here would re-introduce the same multi-frame
        correlation bug that motivated base-frame storage.

        First frame (no prev_T_wb): no base-motion transform, only Q.
        """
        if not self.objects:
            return

        if self.prev_T_wb is None:
            # No base-motion baseline yet. Just inflate by Q.
            for oid, b in self.objects.items():
                b.mu_bo, b.cov_bo = ekf_predict(b.mu_bo, b.cov_bo, Q_fn(oid))
            return

        # ΔT_wb = T_wb(t) · inv(T_wb(t-1)) (from means).
        delta_T_wb = self.T_wb @ np.linalg.inv(self.prev_T_wb)
        inv_delta = np.linalg.inv(delta_T_wb)
        Ad_inv_delta = se3_adjoint(inv_delta)

        for oid, b in self.objects.items():
            Q = Q_fn(oid)
            b.mu_bo = inv_delta @ b.mu_bo
            b.cov_bo = Ad_inv_delta @ b.cov_bo @ Ad_inv_delta.T + Q
            b.cov_bo = 0.5 * (b.cov_bo + b.cov_bo.T)

    def rigid_attachment_predict(self,
                                 oid: int,
                                 delta_T_bg: np.ndarray,
                                 Q_manip: np.ndarray) -> None:
        """Apply a rigid-attachment predict in base frame.

        `delta_T_bg = T_bg(t) · inv(T_bg(t-1))` is the gripper change
        in the base frame (from proprioception).

        Base-frame kinematics of a gripper-attached object:
            T_bo(t) = T_bg(t) · T_go = ΔT_bg · T_bo(t-1)

        Covariance:
            μ ← ΔT_bg · μ
            Σ ← Ad(ΔT_bg) · Σ · Ad(·)ᵀ + Q_manip

        Cleaner than the RBPF rigid predict (which couples to T_wb on
        both sides) because no base-to-world cancellation is needed.
        """
        b = self.objects.get(oid)
        if b is None:
            return
        Ad = se3_adjoint(delta_T_bg)
        b.mu_bo = delta_T_bg @ b.mu_bo
        b.cov_bo = Ad @ b.cov_bo @ Ad.T + Q_manip
        b.cov_bo = 0.5 * (b.cov_bo + b.cov_bo.T)

    # --------------------------------------------------------------- #
    #  Measurement update (base-frame IEKF)
    # --------------------------------------------------------------- #

    def update_observation(self,
                           oid: int,
                           T_co_meas: np.ndarray,
                           R_icp: np.ndarray,
                           iekf_iters: int = 2) -> None:
        """Base-frame IEKF update. Σ_wb does NOT enter here.

        T_bo_meas = T_bc · T_co_meas
        R_bo      = Ad(T_bc) · R_icp · Ad(T_bc)ᵀ
        δ         = se3_log(μ_bo⁻¹ · T_bo_meas)
        S         = Σ_bo + R_bo
        K         = Σ_bo · S⁻¹
        IEKF: relinearize δ at the running estimate (standard IEKF —
        covariance updated once at the end).
        """
        b = self.objects.get(oid)
        if b is None:
            return

        T_bo_meas = self.T_bc @ np.asarray(T_co_meas, dtype=np.float64)
        R_sym = 0.5 * (R_icp + R_icp.T)
        R_bo = self._Ad_bc @ R_sym @ self._Ad_bc.T
        R_bo = 0.5 * (R_bo + R_bo.T)

        S = b.cov_bo + R_bo
        K = np.linalg.solve(S.T, b.cov_bo.T).T
        I6 = np.eye(6)

        mu_lin = b.mu_bo.copy()
        for _ in range(max(1, iekf_iters)):
            delta = se3_log(np.linalg.inv(mu_lin) @ T_bo_meas)
            mu_lin = b.mu_bo @ se3_exp(K @ delta)

        b.mu_bo = mu_lin
        b.cov_bo = (I6 - K) @ b.cov_bo
        b.cov_bo = 0.5 * (b.cov_bo + b.cov_bo.T)

    # --------------------------------------------------------------- #
    #  Output composition (the only place Σ_wb enters)
    # --------------------------------------------------------------- #

    def collapsed_base(self) -> PoseEstimate:
        """The current SLAM posterior, echoed for API symmetry with RBPF."""
        assert self.T_wb is not None, "ingest_slam has not been called yet"
        return PoseEstimate(T=self.T_wb.copy(), cov=self.Sigma_wb.copy())

    def collapsed_object_base(self, oid: int) -> Optional[PoseEstimate]:
        b = self.objects.get(oid)
        if b is None:
            return None
        return PoseEstimate(T=b.mu_bo.copy(), cov=b.cov_bo.copy())

    def collapsed_object_world(self, oid: int) -> Optional[PoseEstimate]:
        """Compose T_wo = T_wb · T_bo and propagate covariance.

        Under independence δ_wb ⊥ δ_bo (a reasonable approximation
        given base-frame storage keeps them decoupled through the
        recursion):
            δ_wo = Ad(T_bo⁻¹) · δ_wb + δ_bo
            Σ_wo = Ad(T_bo⁻¹) · Σ_wb · Ad(·)ᵀ + Σ_bo

        Σ_wo is lower-bounded by the projected Σ_wb term, which is
        exactly what we want for a static object observed many times
        from the same base.
        """
        if self.T_wb is None:
            return None
        b = self.objects.get(oid)
        if b is None:
            return None
        mu_wo = self.T_wb @ b.mu_bo
        Ad_bo_inv = se3_adjoint(np.linalg.inv(b.mu_bo))
        cov_wo = Ad_bo_inv @ self.Sigma_wb @ Ad_bo_inv.T + b.cov_bo
        cov_wo = 0.5 * (cov_wo + cov_wo.T)
        return PoseEstimate(T=mu_wo, cov=cov_wo)

    def collapsed_objects_world(self) -> Dict[int, PoseEstimate]:
        """World-frame collapsed view for every tracked object."""
        out: Dict[int, PoseEstimate] = {}
        for oid in self.objects:
            pe = self.collapsed_object_world(oid)
            if pe is not None:
                out[oid] = pe
        return out

    # --------------------------------------------------------------- #
    #  Slow-tier reconcile
    # --------------------------------------------------------------- #

    def inject_posterior_world(self,
                                oid: int,
                                posterior: PoseEstimate) -> None:
        """Re-center the base-frame mean from a world-frame slow-tier
        posterior. We deliberately do NOT overwrite Σ_bo — the
        world-frame posterior covariance has Σ_wb folded in, and
        injecting it back into base-frame covariance would double-count.
        The next observation will tighten Σ_bo naturally.
        """
        b = self.objects.get(oid)
        if b is None or self.T_wb is None:
            return
        b.mu_bo = np.linalg.inv(self.T_wb) @ posterior.T
