"""
Two-tier orchestrator — Rao-Blackwellized variant.

Wires Layer 1 (SLAM) to Layer 2 (movable-object tracking) via a proper
Rao-Blackwellized factorization instead of collapsing the SLAM posterior
to one Gaussian and forwarding its covariance into R (the "cascaded KF"
pattern we deliberately avoid).

Factorization:
    p(x_{1:t}, {o}_i | z_{1:t}) = p(x_{1:t} | z_{1:t}) · Π_i p(o_i | x_{1:t}, z_{1:t})

* `p(x_{1:t} | z_{1:t})` — particles.
* `p(o_i | x_{1:t}, z_{1:t})` — per-particle EKF on SE(3), world frame.

Each detection contributes to (a) the per-particle object EKF and
(b) the per-particle log-weight, via the same innovation likelihood.
That is the "dual role" of vision the design discussion called out.

Slow tier (factor graph over raw observations) still exists. It consumes
the collapsed-mixture summary as its prior, runs the smoother, and the
posterior is injected back into every particle (Option A from the plan —
loses mixture structure but is the minimal integration).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any

import numpy as np

from pose_update.slam_interface import (
    PoseEstimate, ParticlePose,
    collect_movable_masks, mask_out_movable, SlamBackend,
)
from pose_update.ekf_se3 import process_noise_for_phase, huber_weight
from pose_update.factor_graph import (
    PoseGraphOptimizer, Observation, RelationEdge, OptimizationResult,
)
from pose_update.rbpf_state import RBPFState
from pose_update.association import (
    hungarian_associate, oracle_associate, sam2_alpha_from_q_s,
    AssociationResult,
)
from pose_update.bernoulli import (
    r_predict, r_assoc_update_loglik, r_miss_update, r_birth,
)
from pose_update.visibility import visibility_p_v


# ─────────────────────────────────────────────────────────────────────
# Trigger policy
# ─────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────
# Relation temporal filter
# ─────────────────────────────────────────────────────────────────────

class RelationFilter:
    """Exponential moving-average filter over scene-graph edges.

    Raw edge scores flicker frame-to-frame because the geometric
    relation test is noisy (random mock point clouds, bbox jitter).
    This filter smooths the binary present/absent signal into a
    stable 0-or-1 output.

    Per-edge EMA:
        ema(t) = α · raw(t) + (1 − α) · ema(t − 1)

    Output: emit the edge (score=1) when ema ≥ threshold, suppress
    otherwise. An edge not detected this frame gets raw=0.
    """

    def __init__(self, alpha: float = 0.3, threshold: float = 0.5):
        self.alpha = alpha
        self.threshold = threshold
        self._ema: Dict[tuple, float] = {}

    def update(self, raw_edges: List[RelationEdge]) -> List[RelationEdge]:
        """Accept raw edges from one frame; return the filtered set."""
        detected: Dict[tuple, float] = {}
        raw_meta: Dict[tuple, RelationEdge] = {}
        for edge in raw_edges:
            key = (edge.parent, edge.child, edge.relation_type)
            detected[key] = edge.score
            raw_meta[key] = edge

        all_keys = set(self._ema.keys()) | set(detected.keys())
        filtered: List[RelationEdge] = []
        for key in all_keys:
            raw = detected.get(key, 0.0)
            prev = self._ema.get(key, raw)
            ema = self.alpha * raw + (1.0 - self.alpha) * prev
            self._ema[key] = ema
            if ema >= self.threshold:
                parent, child, rel_type = key
                ref = raw_meta.get(key)
                filtered.append(RelationEdge(
                    parent=parent, child=child,
                    relation_type=rel_type,
                    score=1.0,
                    parent_size=ref.parent_size if ref else None,
                    child_size=ref.child_size if ref else None,
                ))
        # Prune dead edges (EMA decayed to near zero).
        self._ema = {k: v for k, v in self._ema.items() if v > 0.01}
        return filtered


# ─────────────────────────────────────────────────────────────────────
# Trigger policy
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TriggerConfig:
    """Configuration for when the slow tier fires.

    Fire on any manipulation event, on residual surprises, and as a periodic
    safety net every ~3 seconds at 30 Hz.
    """
    on_grasp: bool = True
    on_release: bool = True
    on_new_object: bool = True
    residual_threshold: float = 0.1        # in world-frame tangent norm
    periodic_every_n_frames: int = 90      # ~3 s at 30 Hz


# ─────────────────────────────────────────────────────────────────────
# Bernoulli-EKF mode config (docs/latex/bernoulli_ekf.tex)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BernoulliConfig:
    """Opts the orchestrator into the Bernoulli-EKF fast tier.

    Default values match the paper (§12 Calibrated parameters). Setting
    ``association_mode='oracle'`` + ``P_max=None`` + ``r_min=0.0`` +
    ``enable_visibility=False`` + ``G_in=G_out=inf`` reproduces the
    pre-Bernoulli legacy behaviour (the five substitutions of the
    degeneracy analysis).

    Attributes:
        association_mode: 'hungarian' for the full Mahalanobis cost with
            chi^2 gating, 'oracle' to short-circuit via ``det['id']``.
        p_s, p_d, q_s, lambda_c, lambda_b: Bayesian rates (§5, §7, §8).
        r_conf, r_min: emission and pruning thresholds on existence r.
        G_in, G_out: chi^2 Huber gates (eq:huber). +inf disables Huber
            re-weighting (w=1 always) and the outer gate effectively
            admits every pair (matched by the inner-gate branch).
        P_max: covariance-saturation cap (eq:phi). None = no cap.
        enable_visibility: if False, p_v = 1 for all tracks (miss branch
            unchanged from pre-Bernoulli); otherwise compute eq:pv.
        K, image_shape: camera intrinsics / image size for p_v.
        init_cov_from_R: if True, birth covariance = R_icp (paper §8);
            if False, use the orchestrator's _INIT_OBJ_COV constant.
    """
    association_mode: str = "hungarian"
    p_s: float = 1.0
    p_d: float = 0.9
    q_s: float = 0.9
    lambda_c: float = 1.0
    lambda_b: float = 1.0
    r_conf: float = 0.5
    r_min: float = 1e-3
    G_in: float = 12.59
    G_out: float = 25.0
    P_max: Optional[np.ndarray] = None
    enable_visibility: bool = True
    enable_huber: bool = True
    init_cov_from_R: bool = True
    enforce_label_match: bool = True
    K: Optional[np.ndarray] = None
    image_shape: Optional[tuple] = None

    @property
    def alpha(self) -> float:
        """SAM2 continuity bonus in Mahalanobis^2 units."""
        return sam2_alpha_from_q_s(self.q_s)

    @classmethod
    def degeneracy(cls, **overrides) -> "BernoulliConfig":
        """Build a config that reproduces the pre-Bernoulli legacy behaviour
        exactly (used by the degeneracy test)."""
        base = dict(
            association_mode="oracle",
            p_s=1.0,
            p_d=0.9,
            q_s=0.5,           # alpha = 0, no SAM2 bonus
            lambda_c=1.0,
            lambda_b=1.0,
            r_conf=0.0,        # emit everything
            r_min=0.0,         # never prune
            G_in=float("inf"),
            G_out=float("inf"),
            P_max=None,
            enable_visibility=False,
            enable_huber=False,
            init_cov_from_R=False,
            enforce_label_match=False,
        )
        base.update(overrides)
        return cls(**base)


# ─────────────────────────────────────────────────────────────────────
# Two-tier orchestrator (RBPF)
# ─────────────────────────────────────────────────────────────────────

class TwoTierOrchestrator:
    """Coordinator for the hierarchical movable-object tracking pipeline.

    The SLAM backend may return a `PoseEstimate` (Gaussian) or a `ParticlePose`.
    Internally we always keep N particles; if the backend gave us a Gaussian,
    we sample fresh each frame (approximate RBPF — particle identity across
    frames is carried by the *slot* index, not by trajectory continuity).

    Attributes:
        state:             RBPFState — particles + per-particle per-object EKFs
        n_particles:       int
        object_labels:     Dict[int, str]
        object_first_seen: Dict[int, int] (frame idx of first detection)
        frames_since_obs:  Dict[int, int]
        T_oe:              Dict[int, Optional[np.ndarray]]
                           (locked at grasp onset, object-in-EE frame)
        frame_count, last_opt_frame, last_state — orchestrator bookkeeping.
    """

    # Default loose init covariance for a newly-detected object (same
    # magnitude the legacy orchestrator used).
    _INIT_OBJ_COV = np.diag([0.05] * 3 + [0.05] * 3)
    # Additional per-step noise for manipulated objects (grip slack /
    # deformation). Additive on top of Ad(ΔT)·Σ·Adᵀ.
    _Q_MANIP_PER_STEP = np.diag([1e-6] * 3 + [1e-6] * 3)
    # Uniform Q used by the TSDF++-style vision-only baseline. Chosen so
    # the Kalman gain lands near 0.5 under R_icp ~ 1e-4, which allows the
    # filter to chase vision for moving objects without runaway noise for
    # static ones. The baseline has no manipulation phase to branch on.
    _Q_BASELINE = np.diag([1e-4] * 3 + [1e-4] * 3)

    def __init__(self,
                 slam_backend: SlamBackend,
                 trigger: Optional[TriggerConfig] = None,
                 optimizer: Optional[PoseGraphOptimizer] = None,
                 n_particles: int = 32,
                 ess_resample_frac: float = 0.5,
                 iekf_iters: int = 2,
                 rng_seed: Optional[int] = None,
                 verbose: bool = False,
                 baseline_mode: bool = False,
                 bernoulli: Optional[BernoulliConfig] = None):
        """Args:
            baseline_mode: if True, ignore all proprioception (T_ec, T_bg,
                gripper_state) and use a uniform vision-friendly Q. This
                is the TSDF++-style baseline: vision-only, no manipulation
                awareness, no rigid-attachment predict.
            bernoulli: if not None, use the Bernoulli-EKF fast tier
                (bernoulli_ekf.tex). The default None preserves the legacy
                upstream-ID lookup behaviour for backward compatibility.
        """
        self.slam = slam_backend
        self.trigger = trigger or TriggerConfig()
        self.optimizer = optimizer or PoseGraphOptimizer()
        self.verbose = verbose
        self.baseline_mode = baseline_mode
        self.bernoulli = bernoulli  # None = legacy path

        self.n_particles = n_particles
        self.ess_resample_frac = ess_resample_frac
        self.iekf_iters = iekf_iters

        rng = np.random.default_rng(rng_seed) if rng_seed is not None \
            else np.random.default_rng()
        self.state = RBPFState(n_particles=n_particles, rng=rng)

        # Object-level metadata (particle-independent)
        self.object_labels: Dict[int, str] = {}
        self.object_first_seen: Dict[int, int] = {}
        self.frames_since_obs: Dict[int, int] = {}
        self.T_oe: Dict[int, Optional[np.ndarray]] = {}

        # Bernoulli-EKF per-track state (only populated when bernoulli != None):
        #   existence r^(i)_{k|k} ; last-matched SAM2 tracklet id tau^(i).
        self.existence: Dict[int, float] = {}
        self.sam2_tau: Dict[int, int] = {}

        self.frame_count = 0
        self.last_opt_frame = -1
        self.last_state: Dict[str, Any] = {
            "phase": "idle", "held_obj_id": None}

        self._cached_relations: List[RelationEdge] = []
        self._relation_filter = RelationFilter(alpha=0.3, threshold=0.5)
        # Previous gripper pose in base frame; used to compute ΔT for the
        # rigid-attachment predict of the manipulation set.
        self._prev_T_bg: Optional[np.ndarray] = None

    # --------------------------------------------------------------- #
    #  Backward-compatibility view: dict-of-dicts like the old API
    # --------------------------------------------------------------- #

    @property
    def objects(self) -> Dict[int, Dict[str, Any]]:
        """Collapsed-mixture view for legacy consumers.

        Each entry has the same shape the old orchestrator exposed:
            {"T": (4,4), "cov": (6,6), "label": str,
             "frames_since_observation": int, "T_oe": Optional[(4,4)]}
        In Bernoulli mode, tracks whose r_{k|k} < r_conf are filtered out
        of this view (they remain predicted / associated / updated but are
        tentative; callers should use .tentative_objects to see them).
        """
        out: Dict[int, Dict[str, Any]] = {}
        r_conf = (self.bernoulli.r_conf
                  if self.bernoulli is not None else 0.0)
        for oid, pe in self.state.collapsed_objects().items():
            if self.bernoulli is not None:
                r = self.existence.get(oid, 0.0)
                if r < r_conf:
                    continue
            entry = {
                "T": pe.T,
                "cov": pe.cov,
                "label": self.object_labels.get(oid, "unknown"),
                "frames_since_observation":
                    self.frames_since_obs.get(oid, 0),
                "T_oe": self.T_oe.get(oid),
            }
            if self.bernoulli is not None:
                entry["r"] = float(self.existence.get(oid, 0.0))
                entry["sam2_id"] = int(self.sam2_tau.get(oid, -1))
            out[oid] = entry
        return out

    @property
    def tentative_objects(self) -> Dict[int, Dict[str, Any]]:
        """Bernoulli mode: tracks with r < r_conf (below emission).

        Empty in legacy mode.
        """
        if self.bernoulli is None:
            return {}
        out: Dict[int, Dict[str, Any]] = {}
        r_conf = self.bernoulli.r_conf
        for oid, pe in self.state.collapsed_objects().items():
            r = self.existence.get(oid, 0.0)
            if r >= r_conf:
                continue
            out[oid] = {
                "T": pe.T,
                "cov": pe.cov,
                "label": self.object_labels.get(oid, "unknown"),
                "r": float(r),
                "sam2_id": int(self.sam2_tau.get(oid, -1)),
            }
        return out

    # --------------------------------------------------------------- #
    #  Public entry point
    # --------------------------------------------------------------- #

    def step(self,
             rgb: np.ndarray,
             depth: np.ndarray,
             detections: List[Dict[str, Any]],
             gripper_state: Dict[str, Any],
             T_ec: Optional[np.ndarray] = None,
             T_bg: Optional[np.ndarray] = None,
             odom_prior: Optional[PoseEstimate] = None,
             ) -> Dict[str, Any]:
        """Process one frame end-to-end.

        Args:
            rgb, depth: current frame images.
            detections: list of dicts with at minimum 'id' (int), 'mask' (H,W),
                'label' (str), 'score' (float), 'T_co' (4,4 ICP), 'R_icp'
                (6,6), 'fitness', 'rmse'. Detections without an 'id' are
                ignored (no data association done here).
            gripper_state: {'phase': 'idle'|'grasping'|'holding'|'releasing',
                            'held_obj_id': int or None}.
            T_ec: end-effector-to-camera transform (needed at grasp onset to
                  lock T_oe).
            T_bg: gripper pose in base frame from proprioception. Used by the
                  rigid-attachment predict for the manipulation set.
            odom_prior: optional odometry prior for the SLAM backend.

        Returns:
            Report dict. Backward-compatible keys: 'slam_pose', 'triggered',
            'alpha', 'residuals', 'objects'. Added: 'slam_raw',
            'base_particles', 'ess', 'resampled'.
        """
        report: Dict[str, Any] = {}
        # Snapshot known objects BEFORE the fast tier can add newly-seen
        # ones; the new-object trigger compares against this set.
        self._known_before_this_step = set(self.object_labels.keys())

        # ── 1. Layer 1: SLAM on depth-with-movable-masked-out ──────────
        movable_mask = collect_movable_masks(detections, depth.shape)
        masked_depth = mask_out_movable(depth, movable_mask)
        slam_raw = self.slam.step(rgb, masked_depth, odom_prior)

        # Ingest into particles (initialize on first call, otherwise refresh
        # T_wb per slot). Does NOT collapse to a single Gaussian.
        self.state.ingest_slam(slam_raw)

        # Collapsed summaries for legacy callers.
        slam_pose = self.state.collapsed_base()
        report["slam_raw"] = slam_raw
        report["slam_pose"] = slam_pose
        report["base_particles"] = ParticlePose(
            particles=np.stack(
                [p.T_wb for p in self.state.particles], axis=0),
            weights=self.state.normalized_weights(),
        )

        # ── 2. Fast tier: per-particle per-object EKF + weight update ──
        if self.bernoulli is not None:
            self._fast_tier_bernoulli(
                detections, gripper_state, T_ec, T_bg, depth.shape[:2])
        else:
            self._fast_tier(detections, gripper_state, T_ec, T_bg)

        # ── 3. Scene graph relations (on the collapsed view) ───────────
        self._cached_relations = self._recompute_relations()

        # ── 4. Slow-tier trigger ───────────────────────────────────────
        should_trigger = self._should_trigger(gripper_state, detections)
        report["triggered"] = should_trigger

        if should_trigger:
            opt_result = self._slow_tier(
                slam_pose, detections, gripper_state, T_ec)
            report["alpha"] = opt_result.alpha
            report["residuals"] = opt_result.residuals
            self.last_opt_frame = self.frame_count
        else:
            report["alpha"] = None
            report["residuals"] = {}

        # ── 5. ESS-triggered resampling ────────────────────────────────
        resampled = self.state.resample_if_needed(
            threshold_frac=self.ess_resample_frac)
        report["resampled"] = resampled
        report["ess"] = self.state.ess()

        # Update bookkeeping
        self.last_state = dict(gripper_state)
        self._prev_T_bg = None if T_bg is None else T_bg.copy()
        self.frame_count += 1

        # Final collapsed view for the report
        report["objects"] = {
            oid: {"T": entry["T"].copy(),
                  "cov": entry["cov"].copy(),
                  "label": entry["label"]}
            for oid, entry in self.objects.items()
        }
        return report

    # --------------------------------------------------------------- #
    #  Fast tier: per-particle per-object EKF + weight accumulation
    # --------------------------------------------------------------- #

    def _fast_tier(self,
                   detections: List[Dict[str, Any]],
                   gripper_state: Dict[str, Any],
                   T_ec: Optional[np.ndarray],
                   T_bg: Optional[np.ndarray]) -> None:
        # Baseline mode: strip all proprioception and use uniform Q. This
        # is the vision-only TSDF++-style comparison.
        if self.baseline_mode:
            gripper_state = {"phase": "idle", "held_obj_id": None}
            T_ec = None
            T_bg = None
        phase = gripper_state.get("phase", "idle")
        held_id = gripper_state.get("held_obj_id")

        # Objects considered "moving with the robot" — the held one plus any
        # scene-graph passengers ("in"/"on" relations).
        manipulation_set = self._get_manipulation_set(held_id)

        # Manipulation-set members get a rigid-attachment predict when we
        # have two proprioception samples to form ΔT_gripper_b. Otherwise
        # we fall back to the legacy phase-aware inflated-Q predict for
        # them. Either way, each object is predicted exactly once.
        apply_rigid = (T_bg is not None
                       and self._prev_T_bg is not None
                       and bool(manipulation_set))
        delta_T_grip_b = (T_bg @ np.linalg.inv(self._prev_T_bg)
                          if apply_rigid else None)

        def Q_fn(oid: int, _particle) -> np.ndarray:
            if self.baseline_mode:
                return self._Q_BASELINE
            rigid_here = apply_rigid and (oid in manipulation_set)
            # Under rigid predict, Q_manip is added by rigid_attachment_predict
            # below — we skip the generic inflation here to avoid double-Q.
            if rigid_here:
                return np.zeros((6, 6))
            return process_noise_for_phase(
                phase=phase,
                is_target=(oid in manipulation_set),
                frames_since_observation=self.frames_since_obs.get(oid, 0),
                frame="world",
            )

        self.state.predict_objects(Q_fn)

        # Per-frame tick of frames_since_obs.
        for oid in self.frames_since_obs:
            self.frames_since_obs[oid] += 1

        if apply_rigid:
            for oid in manipulation_set:
                if any(oid in p.objects for p in self.state.particles):
                    self.state.rigid_attachment_predict(
                        oid, delta_T_grip_b, self._Q_MANIP_PER_STEP)

        # ── Measurement update for each observed object ────────────────
        for det in detections:
            oid = det.get("id")
            if oid is None:
                continue

            # Initialize on first sight (per-particle world-frame).
            if oid not in self.object_labels:
                T_co_meas = np.asarray(det["T_co"], dtype=np.float64)
                self.state.ensure_object(
                    oid, T_co_meas, self._INIT_OBJ_COV)
                self.object_labels[oid] = det.get("label", "unknown")
                self.object_first_seen[oid] = self.frame_count
                self.frames_since_obs[oid] = 0
                self.T_oe[oid] = None
                continue

            # EKF update (+ log-weight accumulation).
            R_icp = det.get("R_icp", np.eye(6) * 1e-3)
            T_co_meas = np.asarray(det["T_co"], dtype=np.float64)
            self.state.update_observation(
                oid=oid,
                T_co_meas=T_co_meas,
                R_icp=R_icp,
                iekf_iters=self.iekf_iters,
            )
            self.frames_since_obs[oid] = 0

            # Lock T_oe at grasp onset (first frame entering 'grasping').
            if (phase == "grasping"
                    and self.last_state.get("phase") != "grasping"
                    and oid == held_id
                    and T_ec is not None):
                collapsed = self.state.collapsed_object(oid)
                if collapsed is not None:
                    slam_pose = self.state.collapsed_base()
                    T_ew = slam_pose.T @ T_ec
                    self.T_oe[oid] = np.linalg.inv(T_ew) @ collapsed.T

    # --------------------------------------------------------------- #
    #  Fast tier (Bernoulli-EKF mode; bernoulli_ekf.tex)
    # --------------------------------------------------------------- #

    def _fast_tier_bernoulli(self,
                             detections: List[Dict[str, Any]],
                             gripper_state: Dict[str, Any],
                             T_ec: Optional[np.ndarray],
                             T_bg: Optional[np.ndarray],
                             image_shape: tuple) -> None:
        """Bernoulli-EKF fast tier following docs/latex/bernoulli_ekf.tex.

        Implements the 7-step frame-by-frame algorithm (§11):
          1. Predict with Phi saturation + eq:bern_pred_r.
          2. Measure: ICP outputs already baked into det['T_co'], det['R_icp'].
          3. Associate: Hungarian on Mahalanobis cost (or oracle).
          4. Update per track: matched -> eq:r_assoc + EKF; missed -> eq:r_miss
             with p_v from the visibility predicate.
          5. Birth: unmatched measurements -> eq:birth_r + eq:birth_state.
          6. Prune: drop r < r_min.
          7. Emit: report via `self.objects` (filters by r_conf).
        """
        cfg = self.bernoulli
        if self.baseline_mode:
            gripper_state = {"phase": "idle", "held_obj_id": None}
            T_ec = None
            T_bg = None
        phase = gripper_state.get("phase", "idle")
        held_id = gripper_state.get("held_obj_id")

        manipulation_set = self._get_manipulation_set(held_id)

        apply_rigid = (T_bg is not None
                       and self._prev_T_bg is not None
                       and bool(manipulation_set))
        delta_T_grip_b = (T_bg @ np.linalg.inv(self._prev_T_bg)
                          if apply_rigid else None)

        def Q_fn(oid: int, _particle) -> np.ndarray:
            if self.baseline_mode:
                return self._Q_BASELINE
            rigid_here = apply_rigid and (oid in manipulation_set)
            if rigid_here:
                return np.zeros((6, 6))
            return process_noise_for_phase(
                phase=phase,
                is_target=(oid in manipulation_set),
                frames_since_observation=self.frames_since_obs.get(oid, 0),
                frame="world",
            )

        # ── 1. Predict state and existence (eq:ekf_*_pred + eq:bern_pred_r) ─
        self.state.predict_objects(Q_fn, P_max=cfg.P_max)

        for oid in self.frames_since_obs:
            self.frames_since_obs[oid] += 1

        if apply_rigid:
            for oid in manipulation_set:
                if any(oid in p.objects for p in self.state.particles):
                    self.state.rigid_attachment_predict(
                        oid, delta_T_grip_b, self._Q_MANIP_PER_STEP)

        for oid in list(self.existence.keys()):
            self.existence[oid] = r_predict(self.existence[oid], cfg.p_s)

        # ── 2. Associate: Hungarian (with SAM2-ID bonus) or oracle ─────
        track_oids = list(self.object_labels.keys())
        if cfg.association_mode == "oracle":
            assoc = oracle_associate(track_oids, detections)
        else:
            assoc = hungarian_associate(
                track_oids=track_oids,
                detections=detections,
                innovation_fn=self.state.innovation_stats,
                track_labels=self.object_labels,
                track_tau=self.sam2_tau,
                alpha=cfg.alpha,
                G_out=cfg.G_out,
                enforce_label_match=cfg.enforce_label_match,
            )

        # ── Visibility for missed tracks (eq:pv) ────────────────────────
        if cfg.enable_visibility:
            p_v_map = self._compute_visibility(detections, image_shape)
        else:
            p_v_map = {oid: 1.0 for oid in track_oids}

        # Track which detections have been consumed (matched or rejected
        # into birth). The set grows as we process the match loop so that
        # outer-gate rejects can also spawn new tracks.
        consumed_dets: Set[int] = set()

        # ── 3. Matched tracks: Huber + EKF + eq:r_assoc ─────────────────
        for oid, l in list(assoc.match.items()):
            det = detections[l]
            T_co = np.asarray(det.get("T_co"), dtype=np.float64)
            R_icp = np.asarray(det.get("R_icp"), dtype=np.float64)

            stats = self.state.innovation_stats(oid, T_co, R_icp)
            if stats is None:
                # Track was pruned between association and update; treat
                # this detection as a birth candidate.
                continue
            _nu, _S, d2, log_lik = stats

            if cfg.enable_huber:
                w = huber_weight(d2, cfg.G_in, cfg.G_out)
            else:
                w = 1.0

            if w <= 0.0:
                # Outer-gate reject: route to miss branch for this track
                # and let the detection go to birth.
                assoc.unmatched_tracks.append(oid)
                del assoc.match[oid]
                continue

            self.state.update_observation(
                oid=oid,
                T_co_meas=T_co,
                R_icp=R_icp,
                iekf_iters=self.iekf_iters,
                huber_w=w,
                P_max=cfg.P_max,
            )
            self.frames_since_obs[oid] = 0
            consumed_dets.add(l)

            # Existence update (eq:r_assoc) in log-space so a very low
            # likelihood does not underflow.
            r_prev = self.existence.get(oid, 1.0)
            r_new = r_assoc_update_loglik(
                r_prev, log_L=log_lik,
                p_d=cfg.p_d, lambda_c=cfg.lambda_c)
            self.existence[oid] = r_new

            # SAM2-ID bookkeeping: update only on a successful match.
            # Fall back to `id` if the detector client hasn't populated
            # `sam2_id` explicitly (paper §6.1: the upstream tracklet
            # identifier is the same quantity either way).
            d_tau = det.get("sam2_id", det.get("id"))
            if d_tau is not None:
                try:
                    self.sam2_tau[oid] = int(d_tau)
                except (TypeError, ValueError):
                    pass

            # Lock T_oe at grasp onset (mirrors legacy path).
            if (phase == "grasping"
                    and self.last_state.get("phase") != "grasping"
                    and oid == held_id
                    and T_ec is not None):
                collapsed = self.state.collapsed_object(oid)
                if collapsed is not None:
                    slam_pose = self.state.collapsed_base()
                    T_ew = slam_pose.T @ T_ec
                    self.T_oe[oid] = np.linalg.inv(T_ew) @ collapsed.T

        # ── 4. Missed tracks: eq:r_miss ─────────────────────────────────
        for oid in assoc.unmatched_tracks:
            if oid not in self.existence:
                continue
            p_v = p_v_map.get(oid, 1.0)
            p_d_tilde = cfg.p_d * p_v
            r_prev = self.existence[oid]
            self.existence[oid] = r_miss_update(r_prev, p_d_tilde)

        # ── 5. Birth (eq:birth_r + eq:birth_state) ──────────────────────
        # A detection births iff it was not consumed by a surviving match.
        # Huber-rejected matches remove their detection from consumed_dets
        # so it flows to birth here.
        for l in range(len(detections)):
            if l in consumed_dets:
                continue
            det = detections[l]
            self._birth_track(det, cfg)

        # ── 6. Prune (r < r_min) ────────────────────────────────────────
        if cfg.r_min > 0.0:
            to_prune = [oid for oid, r in self.existence.items()
                        if r < cfg.r_min]
            for oid in to_prune:
                self._prune_track(oid)

    def _compute_visibility(self,
                             detections: List[Dict[str, Any]],
                             image_shape: tuple) -> Dict[int, float]:
        """Collect per-track bboxes + mean depth from the CURRENT frame's
        matched detections (matched pairs only), then run eq:pv.

        Tracks without a matching detection this frame fall back to a
        bbox-less record -- so their p_v is driven solely by the frustum
        projection gate. This is a reasonable first cut; a more faithful
        implementation would project each track's TSDF to image space.
        """
        cfg = self.bernoulli
        if cfg.K is None:
            return {oid: 1.0 for oid in self.object_labels}

        # Base pose -> camera pose.
        slam_pose = self.state.collapsed_base()
        T_wb = slam_pose.T

        tracks_for_vis: List[Dict[str, Any]] = []
        for oid in self.object_labels:
            pe = self.state.collapsed_object(oid)
            if pe is None:
                continue
            bbox_im = None
            mean_depth = None
            # Look up the most recent detection that matched this oid's
            # label as a proxy for the image-space bbox.
            for det in detections:
                if det.get("label") == self.object_labels.get(oid):
                    if det.get("box") is not None:
                        bx = det["box"]
                        try:
                            bbox_im = tuple(float(v) for v in bx)
                        except (TypeError, ValueError):
                            bbox_im = None
                    if det.get("T_co") is not None:
                        T_co = np.asarray(det["T_co"], dtype=np.float64)
                        mean_depth = float(T_co[2, 3])
                    break
            tracks_for_vis.append({
                "oid": int(oid),
                "T": pe.T,
                "bbox_image": bbox_im,
                "mean_depth_camera": mean_depth,
            })

        # Camera-to-world: T_cw = T_wb (camera == base-frame in the test
        # harness). For the real pipeline, callers should set
        # cfg.K = intrinsics and treat T_wb as camera_to_world; if an
        # explicit T_cw is needed it can be supplied via cfg attributes.
        return visibility_p_v(tracks_for_vis, cfg.K, T_wb,
                               cfg.image_shape or image_shape)

    def _birth_track(self, det: Dict[str, Any],
                     cfg: BernoulliConfig) -> None:
        """Initialise a new Bernoulli track from an unmatched detection
        (eq:birth_r + eq:birth_state).

        In Hungarian mode, the internal track id is minted fresh by the
        tracker -- upstream ``det['id']`` is NOT reused, because an
        unmatched detection by definition does not correspond to any
        currently-known track. Reusing a colliding ``det['id']`` would
        silently overwrite an existing track's (pose, r, label) state.

        In oracle mode, identity is taken from upstream directly.
        """
        T_co = det.get("T_co")
        if T_co is None:
            return
        T_co = np.asarray(T_co, dtype=np.float64)
        R_icp = det.get("R_icp")
        if R_icp is None:
            R_icp = np.eye(6) * 1e-3
        R_icp = np.asarray(R_icp, dtype=np.float64)

        # Mint the internal track id. Oracle mode respects upstream; Hungarian
        # mode always assigns a fresh id to avoid colliding with existing
        # tracks that happen to share det['id'].
        if cfg.association_mode == "oracle":
            raw_id = det.get("id")
            if raw_id is None:
                d_id = self._next_track_id()
            else:
                d_id = int(raw_id)
                if d_id in self.object_labels:
                    # Even in oracle mode, a colliding upstream id on an
                    # otherwise unmatched detection is a bug; mint a fresh
                    # one to keep the track state consistent.
                    d_id = self._next_track_id()
        else:
            d_id = self._next_track_id()

        if cfg.init_cov_from_R:
            init_cov = 0.5 * (R_icp + R_icp.T)
        else:
            init_cov = self._INIT_OBJ_COV.copy()

        self.state.ensure_object(d_id, T_co, init_cov)
        self.object_labels[d_id] = det.get("label", "unknown")
        self.object_first_seen[d_id] = self.frame_count
        self.frames_since_obs[d_id] = 0
        self.T_oe.setdefault(d_id, None)

        score = float(det.get("score", 1.0))
        self.existence[d_id] = r_birth(
            score, lambda_b=cfg.lambda_b, lambda_c=cfg.lambda_c)

        d_tau = det.get("sam2_id", det.get("id"))
        if d_tau is not None:
            try:
                self.sam2_tau[d_id] = int(d_tau)
            except (TypeError, ValueError):
                self.sam2_tau[d_id] = -1
        else:
            self.sam2_tau[d_id] = -1

    def _prune_track(self, oid: int) -> None:
        """Remove `oid` from every particle and every orchestrator dict."""
        self.state.delete_object(oid)
        self.object_labels.pop(oid, None)
        self.object_first_seen.pop(oid, None)
        self.frames_since_obs.pop(oid, None)
        self.T_oe.pop(oid, None)
        self.existence.pop(oid, None)
        self.sam2_tau.pop(oid, None)

    def _next_track_id(self) -> int:
        """Next unused track id (for Hungarian-mode births without id)."""
        if not self.object_labels:
            return 1
        return max(self.object_labels.keys()) + 1

    # --------------------------------------------------------------- #
    #  Slow tier: joint pose graph (Option A — operates on collapsed)
    # --------------------------------------------------------------- #

    def _slow_tier(self,
                   slam_pose: PoseEstimate,
                   detections: List[Dict[str, Any]],
                   gripper_state: Dict[str, Any],
                   T_ec: Optional[np.ndarray]) -> OptimizationResult:
        collapsed = self.state.collapsed_objects()
        priors: Dict[int, PoseEstimate] = dict(collapsed)

        observations: List[Observation] = []
        for det in detections:
            oid = det.get("id")
            if oid is None or oid not in priors:
                continue
            observations.append(Observation(
                obj_id=oid,
                T_co=det["T_co"],
                R_icp=det.get("R_icp", np.eye(6) * 1e-3),
                fitness=det.get("fitness", 0.9),
                rmse=det.get("rmse", 0.005),
            ))

        held_id = gripper_state.get("held_obj_id")
        T_ew = None
        T_oe = None
        if (held_id is not None
                and held_id in priors
                and T_ec is not None
                and self.T_oe.get(held_id) is not None):
            T_ew = slam_pose.T @ T_ec
            T_oe = self.T_oe[held_id]

        result = self.optimizer.run(
            slam_pose=slam_pose,
            priors=priors,
            observations=observations,
            relations=self._cached_relations,
            held_obj_id=held_id,
            T_ew=T_ew,
            T_oe=T_oe,
        )

        # Inject optimized posteriors back into every particle (Option A).
        for oid, pe in result.posteriors.items():
            self.state.inject_posterior(oid, pe)

        if self.verbose:
            print(f"[slow tier] α={result.alpha:.2f}, "
                  f"iters={result.num_iterations}, "
                  f"residuals={ {k: v.tolist() for k, v in result.residuals.items()} }")

        return result

    # --------------------------------------------------------------- #
    #  Manipulation-set propagation via scene-graph relations
    # --------------------------------------------------------------- #

    def _get_manipulation_set(self,
                              held_id: Optional[int]) -> Set[int]:
        """Transitive closure of the held object under "in"/"on" relations.

        If the robot holds a bowl and the scene graph says "apple in bowl",
        then the apple rides with the bowl under the rigid-attachment predict.

        Uses `self._cached_relations` from the previous step — spatial
        relations don't change instantaneously, and the trigger policy
        recomputes them frequently.
        """
        if held_id is None:
            return set()

        manipulated: Set[int] = {held_id}
        for _ in range(8):
            changed = False
            for edge in self._cached_relations:
                if edge.relation_type not in ("in", "on"):
                    continue
                # parent = the contained/on-top object, child = container/
                # base: if child is in the manipulation set, parent rides too.
                if edge.child in manipulated and edge.parent not in manipulated:
                    manipulated.add(edge.parent)
                    changed = True
            if not changed:
                break
        return manipulated

    # --------------------------------------------------------------- #
    #  Relation recomputation
    # --------------------------------------------------------------- #

    def _recompute_relations(self) -> List[RelationEdge]:
        """Build scene graph edges from the collapsed object state.

        Uses soft scores from the relations module. Only 'on' / 'in' edges
        above a threshold are kept as factors.
        """
        collapsed = self.state.collapsed_objects()
        if len(collapsed) < 2:
            return []

        # Wrap objects into the interface compute_spatial_relations_with_scores
        # expects (SceneObject-like).
        class _OrchObj:
            def __init__(self, oid, T, size=None):
                self.id = oid
                self.pose_init = T.copy()
                self.pose_cur = T.copy()
                extent = size if size is not None else np.array([0.05] * 3)
                pts = np.random.uniform(-extent, extent, size=(50, 3)) \
                    + T[:3, 3]
                self._points = pts.astype(np.float32)
                self.child_objs = {}
                self.parent_obj_id = None

        mock_objs = [_OrchObj(oid, pe.T) for oid, pe in collapsed.items()]

        try:
            from scene.object_relation_graph import (
                compute_spatial_relations_with_scores,
            )
            _, scores = compute_spatial_relations_with_scores(
                mock_objs, tolerance=0.02, overlap_threshold=0.2)
        except Exception:
            return []

        raw_edges: List[RelationEdge] = []
        for (parent_id, child_id, rel_type), score in scores.items():
            if rel_type not in ("on", "in"):
                continue
            raw_edges.append(RelationEdge(
                parent=parent_id,
                child=child_id,
                relation_type=rel_type,
                score=score,
            ))
        return self._relation_filter.update(raw_edges)

    # --------------------------------------------------------------- #
    #  Trigger policy
    # --------------------------------------------------------------- #

    def _should_trigger(self,
                        gripper_state: Dict[str, Any],
                        detections: List[Dict[str, Any]]) -> bool:
        last_phase = self.last_state.get("phase", "idle")
        cur_phase = gripper_state.get("phase", "idle")

        # Manipulation events
        if self.trigger.on_grasp and \
                last_phase != "grasping" and cur_phase == "grasping":
            return True
        if self.trigger.on_release and \
                last_phase == "releasing" and cur_phase != "releasing":
            return True

        # New object appeared (compare against known set *before* this step)
        if self.trigger.on_new_object:
            seen_ids = {d.get("id") for d in detections
                        if d.get("id") is not None}
            known = getattr(self, "_known_before_this_step", set())
            if not seen_ids.issubset(known):
                return True

        # Periodic safety net
        if self.trigger.periodic_every_n_frames > 0:
            if (self.frame_count - self.last_opt_frame
                    >= self.trigger.periodic_every_n_frames):
                return True

        return False
