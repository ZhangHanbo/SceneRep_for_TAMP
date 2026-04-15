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

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Any

import numpy as np

from pose_update.slam_interface import (
    PoseEstimate, ParticlePose,
    collect_movable_masks, mask_out_movable, SlamBackend,
)
from pose_update.ekf_se3 import process_noise_for_phase
from pose_update.factor_graph import (
    PoseGraphOptimizer, Observation, RelationEdge, OptimizationResult,
)
from pose_update.rbpf_state import RBPFState


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

    def __init__(self,
                 slam_backend: SlamBackend,
                 trigger: Optional[TriggerConfig] = None,
                 optimizer: Optional[PoseGraphOptimizer] = None,
                 n_particles: int = 32,
                 ess_resample_frac: float = 0.5,
                 iekf_iters: int = 2,
                 rng_seed: Optional[int] = None,
                 verbose: bool = False):
        self.slam = slam_backend
        self.trigger = trigger or TriggerConfig()
        self.optimizer = optimizer or PoseGraphOptimizer()
        self.verbose = verbose

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

        self.frame_count = 0
        self.last_opt_frame = -1
        self.last_state: Dict[str, Any] = {
            "phase": "idle", "held_obj_id": None}

        self._cached_relations: List[RelationEdge] = []
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
        """
        out: Dict[int, Dict[str, Any]] = {}
        for oid, pe in self.state.collapsed_objects().items():
            out[oid] = {
                "T": pe.T,
                "cov": pe.cov,
                "label": self.object_labels.get(oid, "unknown"),
                "frames_since_observation":
                    self.frames_since_obs.get(oid, 0),
                "T_oe": self.T_oe.get(oid),
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

        edges: List[RelationEdge] = []
        for (parent_id, child_id, rel_type), score in scores.items():
            if rel_type not in ("on", "in"):
                continue
            edges.append(RelationEdge(
                parent=parent_id,
                child=child_id,
                relation_type=rel_type,
                score=score,
            ))
        return edges

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
