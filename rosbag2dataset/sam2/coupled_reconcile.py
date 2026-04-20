#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Coupled online OWL + SAM2 + EKF-tracker pipeline.

The orchestrator's Bernoulli-EKF state (RBPFState + existence r +
scene graph + sam2_tau) is the single source of truth. Perception is
its driver: each frame we build observations and call ``orch.step``,
which internally runs the 7-step §11 algorithm (predict -> associate
-> update -> birth -> prune -> emit). After ``orch.step`` we diff the
orchestrator's track-id set to detect births, and seed SAM2 with a
fresh prompt for each. After the full video is processed, a single
``sam2.propagate()`` refines masks for every track; detection_h is
written from that.

Contrast with the old ``state_driven_reconcile`` (local TrackState3D,
3-D centroid-only matching) and the legacy 2-D IoU reconciliation:

  old state-driven        | coupled (this file)
  ------------------------+----------------------------------------
  TrackState3D            | orchestrator's RBPFState (SE(3))
  mu_w, cov_w (3, 3)      | per-particle (T, P) on SE(3)
  no rotation             | full 6-D Mahalanobis via innovation_stats
  no existence            | r tracked and updated via r_assoc/r_miss
  no rigid-attach         | rigid_attachment_predict for held objects
                          | + scene-graph transitive closure carries
                          |   passengers (apple-in-bowl)
  iterative (max_iters=4) | single online pass
  local clustering        | orchestrator births per-detection

Per frame algorithm:

    1. Build detections from OWL boxes:
       for each OWL box (class/label/box/score):
         box -> solid binary mask
         back-project mask + depth -> centroid in CAMERA frame
         T_co = [I | centroid]
         detection dict = {id=None, sam2_id=None, label, mask, score,
                           T_co, R_icp, box}

    2. Snapshot the orchestrator's known track id set BEFORE the step.

    3. orch.step(rgb, depth, detections, gripper_state, T_ec, T_bg)
       internally:
         - SLAM ingest (PassThroughSlam uses the preset camera poses)
         - predict (EKF + Phi + rigid-attach for held-object set)
         - Hungarian 6-D Mahalanobis + chi^2 gate (paper §6)
         - EKF update on matches with Huber re-weighting
         - r_assoc on matches; r_miss on unassigned tracks
         - birth from unmatched detections
         - prune at r_min

    4. Diff track ids -> newly-birthed tracks. For each new_oid, find
       the OWL det that triggered it (greedy same-label closest-world-
       position matching) and seed SAM2 with ``sam2.add_box(video_idx,
       new_oid, det.box)``.

    5. After all frames: ``sam2.propagate()`` to get per-frame masks,
       then write detection_h/*_final.json.

Requires: the orchestrator be configured with a ``BernoulliConfig``
and a ``PassThroughSlam`` backend pre-loaded with the frame-ordered
camera poses. ``make_coupled_orchestrator`` provides a sensible
default setup.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rosbag2dataset.sam2.sam2_client import (
    OwlDet, SAM2Client, PropagatedFrame,
)
from pose_update.orchestrator import (
    TwoTierOrchestrator, TriggerConfig, BernoulliConfig,
)


# ---------------------------------------------------------------------------
# Detection construction helpers.
# ---------------------------------------------------------------------------

def _box_to_solid_mask(box: List[int],
                        image_shape: Tuple[int, int]) -> np.ndarray:
    H, W = int(image_shape[0]), int(image_shape[1])
    m = np.zeros((H, W), dtype=np.uint8)
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = 1
    return m


def _backproject_camera_centroid(
    mask: np.ndarray, depth: np.ndarray, K: np.ndarray,
    min_valid: int = 20,
    depth_clip: Tuple[float, float] = (0.1, 5.0),
) -> Optional[np.ndarray]:
    """Masked depth -> CAMERA-frame centroid (3,). None if depth coverage
    is too sparse."""
    ys, xs = np.where(mask > 0)
    if len(xs) < min_valid:
        return None
    ds = depth[ys, xs].astype(np.float64)
    v = np.isfinite(ds) & (ds > depth_clip[0]) & (ds < depth_clip[1])
    if int(v.sum()) < min_valid:
        return None
    xs, ys, ds = xs[v], ys[v], ds[v]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    Xc = (xs - cx) / fx * ds
    Yc = (ys - cy) / fy * ds
    return np.array([float(np.mean(Xc)),
                      float(np.mean(Yc)),
                      float(np.mean(ds))], dtype=np.float64)


# Default measurement noise for the EKF update.
# Translation: 1 cm 1-sigma (back-projected centroid is accurate at that
# scale when the depth image is clean).
# Rotation: 1 rad 1-sigma (we have no rotation evidence from a mask
# centroid; this floor keeps the EKF from snapping state rotation to
# the observed identity, but is tight enough that the pose likelihood
# peak L = (2π)^-3 |S|^-1/2 stays O(10^3), so on a clean match
# L/lambda_c is large enough to push the Bernoulli existence r -> 1
# under the paper's (eq:r_assoc). Earlier 10 rad^2 was too diffuse --
# L shrank to ~0.1 and r decreased on each match.
_R_ICP_DEFAULT: np.ndarray = np.diag(
    [1e-4] * 3 +      # 1 cm 1-sigma translation
    [1.0]  * 3        # 1 rad 1-sigma rotation
)


def _build_detections_for_orch(
    owl_dets: List[OwlDet],
    depth: np.ndarray,
    K: np.ndarray,
    image_shape: Tuple[int, int],
    R_icp: Optional[np.ndarray] = None,
) -> List[Dict[str, Any]]:
    """One OWL detection -> one orchestrator-format detection dict.

    Each output carries an 'owl_idx' field (frame-local) that lets
    callers map orchestrator tracks back to the OWL box that spawned
    them without relying on det['id'] (which the orchestrator treats
    as the upstream tracklet id; we want it None here so the Bernoulli
    Hungarian decides identity).
    """
    if R_icp is None:
        R_icp = _R_ICP_DEFAULT
    out: List[Dict[str, Any]] = []
    for owl_idx, owl in enumerate(owl_dets):
        mask = _box_to_solid_mask(owl.box, image_shape)
        cen = _backproject_camera_centroid(mask, depth, K)
        if cen is None:
            continue
        T_co = np.eye(4, dtype=np.float64)
        T_co[:3, 3] = cen
        out.append({
            # Orchestrator-consumed fields:
            "id":       None,   # Hungarian mints fresh ids; keep None
            "sam2_id":  None,   # no SAM2-id continuity bonus here
            "label":    owl.label,
            "mask":     mask,
            "score":    float(owl.score),
            "T_co":     T_co,
            "R_icp":    R_icp.copy(),
            "box":      list(owl.box),
            "fitness":  float(max(0.3, owl.score)),
            "rmse":     0.005,
            # Private bookkeeping for birth reverse-lookup:
            "owl_idx":  owl_idx,
        })
    return out


# ---------------------------------------------------------------------------
# Orchestrator factory.
# ---------------------------------------------------------------------------

def make_coupled_orchestrator(
    cam_poses: List[np.ndarray],
    bernoulli: Optional[BernoulliConfig] = None,
    trigger: Optional[TriggerConfig] = None,
    n_particles: int = 32,
    rng_seed: int = 42,
) -> TwoTierOrchestrator:
    """Set up a TwoTierOrchestrator for coupled perception use.

    The SLAM backend is a PassThroughSlam pre-loaded with the camera
    poses (treated as T_wb under the convention of the Fetch test
    harness -- camera == base frame). The Bernoulli config defaults
    to Hungarian association + loose init covariance + no r pruning
    at the output layer (so every track the orchestrator knows about
    is reachable; callers can filter on r themselves).
    """
    from pose_update.slam_interface import PassThroughSlam

    if bernoulli is None:
        bernoulli = BernoulliConfig(
            association_mode="hungarian",
            p_s=1.0, p_d=0.9, alpha=4.4,
            lambda_c=1.0, lambda_b=1.0,
            r_conf=0.0,          # emit every track; leave filtering to caller
            r_min=0.0,           # never prune in perception layer
            G_in=12.59, G_out=25.0,
            P_max=None,          # no covariance saturation
            enable_visibility=False,  # FOV gating is implicit via Hungarian d^2
            enable_huber=True,
            init_cov_from_R=False,    # use _INIT_OBJ_COV = diag([0.05]*6)
            enforce_label_match=True,
        )

    if trigger is None:
        trigger = TriggerConfig(periodic_every_n_frames=30)

    slam = PassThroughSlam(
        cam_poses, default_cov=np.diag([1e-4] * 6))
    return TwoTierOrchestrator(
        slam,
        trigger=trigger,
        verbose=False,
        n_particles=n_particles,
        rng_seed=rng_seed,
        bernoulli=bernoulli,
    )


# ---------------------------------------------------------------------------
# Birth reverse-lookup: which OWL det spawned which new track.
# ---------------------------------------------------------------------------

def _match_new_tracks_to_dets(
    new_oids: List[int],
    orch: TwoTierOrchestrator,
    detections: List[Dict[str, Any]],
    T_cw: np.ndarray,
    max_dist_m: float = 0.25,
) -> Dict[int, Dict[str, Any]]:
    """Greedy one-to-one matching by same-label closest-world-position.

    Newly-birthed tracks have mean T_wo = T_wb @ T_co of the triggering
    detection (the orchestrator's birth path does exactly
    ``state.ensure_object(d_id, T_co, init_cov)`` with per-particle
    ``mu = T_wb @ T_co``). So the new track's collapsed T, at the frame
    it was born, equals ``T_cw @ T_co_of_spawning_det`` within particle
    spread. Find that det per new_oid.
    """
    if not new_oids or not detections:
        return {}
    collapsed = orch.state.collapsed_objects()

    # Cache per-det world position by label.
    det_worlds: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    for det in detections:
        t_w = (T_cw @ det["T_co"])[:3, 3]
        det_worlds.append((t_w, det))

    used: set = set()
    out: Dict[int, Dict[str, Any]] = {}
    for new_oid in new_oids:
        pe = collapsed.get(new_oid)
        if pe is None:
            continue
        t_new = pe.T[:3, 3]
        label = orch.object_labels.get(new_oid, "?")
        best_d = float("inf")
        best_idx: Optional[int] = None
        for i, (t_w, det) in enumerate(det_worlds):
            if i in used:
                continue
            if det.get("label") != label:
                continue
            d = float(np.linalg.norm(t_w - t_new))
            if d < best_d:
                best_d = d
                best_idx = i
        if best_idx is not None and best_d <= max_dist_m:
            used.add(best_idx)
            out[new_oid] = det_worlds[best_idx][1]
    return out


# ---------------------------------------------------------------------------
# Main coupled reconciler.
# ---------------------------------------------------------------------------

@dataclass
class CoupledInputs:
    frame_ids: List[int]
    rgb: List[np.ndarray]
    depth: List[np.ndarray]
    cam_poses: List[np.ndarray]  # T_cw per frame (camera in world)
    K: np.ndarray
    image_shape: Tuple[int, int]
    owl_by_fid: Dict[int, List[OwlDet]]
    # Optional proprioception.
    gripper_states: Optional[List[Dict[str, Any]]] = None
    T_ec_list: Optional[List[np.ndarray]] = None
    T_bg_list: Optional[List[np.ndarray]] = None


def coupled_reconcile(
    orch: TwoTierOrchestrator,
    inputs: CoupledInputs,
    sam2: SAM2Client,
    verbose: bool = True,
) -> List[PropagatedFrame]:
    """Single-pass online coupled perception + Bernoulli-EKF tracker.

    Orchestrator must be Bernoulli-configured. All identity and state
    lives in the orchestrator's RBPFState + dicts; this function only
    drives ``orch.step`` per frame and seeds SAM2 on births.
    """
    if orch.bernoulli is None:
        raise ValueError(
            "coupled_reconcile requires a Bernoulli-configured "
            "orchestrator; call make_coupled_orchestrator() first")

    sam2.start(inputs.rgb)

    t_start = time.time()
    total_dets = 0
    total_births = 0

    for video_idx, fid in enumerate(inputs.frame_ids):
        rgb = inputs.rgb[video_idx]
        depth = inputs.depth[video_idx]
        T_cw = inputs.cam_poses[video_idx]
        owl_dets = inputs.owl_by_fid.get(fid, [])

        detections = _build_detections_for_orch(
            owl_dets, depth, inputs.K, inputs.image_shape)
        total_dets += len(detections)

        # Snapshot track id set BEFORE the step. Births are the diff.
        known_before = set(orch.object_labels.keys())

        gripper_state = {"phase": "idle", "held_obj_id": None}
        T_ec = None
        T_bg = None
        if inputs.gripper_states is not None:
            gripper_state = inputs.gripper_states[video_idx]
        if inputs.T_ec_list is not None:
            T_ec = inputs.T_ec_list[video_idx]
        if inputs.T_bg_list is not None:
            T_bg = inputs.T_bg_list[video_idx]

        # Orchestrator runs the full 7-step §11 algorithm.
        orch.step(rgb, depth, detections, gripper_state,
                   T_ec=T_ec, T_bg=T_bg)

        # Births = newly-present track ids.
        known_after = set(orch.object_labels.keys())
        new_oids = sorted(known_after - known_before)
        total_births += len(new_oids)

        # Seed SAM2 for each new track with its triggering OWL box.
        if new_oids:
            new_to_det = _match_new_tracks_to_dets(
                new_oids, orch, detections, T_cw)
            for new_oid, det in new_to_det.items():
                sam2.add_box(video_idx, new_oid, det["box"])

        if verbose and (video_idx % 10 == 0 or video_idx == len(inputs.frame_ids) - 1):
            tracked = len(orch.object_labels)
            print(f"[coupled] [{video_idx+1}/{len(inputs.frame_ids)}] "
                  f"fid={fid}  owl={len(owl_dets)}  dets={len(detections)}  "
                  f"tracked={tracked}  new={len(new_oids)}")

    if verbose:
        dt = time.time() - t_start
        print(f"[coupled] processed {len(inputs.frame_ids)} frames in "
              f"{dt:.1f}s  total dets={total_dets}  total births="
              f"{total_births}  final tracks={len(orch.object_labels)}")

    # Final SAM2 propagate to get masks for all seeded prompts.
    if orch.object_labels:
        t0 = time.time()
        prop = sam2.propagate()
        if verbose:
            print(f"[coupled] sam2.propagate {time.time()-t0:.1f}s  "
                  f"returned masks for {len(prop)} frames")
    else:
        prop = []
    return prop
