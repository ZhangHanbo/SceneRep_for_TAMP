#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""State-driven SAM2 + OWL reconciliation (bernoulli_ekf.tex §6 + §9).

Drop-in replacement for the 2-D IoU new-object gate in
``sam2_client.track_dataset``. The old gate asked: "does this OWL box
overlap any propagated SAM2 mask in image space?" That's perception
matching perception. The new gate asks: "does this OWL box correspond
to any tracker-state object that should be visible right now?" --
perception matched against **world-frame state**, with the camera
model used as the visibility predicate.

State per track
---------------
Each track carries a world-frame 3-D centroid belief::

    TrackState3D(
        state_oid:  int   -- identity assigned by this reconciler
        sam2_oid:   int   -- identity used by the SAM2 server
                          -- (kept equal to state_oid here)
        label:      str   -- class label at birth
        mu_w:       (3,)  -- world-frame centroid mean (meters)
        cov_w:      (3,3) -- world-frame covariance (m^2)
        first_frame: int  -- video index where it was born
        scores_by_frame: dict[int, float]
    )

Only translation is tracked. Rotation is not needed for matching, since
OWL boxes are axis-aligned and identity hinges on "is it the same
physical point in the world". Birth cov is a loose prior; each matched
observation shrinks it via a 3-D Kalman-like update against the
back-projected centroid.

Visibility
----------
A track is *in FOV* at frame k iff the camera model projects ``mu_w``
into the image domain with positive depth. Out-of-FOV tracks are kept
in state verbatim (per the user's constraint) -- not deleted, not
penalised, not matched, not missed.

Per-frame loop
--------------
1. Predict step (pass -- translation beliefs don't drift frame-to-frame
   without a dynamics model; cov just inflates by Q_static).
2. Build in-FOV subset of state.
3. Back-project each OWL candidate's mask into the camera frame, then
   to the world via T_cw. Drop candidates without enough valid depth
   pixels.
4. Hungarian on 3-D Mahalanobis cost ``d^2 = nu^T S^-1 nu`` with label
   feasibility and chi^2_3 outer gate (matches the paper's §6 with the
   3-D projection).
5. Matched: 3-D Kalman-like update of ``(mu_w, cov_w)`` against the
   back-projected centroid; record score.
6. Unmatched OWL candidate (in FOV): candidate new prompt -- cluster
   across frames, pick best, seed SAM2.
7. Missed in-FOV track: bookkeeping only; state preserved.

No score threshold gating in this module -- that's OWL-client's job
(``OWL_SCORE_THRESHOLDS`` in server_configs.py, currently 0.15).

The SAM2 session is still driven through ``sam2_client.SAM2Client`` so
HTTP plumbing is unchanged.
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rosbag2dataset.sam2.sam2_client import (
    OwlDet, SAM2Client, PropagatedFrame,
    _load_owl_detections, _load_frames,
)
from rosbag2dataset.server_configs import SAM2_SERVER_URL


# Initial translation covariance for a freshly-born track (m^2 per axis).
# 25 cm 1-sigma: loose enough that the first subsequent match pulls the
# track mean toward the observation, even if the centroid jitters with
# camera viewpoint.
_INIT_SIGMA_M: float = 0.25
_INIT_COV_W: np.ndarray = np.diag([_INIT_SIGMA_M ** 2] * 3)

# Per-frame static-object process noise (m^2 per axis). Loose enough to
# admit slow drift of the back-projected centroid as camera moves.
_Q_STATIC: np.ndarray = np.diag([1e-4] * 3)

# Back-projection noise floor (m^2 per axis) for the measurement cov.
# 10 cm 1-sigma. A mask centroid shifts across camera viewpoints by
# 5-20 cm (different visible portion of the object), and for a moving
# object (held bowl during manipulation) the apparent drift per frame
# can be several cm. This is a conservative floor that keeps Hungarian
# matching forgiving without overriding genuine separation between
# distinct objects. Smaller values (< 1 cm) cause every post-birth
# candidate to be rejected and spawn duplicate tracks per frame.
_R_OBS: np.ndarray = np.diag([1e-2] * 3)   # 10 cm 1-sigma

# Chi^2_3(0.9997) outer gate for 3-D Mahalanobis (matches the paper's
# choice for the 6-D case scaled down by DoF).
_G_OUT_3D: float = 18.47

# Cluster-merge radius for unmatched OWL candidates (world frame). Two
# candidates with the same label within this distance go in the same
# cluster. Also used as the world-space merge threshold before spawning
# a new track -- if any same-label state track is within this distance
# of the candidate, merge into it rather than creating a duplicate.
# Wide enough to cover the reach of a hand-held object through a
# manipulation trajectory (typical ~0.5 m table-top motion) without
# collapsing spatially-distinct same-class objects.
_CLUSTER_MERGE_M: float = 0.50


def _cluster_prompts_3d(
    prompts: List[Tuple[int, Dict[str, Any]]],
    dist_thresh_m: float = _CLUSTER_MERGE_M,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Cluster unmatched (video_idx, candidate) pairs by world-frame
    proximity inside the same class label.

    Two candidates merge when their running-average world centroid is
    within ``dist_thresh_m``. This replaces the legacy 2-D bbox-IoU
    clustering, which collapses under camera motion (same physical
    object projects to different image regions frame-to-frame).

    Representative per cluster = **earliest video_idx** so the track's
    ``first_frame`` reflects the true birth frame.
    """
    from collections import defaultdict
    if not prompts:
        return []
    by_label: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for video_idx, cand in prompts:
        by_label[cand["label"]].append((video_idx, cand))

    out: List[Tuple[int, Dict[str, Any]]] = []
    for label, items in by_label.items():
        items.sort(key=lambda p: p[0])
        clusters: List[List[Tuple[int, Dict[str, Any]]]] = []
        for video_idx, cand in items:
            cw = cand["cw"]
            placed = False
            for cl in clusters:
                avg = np.mean(np.stack([c["cw"] for _, c in cl], axis=0),
                              axis=0)
                if np.linalg.norm(avg - cw) < dist_thresh_m:
                    cl.append((video_idx, cand))
                    placed = True
                    break
            if not placed:
                clusters.append([(video_idx, cand)])
        for cl in clusters:
            rep_idx, rep_cand = min(cl, key=lambda p: p[0])
            out.append((rep_idx, rep_cand))
    return out


# ---------------------------------------------------------------------------
# Per-track state.
# ---------------------------------------------------------------------------

@dataclass
class TrackState3D:
    state_oid: int
    sam2_oid: int
    label: str
    mu_w: np.ndarray            # (3,)
    cov_w: np.ndarray           # (3, 3)
    first_frame: int
    scores_by_frame: Dict[int, float] = field(default_factory=dict)
    frames_observed_in_fov: int = 0
    frames_missed_in_fov: int = 0


# ---------------------------------------------------------------------------
# Camera-model helpers.
# ---------------------------------------------------------------------------

def project_in_fov(p_w: np.ndarray,
                    T_cw: np.ndarray,
                    K: np.ndarray,
                    image_shape: Tuple[int, int]
                    ) -> Optional[Tuple[float, float, float]]:
    """World point -> (u, v, depth_cam) or None if behind camera or outside
    image domain."""
    T_wc = np.linalg.inv(T_cw)
    p_c = T_wc[:3, :3] @ p_w + T_wc[:3, 3]
    z = float(p_c[2])
    if z <= 0.0:
        return None
    uv_h = K @ p_c
    if uv_h[2] <= 0.0:
        return None
    u = float(uv_h[0] / uv_h[2])
    v = float(uv_h[1] / uv_h[2])
    H, W = int(image_shape[0]), int(image_shape[1])
    if u < 0.0 or u >= W or v < 0.0 or v >= H:
        return None
    return (u, v, z)


def backproject_centroid(mask: np.ndarray,
                          depth: np.ndarray,
                          T_cw: np.ndarray,
                          K: np.ndarray,
                          min_valid: int = 20,
                          depth_clip: Tuple[float, float] = (0.1, 5.0),
                          ) -> Optional[np.ndarray]:
    """Back-project a binary mask + depth map to a world-frame centroid.

    Returns (3,) world-frame mean of the masked valid depth pixels, or
    None if fewer than ``min_valid`` pixels pass the depth-range filter.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < min_valid:
        return None
    ds = depth[ys, xs].astype(np.float64)
    v_mask = np.isfinite(ds) & (ds > depth_clip[0]) & (ds < depth_clip[1])
    if int(v_mask.sum()) < min_valid:
        return None
    xs = xs[v_mask]; ys = ys[v_mask]; ds = ds[v_mask]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    Xc = (xs - cx) / fx * ds
    Yc = (ys - cy) / fy * ds
    Zc = ds
    p_c_mean = np.array([float(np.mean(Xc)),
                          float(np.mean(Yc)),
                          float(np.mean(Zc))], dtype=np.float64)
    p_w = T_cw[:3, :3] @ p_c_mean + T_cw[:3, 3]
    return p_w


# ---------------------------------------------------------------------------
# Hungarian on 3-D Mahalanobis (paper §6 at DoF=3).
# ---------------------------------------------------------------------------

def _mahalanobis3(cand_w: np.ndarray,
                   track: TrackState3D) -> Tuple[float, np.ndarray]:
    """Return (d^2, S) where S = cov_w + R_obs."""
    nu = cand_w - track.mu_w
    S = track.cov_w + _R_OBS
    try:
        d2 = float(nu @ np.linalg.solve(S, nu))
    except np.linalg.LinAlgError:
        d2 = float("inf")
    return d2, S


def hungarian_match_3d(
    in_fov_oids: List[int],
    candidates: List[Dict[str, Any]],
    state: Dict[int, TrackState3D],
    G_out: float = _G_OUT_3D,
    enforce_label: bool = True,
) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Label-feasible, chi^2-gated linear assignment of candidates to tracks.

    Returns (match, unmatched_candidate_indices, unmatched_track_oids).
    ``match`` maps ``state_oid -> candidate_index``.
    """
    from scipy.optimize import linear_sum_assignment

    _INF = 1e12
    n_t = len(in_fov_oids)
    n_c = len(candidates)
    if n_t == 0 or n_c == 0:
        return {}, list(range(n_c)), list(in_fov_oids)

    cost = np.full((n_t, n_c), _INF, dtype=np.float64)
    for i, oid in enumerate(in_fov_oids):
        tr = state[oid]
        for l, cand in enumerate(candidates):
            if enforce_label and tr.label != cand["label"]:
                continue
            d2, _ = _mahalanobis3(cand["cw"], tr)
            if d2 > G_out:
                continue
            cost[i, l] = d2

    row, col = linear_sum_assignment(cost)
    match: Dict[int, int] = {}
    matched_rows, matched_cols = set(), set()
    for r, c in zip(row, col):
        if cost[r, c] >= _INF:
            continue
        match[in_fov_oids[r]] = int(c)
        matched_rows.add(r); matched_cols.add(c)
    unmatched_cands = [l for l in range(n_c) if l not in matched_cols]
    unmatched_tracks = [in_fov_oids[i] for i in range(n_t)
                        if i not in matched_rows]
    return match, unmatched_cands, unmatched_tracks


# ---------------------------------------------------------------------------
# State update.
# ---------------------------------------------------------------------------

def _update_track_state(track: TrackState3D, obs_w: np.ndarray) -> None:
    """3-D Kalman-like update of (mu_w, cov_w) against an observation."""
    S = track.cov_w + _R_OBS
    K = track.cov_w @ np.linalg.inv(S)
    nu = obs_w - track.mu_w
    track.mu_w = track.mu_w + K @ nu
    I3 = np.eye(3)
    # Joseph form for PSD safety.
    track.cov_w = ((I3 - K) @ track.cov_w @ (I3 - K).T
                   + K @ _R_OBS @ K.T)
    track.cov_w = 0.5 * (track.cov_w + track.cov_w.T)


def _predict_track_state(track: TrackState3D) -> None:
    """Constant-mean predict: cov inflates by Q_static."""
    track.cov_w = track.cov_w + _Q_STATIC
    track.cov_w = 0.5 * (track.cov_w + track.cov_w.T)


# ---------------------------------------------------------------------------
# Main reconciler.
# ---------------------------------------------------------------------------

@dataclass
class ReconcileInputs:
    frame_ids: List[int]         # dataset frame indices in processing order
    rgb: List[np.ndarray]
    depth: List[np.ndarray]
    cam_poses: List[np.ndarray]  # T_cw per frame (camera in world)
    K: np.ndarray                # (3, 3) intrinsics
    image_shape: Tuple[int, int]  # (H, W)
    owl_by_fid: Dict[int, List[OwlDet]]


def state_driven_reconcile(inputs: ReconcileInputs,
                            sam2: SAM2Client,
                            max_iters: int = 4,
                            max_new_per_iter: int = 20,
                            enforce_label: bool = True,
                            verbose: bool = True
                            ) -> Tuple[Dict[int, TrackState3D],
                                        List[PropagatedFrame]]:
    """Run the iterative state-driven reconciliation.

    Drives a SAM2 session the same way as the old ``track_dataset``: seed
    prompts, propagate, reconcile, repeat. The difference is the
    reconcile step -- it runs a 3-D Hungarian against the tracker state
    with FOV gating, not a 2-D IoU against propagated masks.

    Returns (state, propagated_frames). The caller is expected to write
    the per-frame ``detection_h/*_final.json`` from these outputs.
    """
    sam2.start(inputs.rgb)

    state: Dict[int, TrackState3D] = {}
    next_state_oid = 0

    def _add_new_track(video_idx: int, cand: Dict[str, Any]) -> int:
        nonlocal next_state_oid
        oid = next_state_oid
        next_state_oid += 1
        sam2.add_box(video_idx, oid, cand["box"])
        state[oid] = TrackState3D(
            state_oid=oid,
            sam2_oid=oid,
            label=cand["label"],
            mu_w=cand["cw"].copy(),
            cov_w=_INIT_COV_W.copy(),
            first_frame=video_idx,
        )
        state[oid].scores_by_frame[video_idx] = float(cand["score"])
        return oid

    prop_by_idx: List[PropagatedFrame] = []

    for it in range(max_iters):
        # -- SAM2 propagate (empty when no prompts yet).
        if state:
            t0 = time.time()
            prop_by_idx = sam2.propagate()
            if verbose:
                print(f"[state-reconcile] iter={it} propagate "
                      f"{time.time()-t0:.1f}s tracks={len(state)}")
        else:
            prop_by_idx = []
            if verbose:
                print(f"[state-reconcile] iter={it} empty state -- "
                      f"every OWL detection goes to candidate pool")

        # Reset per-iteration counters so we don't double-count.
        for tr in state.values():
            tr.frames_observed_in_fov = 0
            tr.frames_missed_in_fov = 0

        new_prompts: List[Tuple[int, Dict[str, Any]]] = []

        for video_idx, fid in enumerate(inputs.frame_ids):
            rgb = inputs.rgb[video_idx]
            depth = inputs.depth[video_idx]
            T_cw = inputs.cam_poses[video_idx]

            # Predict every track's cov (translation only); FOV test uses
            # just the mean, so predict has no effect on visibility.
            for tr in state.values():
                _predict_track_state(tr)

            # -- In-FOV subset (visibility predicate, paper §9 simplified).
            in_fov_oids: List[int] = []
            for oid, tr in state.items():
                if project_in_fov(tr.mu_w, T_cw, inputs.K,
                                   inputs.image_shape) is not None:
                    in_fov_oids.append(oid)

            # -- OWL candidates with world-frame centroids.
            owl_dets = inputs.owl_by_fid.get(fid, [])
            candidates: List[Dict[str, Any]] = []
            for owl in owl_dets:
                # Back-project using the propagated mask if available
                # (it's usually a better mask than the OWL box); fall
                # back to the OWL box rendered as a solid mask.
                mask_for_bp = None
                if video_idx < len(prop_by_idx):
                    # We don't yet have a per-label mask here; at this
                    # stage we haven't assigned the OWL det to any track.
                    # So back-project from the OWL box itself.
                    pass
                H, W = inputs.image_shape
                x0, y0, x1, y1 = [int(v) for v in owl.box]
                x0 = max(0, x0); y0 = max(0, y0)
                x1 = min(W, x1); y1 = min(H, y1)
                if x1 <= x0 or y1 <= y0:
                    continue
                m = np.zeros((H, W), dtype=np.uint8)
                m[y0:y1, x0:x1] = 1
                cw = backproject_centroid(m, depth, T_cw, inputs.K)
                if cw is None:
                    continue
                candidates.append({
                    "label": owl.label,
                    "score": owl.score,
                    "box": list(owl.box),
                    "cw": cw,
                    "frame_idx": video_idx,
                })

            # -- Hungarian match (3-D Mahalanobis + label + chi^2 gate).
            match, unmatched_cands, missed = hungarian_match_3d(
                in_fov_oids, candidates, state,
                enforce_label=enforce_label,
            )

            # -- Update matched tracks' 3-D state.
            for oid, cand_idx in match.items():
                cand = candidates[cand_idx]
                _update_track_state(state[oid], cand["cw"])
                state[oid].scores_by_frame[video_idx] = float(cand["score"])
                state[oid].frames_observed_in_fov += 1

            # -- Missed in-FOV tracks: preserve state, count only.
            for oid in missed:
                state[oid].frames_missed_in_fov += 1

            # -- Unmatched candidates: new-prompt candidates.
            for cand_idx in unmatched_cands:
                new_prompts.append((video_idx, candidates[cand_idx]))

        if not new_prompts:
            if verbose:
                print(f"[state-reconcile] converged after {it+1} iter(s)")
            break

        # Cluster unmatched by world-frame centroid proximity within
        # label; representative = earliest frame so first_frame reflects
        # the true birth frame.
        kept = _cluster_prompts_3d(new_prompts,
                                    dist_thresh_m=_CLUSTER_MERGE_M)
        kept.sort(key=lambda p: (p[0], -p[1]["score"]))
        kept = kept[:max_new_per_iter]

        # World-space merge check: even if a candidate failed the per-frame
        # Hungarian (because the matching track was out of FOV that frame),
        # absorb it into the nearest same-label state track when the 3-D
        # distance is within the cluster merge radius. This is what
        # prevents duplicate tracks when an object leaves the FOV and
        # comes back, without relying on SAM2 backward propagation.
        added = 0
        merged = 0
        for video_idx, cand in kept:
            best_d = float("inf")
            best_oid: Optional[int] = None
            for oid, tr in state.items():
                if enforce_label and tr.label != cand["label"]:
                    continue
                d = float(np.linalg.norm(tr.mu_w - cand["cw"]))
                if d < best_d:
                    best_d = d
                    best_oid = oid
            if best_oid is not None and best_d < _CLUSTER_MERGE_M:
                # Absorb into the existing track rather than spawning.
                _update_track_state(state[best_oid], cand["cw"])
                state[best_oid].scores_by_frame[video_idx] = \
                    float(cand["score"])
                merged += 1
            else:
                _add_new_track(video_idx, cand)
                added += 1
        if verbose:
            print(f"[state-reconcile] iter={it} added {added} new tracks, "
                  f"merged {merged} into existing "
                  f"(candidates {len(new_prompts)} -> clustered "
                  f"{len(kept)} kept)")

    # Final propagate if we added tracks in the last iteration and did
    # not subsequently propagate.
    if state:
        prop_by_idx = sam2.propagate()

    return state, prop_by_idx
