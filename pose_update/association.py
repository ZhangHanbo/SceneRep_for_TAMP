"""
Data association for the Bernoulli-EKF tracker (bernoulli_ekf.tex §6).

Builds a cost matrix of Mahalanobis squared distances (with a chi^2 outer
gate, per-class feasibility, and optional SAM2 tracklet-ID continuity
bonus) and solves a linear assignment via scipy.optimize.linear_sum_assignment
(Jonker--Volgenant). The result is a dict track_oid -> det_index; tracks
not in the dict are unassigned, detection indices not hit are available
for birth.

A GT-oracle mode is provided for the degeneracy test: when invoked with
``oracle_mode=True``, the function bypasses the cost matrix and uses the
upstream ``det['id']`` field directly. Under oracle_mode, any track whose
oid does not appear in the detection list is marked unassigned, and
detections with an id not currently tracked become births. This replicates
the pre-Bernoulli behaviour of ``TwoTierOrchestrator._fast_tier`` for the
degeneracy check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


# Large finite infeasibility cost (linear_sum_assignment does not accept
# true +inf with all-inf rows/columns without special-case handling).
_INFEASIBLE: float = 1e12


@dataclass
class AssociationResult:
    """Output of a single association pass.

    Attributes:
        match: dict mapping track oid -> detection index it was matched to.
               A track oid NOT in this dict is unassigned.
        unmatched_tracks: list of track oids with no detection.
        unmatched_detections: list of detection indices with no track.
        cost_matrix: (n_tracks, n_dets) cost used (diagnostic).
        gated_pairs: number of (track, det) pairs inside the outer gate
                     that passed the label filter (diagnostic).
    """
    match: Dict[int, int]
    unmatched_tracks: List[int]
    unmatched_detections: List[int]
    cost_matrix: np.ndarray
    gated_pairs: int


def hungarian_associate(
    track_oids: List[int],
    detections: List[Dict[str, Any]],
    innovation_fn: Callable[[int, np.ndarray, np.ndarray],
                            Optional[Tuple[np.ndarray, np.ndarray,
                                           float, float]]],
    track_labels: Mapping[int, str],
    track_tau: Optional[Mapping[int, int]] = None,
    alpha: float = 0.0,
    G_out: float = 25.0,
    enforce_label_match: bool = True,
) -> AssociationResult:
    """Solve the Hungarian assignment of tracks to detections.

    Cost (bernoulli_ekf.tex eq. eq:cost_sam2):
        C[i, l] = d2[i, l] - alpha * 1[tau_l == tau_i]   (feasible)
        C[i, l] = +INFEASIBLE                             (infeasible)

    A pair is infeasible if (a) class labels disagree (when
    ``enforce_label_match``), or (b) d2 exceeds the chi^2 outer gate
    ``G_out`` = 25 (chi^2_6(0.9997)).

    Args:
        track_oids: ordered list of track object ids currently tracked.
        detections: list of per-frame detections in the orchestrator's
            canonical dict format (must contain 'T_co', 'R_icp', 'label';
            'sam2_id' optional).
        innovation_fn: callable (oid, T_co, R_icp) -> (nu, S, d2, log_lik)
            or None if the innovation is undefined (no particle holds oid).
            This is typically `RBPFState.innovation_stats`.
        track_labels: mapping track oid -> class label string.
        track_tau: optional mapping track oid -> last SAM2 tracklet id (as
            an int). Missing or negative values signal "no prior ID" and
            never trigger the alpha bonus. Default None disables the ID
            prior entirely (equivalent to passing an empty mapping).
        alpha: SAM2 continuity bonus in Mahalanobis-squared units. Default
            0 (no bonus). Values in [4.4, 5.9] correspond to q_s in
            [0.9, 0.95] in the paper.
        G_out: chi^2 outer-gate threshold. Pairs beyond this are infeasible.
            Default 25.0 = chi^2_6(0.9997).
        enforce_label_match: if True, only (track, det) pairs with matching
            class labels are feasible. Disable only for label-agnostic
            tests.

    Returns:
        AssociationResult.
    """
    n_tracks = len(track_oids)
    n_dets = len(detections)

    cost = np.full((max(n_tracks, 1), max(n_dets, 1)), _INFEASIBLE,
                   dtype=np.float64)
    log_liks = np.full_like(cost, -np.inf, dtype=np.float64)
    gated = 0

    if n_tracks == 0 or n_dets == 0:
        match: Dict[int, int] = {}
        unmatched_tracks = list(track_oids)
        unmatched_dets = list(range(n_dets))
        return AssociationResult(
            match=match,
            unmatched_tracks=unmatched_tracks,
            unmatched_detections=unmatched_dets,
            cost_matrix=cost,
            gated_pairs=gated,
        )

    # Fill the cost matrix.
    for i, oid in enumerate(track_oids):
        t_label = track_labels.get(oid, None)
        t_tau = None
        if track_tau is not None:
            t_tau_val = track_tau.get(oid, -1)
            t_tau = int(t_tau_val) if (t_tau_val is not None
                                       and t_tau_val >= 0) else None
        for l, det in enumerate(detections):
            if enforce_label_match:
                d_label = det.get("label", None)
                if t_label is not None and d_label is not None \
                        and d_label != t_label:
                    continue
            T_co = det.get("T_co")
            R_icp = det.get("R_icp")
            if T_co is None or R_icp is None:
                continue
            stats = innovation_fn(oid, np.asarray(T_co, dtype=np.float64),
                                  np.asarray(R_icp, dtype=np.float64))
            if stats is None:
                continue
            _nu, _S, d2, log_lik = stats
            if not math.isfinite(d2) or d2 > G_out:
                continue
            # SAM2 continuity bonus.
            bonus = 0.0
            if alpha > 0.0 and t_tau is not None:
                d_tau_raw = det.get("sam2_id")
                if d_tau_raw is not None:
                    d_tau = int(d_tau_raw)
                    if d_tau >= 0 and d_tau == t_tau:
                        bonus = alpha
            cost[i, l] = float(d2) - bonus
            log_liks[i, l] = float(log_lik)
            gated += 1

    # Solve assignment. linear_sum_assignment minimises the total cost.
    # We pad the cost with _INFEASIBLE to handle rectangular inputs.
    row_ind, col_ind = linear_sum_assignment(cost[:n_tracks, :n_dets])

    match: Dict[int, int] = {}
    matched_track_rows = set()
    matched_det_cols = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] >= _INFEASIBLE:
            continue
        oid = track_oids[r]
        match[oid] = int(c)
        matched_track_rows.add(r)
        matched_det_cols.add(c)

    unmatched_tracks = [track_oids[i] for i in range(n_tracks)
                        if i not in matched_track_rows]
    unmatched_dets = [l for l in range(n_dets)
                      if l not in matched_det_cols]

    return AssociationResult(
        match=match,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_dets,
        cost_matrix=cost[:n_tracks, :n_dets],
        gated_pairs=gated,
    )


def oracle_associate(
    track_oids: List[int],
    detections: List[Dict[str, Any]],
) -> AssociationResult:
    """Ground-truth association from the upstream ``det['id']`` field.

    Used by the degeneracy test: pre-Bernoulli ``_fast_tier`` looked up
    tracks directly by ``det['id']``. Calling this function and wiring
    ``AssociationResult.match`` into the same update path reproduces that
    behaviour exactly (substitution A.1 of the reduction in the paper).

    A track oid present in the track list but absent from any detection's
    id is unmatched (routed to the miss branch). A detection whose id is
    NOT in the track list is unmatched (routed to birth).
    """
    n_tracks = len(track_oids)
    n_dets = len(detections)
    track_set = set(track_oids)

    match: Dict[int, int] = {}
    matched_det_cols: set = set()
    for l, det in enumerate(detections):
        d_id = det.get("id")
        if d_id is None:
            continue
        d_id = int(d_id)
        if d_id not in track_set:
            continue
        # Only one detection per track under oracle mode (first wins).
        if d_id in match:
            continue
        match[d_id] = l
        matched_det_cols.add(l)

    unmatched_tracks = [oid for oid in track_oids if oid not in match]
    unmatched_dets = [l for l in range(n_dets) if l not in matched_det_cols]
    return AssociationResult(
        match=match,
        unmatched_tracks=unmatched_tracks,
        unmatched_detections=unmatched_dets,
        cost_matrix=np.zeros((n_tracks, n_dets)),
        gated_pairs=len(match),
    )


def sam2_alpha_from_q_s(q_s: float) -> float:
    """Helper: alpha = 2 log(q_s / (1 - q_s)) (bernoulli_ekf.tex §6.1)."""
    q_s = float(q_s)
    if q_s <= 0.5:
        return 0.0
    if q_s >= 1.0:
        return float("inf")
    return 2.0 * math.log(q_s / (1.0 - q_s))
