"""Geometric support detection at grasp moments.

When the gripper closes on an object, any other live track sitting
*on* it should ride along with the rigid-attachment predict. The
LLM-based relation backend can disagree with the gripper-physical
ground-truth about which object is the support hub (e.g.\ it picks an
unrelated apple instead of the actual tray). This module provides a
cheap geometric fallback that fires only at grasp transitions:

  * For each non-held live track, check whether its world-frame mean
    is within a horizontal radius of the held track AND its z lies in
    a vertical band relative to the held z.
  * If yes, emit a `RelationEdge(parent=track_oid, child=held_oid,
    relation_type="on", score=1.0)` — the same shape the LLM and
    `RelationFilter` expect, so the rest of the pipeline doesn't
    care where the edge came from.

The thresholds are deliberately permissive (catches the apple-in-tray
scene) but can be tightened by callers when they have more scene
prior knowledge.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pose_update.factor_graph import RelationEdge


def detect_support_edges_at_grasp(
        held_oid: int,
        live_tracks: Dict[int, Dict[str, Any]],
        plane_radius_m: float = 0.45,
        height_band_m: Sequence[float] = (+0.05, +0.40),
        excluded_labels: Optional[Sequence[str]] = None,
        existing_edges: Optional[Sequence[RelationEdge]] = None,
        ) -> List[RelationEdge]:
    """Return RelationEdge records for tracks geometrically supported
    by ``held_oid`` at the current frame.

    Args:
        held_oid: oid of the held track (returned by the grasp
            inferrer at grasp onset).
        live_tracks: ``oid → {"xyz_w": (3,) array, "label": str, ...}``
            for every currently-live tracker oid. Typically built from
            ``tracker.state.objects`` + a known label dict.
        plane_radius_m: maximum horizontal Euclidean distance (in the
            world XY plane) between a candidate's world centroid and
            the held track's. Apples spread across a tray sit within
            ~30-45 cm of the tray's centroid.
        height_band_m: ``(z_min, z_max)`` band for ``z_candidate -
            z_held``. Positive values mean the candidate sits above
            the held object. Default ``(-0.10, +0.40)`` admits
            objects sitting on top of the held one.
        excluded_labels: optional list of labels to skip (e.g.\
            ``["bottle"]`` if the dataset has a bottle next to the
            tray that you don't want falsely included).

    Returns:
        ``List[RelationEdge]`` with ``parent=candidate_oid``,
        ``child=held_oid``, ``relation_type="on"``,
        ``score=1.0``. Empty if no candidates pass the gates.
    """
    if held_oid is None or held_oid not in live_tracks:
        return []

    # Defer to the LLM when it has already spoken about the held oid.
    # If the LLM emitted ANY "on"/"in" edge involving the held seed
    # (in either direction), trust its reading and skip the
    # geometric heuristic entirely. The geometric helper assumes
    # "held = support hub", which is wrong when the seed is itself a
    # small object on top of others — the LLM's directional reading
    # disambiguates that.
    if existing_edges:
        held_int = int(held_oid)
        for e in existing_edges:
            try:
                rel = getattr(e, "relation_type", None)
                if rel not in ("in", "on"):
                    continue
                if (int(e.parent) == held_int
                        or int(e.child) == held_int):
                    return []
            except (TypeError, ValueError, AttributeError):
                continue

    held = live_tracks[held_oid]
    held_xyz = np.asarray(held.get("xyz_w") or [0, 0, 0],
                           dtype=np.float64).reshape(3)
    excluded = set(excluded_labels or [])
    z_lo, z_hi = float(height_band_m[0]), float(height_band_m[1])
    plane_r2 = float(plane_radius_m) ** 2

    edges: List[RelationEdge] = []
    for oid, t in live_tracks.items():
        if int(oid) == int(held_oid):
            continue
        if t.get("label") in excluded:
            continue
        xyz = np.asarray(t.get("xyz_w") or [0, 0, 0],
                          dtype=np.float64).reshape(3)
        dx, dy = float(xyz[0] - held_xyz[0]), float(xyz[1] - held_xyz[1])
        if dx * dx + dy * dy > plane_r2:
            continue
        dz = float(xyz[2] - held_xyz[2])
        # Vertical asymmetry: only emit edges for tracks that sit
        # ABOVE the held seed (the support relation is one-directional).
        # An apple SITTING ON a tray has its centroid above the tray's;
        # a cup sitting NEXT TO the held cup does not.
        if dz < z_lo or dz > z_hi:
            continue
        edges.append(RelationEdge(
            parent=int(oid), child=int(held_oid),
            relation_type="on", score=1.0,
        ))
    return edges
