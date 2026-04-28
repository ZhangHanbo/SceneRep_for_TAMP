"""Robot-agnostic grasp-owner detection.

Given a per-frame snapshot (gripper geometry + state, current
detections, depth image, camera intrinsics, base→world / base→camera
extrinsics), decide which detection (and tracker oid) is currently
being grasped by the gripper.

Three signals are considered, in priority order:

    1. **Perception override** — if any detection carries
       ``det["grasp_owner_pid"]`` or ``det["is_grasped"] == True``
       (typically populated by a downstream grasp-detection algorithm),
       trust that. Used when the perception pipeline already knows.
    2. **Geometric containment** — back-project each detection's mask
       through the depth image, transform to gripper frame, and count
       how many points fall inside the gripper's
       ``inside_volume_g(state)`` AABB. The detection with the highest
       count above a threshold wins. Tiebreaker: smallest mask area
       (the smaller of two co-grasped objects, e.g., apple-on-tray).
    3. **Fallback** — nearest live tracker centroid to the EE position
       within a wider radius, mirroring the legacy behaviour.

The detector is robot-agnostic; it accepts any ``GripperGeometry``
subclass and never inspects robot-specific fields.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pose_update.gripper_geometry import GripperGeometry
from pose_update.icp_pose import _back_project


@dataclass
class HeldDecision:
    """Outcome of one grasp-owner inference call.

    Fields
    ------
    held_oid       : tracker oid of the held object (or None if no track).
    held_pid       : perception id of the chosen detection (or None).
    inside_count   : number of mask points found inside the gripper
                     volume. -1 when chosen via perception override; 0
                     when chosen via fallback.
    source         : 'perception' | 'geometric' | 'fallback' | 'none'.
    note           : optional human-readable diagnostic.
    """
    held_oid:     Optional[int]
    held_pid:     Optional[Any] = None
    inside_count: int = 0
    source:       str = "none"
    note:         str = ""


class GraspOwnerDetector:
    """Decide which object is being grasped each call.

    Parameters
    ----------
    gripper :
        A `GripperGeometry` subclass for the active robot.
    min_inside_count :
        Minimum number of mask points inside the gripper volume for a
        detection to be considered for the geometric signal.
    fallback_radius_m :
        For the legacy fallback: pick the nearest live track within
        this Euclidean radius of the EE position in world frame.
    perception_keys :
        Names checked on each detection dict for the perception
        override signal. The first non-None / non-False match wins.
        Default: ``("grasp_owner_pid", "is_grasped")``.
    """

    def __init__(self,
                 gripper: GripperGeometry,
                 min_inside_count: int = 20,
                 fallback_radius_m: float = 0.30,
                 perception_keys: Tuple[str, ...] =
                     ("grasp_owner_pid", "is_grasped")):
        self.gripper = gripper
        self.min_inside_count = int(min_inside_count)
        self.fallback_radius_m = float(fallback_radius_m)
        self.perception_keys = tuple(perception_keys)

    # ────────── main entry ──────────

    def select(self,
               detections: Optional[List[Dict[str, Any]]],
               depth: Optional[np.ndarray],
               K: Optional[np.ndarray],
               T_wb: np.ndarray,
               T_bg: Optional[np.ndarray],
               T_bc: Optional[np.ndarray],
               joints: Optional[Dict[str, Any]],
               tracker_state: "TrackerState",
               ) -> HeldDecision:
        """Pick the held oid (and the detection it came from)."""
        # 1. Perception override -- if available.
        if detections:
            for d in detections:
                pid = self._perception_override_pid(d)
                if pid is not None:
                    oid = self._map_pid_to_track(tracker_state, pid, d, depth)
                    return HeldDecision(
                        held_oid=oid, held_pid=pid,
                        inside_count=-1, source="perception",
                        note=f"override field on detection (pid={pid})")

        # 2. Geometric containment.
        if (detections and depth is not None and K is not None
                and T_bg is not None and T_bc is not None
                and joints is not None):
            state = self.gripper.state_from_joints(joints)
            if state is not None:
                chosen, count = self._geometric_pick(
                    detections, depth, K, T_wb, T_bg, T_bc, state)
                if chosen is not None:
                    pid = chosen.get("id")
                    oid = self._map_pid_to_track(
                        tracker_state, pid, chosen, depth)
                    return HeldDecision(
                        held_oid=oid, held_pid=pid,
                        inside_count=count, source="geometric",
                        note=f"{count} pts inside gripper volume")

        # 3. Fallback.
        oid = self._nearest_live_track(tracker_state, T_wb, T_bg)
        return HeldDecision(
            held_oid=oid, held_pid=None,
            inside_count=0, source="fallback",
            note=f"nearest-track-to-EE within {self.fallback_radius_m:.2f} m")

    # ────────── helpers ──────────

    def _perception_override_pid(self,
                                  det: Dict[str, Any]) -> Optional[Any]:
        for key in self.perception_keys:
            v = det.get(key)
            if v is None or v is False:
                continue
            if isinstance(v, bool):
                # is_grasped flag → return the detection's own pid.
                return det.get("id")
            return v       # explicit pid value
        return None

    def _geometric_pick(self,
                         detections: List[Dict[str, Any]],
                         depth: np.ndarray,
                         K: np.ndarray,
                         T_wb: np.ndarray,
                         T_bg: np.ndarray,
                         T_bc: np.ndarray,
                         state: Dict[str, Any],
                         ) -> Tuple[Optional[Dict[str, Any]], int]:
        """Score every detection on inside-gripper-count and pick the winner."""
        box = self.gripper.inside_volume_g(state)
        T_wg = T_wb @ T_bg
        T_gw = np.linalg.inv(T_wg)
        # Each entry: (inside_count, inside_frac, mask_area, det)
        scored: List[Tuple[int, float, int, Dict[str, Any]]] = []
        for d in detections:
            mask = d.get("mask")
            if mask is None:
                continue
            pts_cam = _back_project(np.asarray(mask), np.asarray(depth),
                                     np.asarray(K), clean_mask=False)
            if pts_cam is None or len(pts_cam) == 0:
                continue
            N = len(pts_cam)
            homog = np.hstack([pts_cam, np.ones((N, 1))])
            pts_w = (T_wb @ T_bc @ homog.T).T[:, :3]
            homog_w = np.hstack([pts_w, np.ones((N, 1))])
            pts_g = (T_gw @ homog_w.T).T[:, :3]
            count = box.count_inside(pts_g)
            if count >= self.min_inside_count:
                inside_frac = float(count) / float(max(1, N))
                mask_area = int(np.asarray(mask).sum())
                scored.append((count, inside_frac, mask_area, d))
        if not scored:
            return None, 0
        # Pick the detection most contained inside the gripper jaws,
        # not just the one with most points inside. The big tray has
        # many points inside but mostly outside; an apple sitting
        # inside the jaws has frac → 1. Tiebreak by smallest mask
        # only when multiple candidates are essentially fully inside.
        top_count = max(s[0] for s in scored)
        top = [s for s in scored if s[0] >= 0.8 * top_count]
        top.sort(key=lambda s: (-s[1], s[2]))
        chosen_count, _frac, _area, chosen = top[0]
        return chosen, chosen_count

    def _map_pid_to_track(self,
                          tracker_state: "TrackerState",
                          pid: Optional[Any],
                          det: Optional[Dict[str, Any]],
                          depth: Optional[np.ndarray],
                          ) -> Optional[int]:
        """Find the live tracker oid whose ``sam2_tau`` matches ``pid``.

        If no track owns the pid, force-admit the detection: the gripper
        closing is strong physical evidence the object exists, so we
        bypass the policy gate and birth a new track on the spot.
        """
        if pid is None:
            return None
        sam2_tau = tracker_state.sam2_tau()
        for oid, tau in sam2_tau.items():
            try:
                if int(tau) == int(pid):
                    return int(oid)
            except (TypeError, ValueError):
                continue
        # Force-admit path.
        if det is None or depth is None:
            return None
        return tracker_state.force_admit(det, np.asarray(depth))

    def _nearest_live_track(self,
                             tracker_state: "TrackerState",
                             T_wb: np.ndarray,
                             T_bg: Optional[np.ndarray],
                             ) -> Optional[int]:
        if T_bg is None:
            return None
        ee_world = (T_wb @ T_bg)[:3, 3]
        best_oid, best_d = None, float("inf")
        for oid, mu_w in tracker_state.iter_world_centroids():
            d = float(np.linalg.norm(np.asarray(mu_w) - ee_world))
            if d < best_d:
                best_d, best_oid = d, oid
        if best_oid is not None and best_d <= self.fallback_radius_m:
            return int(best_oid)
        return None


class TrackerState:
    """Adapter that exposes the small subset of tracker state the
    detector needs, without depending on a specific tracker class.

    Subclass and implement these three methods to plug in any tracker.
    The default impl wraps ``InstrumentedTracker`` from
    ``tests/visualize_ekf_tracking.py``.
    """
    def sam2_tau(self) -> Dict[int, int]:
        """Map oid → perception (SAM2) id stored on each track."""
        raise NotImplementedError

    def iter_world_centroids(self):
        """Yield ``(oid, mu_w)`` for every live track. ``mu_w`` is a
        length-3 numpy array (world-frame translation)."""
        raise NotImplementedError

    def force_admit(self,
                    det: Dict[str, Any],
                    depth: np.ndarray) -> Optional[int]:
        """Birth a new track for `det`, bypassing policy gates.
        Return the new oid (or None on failure)."""
        raise NotImplementedError


class InstrumentedTrackerState(TrackerState):
    """Adapter for the ``InstrumentedTracker`` in
    ``tests/visualize_ekf_tracking.py``."""

    def __init__(self, tracker):
        self._t = tracker

    def sam2_tau(self) -> Dict[int, int]:
        return dict(self._t.sam2_tau)

    def iter_world_centroids(self):
        T_wb = self._t.state.T_wb
        for oid, b in self._t.state.objects.items():
            mu_b = np.asarray(b.mu_bo)[:3, 3]
            mu_w = (T_wb @ np.append(mu_b, 1.0))[:3]
            yield int(oid), mu_w

    def force_admit(self, det, depth):
        if self._t.pose_est is None:
            return None
        mask = det.get("mask")
        if mask is None:
            return None
        new_oid = self._t._mint_tracker_oid()
        T_co, R_icp, fitness, rmse = self._t.pose_est.estimate(
            oid=int(new_oid), mask=mask, depth=depth, T_co_init=None)
        if T_co is None:
            self._t.pose_est._refs.pop(int(new_oid), None)
            return None
        det_for_birth = dict(det)
        det_for_birth["T_co"] = T_co
        det_for_birth["R_icp"] = R_icp
        det_for_birth["fitness"] = fitness
        det_for_birth["rmse"] = rmse
        det_for_birth["_icp_ok"] = True
        born = self._t._birth(det_for_birth, forced_oid=int(new_oid))
        return int(born) if born is not None else None
