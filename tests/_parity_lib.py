"""Shared core for the parity tests on apple_drop and apple_in_the_tray.

Both ``tests/parity_apple_drop.py`` and ``tests/parity_apple_in_the_tray.py``
share the same loop pattern: drive ``EkfTracker.step()`` over cached
perception, compare per-track world-frame mean / covariance / Bernoulli r
against the JSON state dumps under
``tests/visualization_pipeline/<traj>/ekf_state/``, and report divergence.

Exposed:
  * :func:`compute_parity_stats` --- pure function that returns a stats
    dict. Used by both the CLI parity scripts and the pytest wrappers.
  * :func:`print_parity_report` --- formats the stats dict as the
    canonical CLI output.

Pass criterion (for pytest assertions):
    stats["pose_max_err_world"] == 0.0
    stats["r_max_err"]          == 0.0
    stats["n_oid_mismatch"]     == 0
"""
from __future__ import annotations

import base64
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Camera intrinsics --- shared between both trajectories.
_K = np.array([[554.3827, 0.0, 320.5],
               [0.0, 554.3827, 240.5],
               [0.0, 0.0, 1.0]], dtype=np.float64)


def _load_amcl(p: str) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for line in open(p):
        a = line.strip().split()
        if len(a) != 8:
            continue
        _, tx, ty, tz, qx, qy, qz, qw = map(float, a)
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        out.append(T)
    return out


def _load_idx_pose(p: str) -> Dict[int, np.ndarray]:
    out: Dict[int, np.ndarray] = {}
    if not os.path.exists(p):
        return out
    for line in open(p):
        a = line.strip().split()
        if len(a) != 8:
            continue
        idx = int(a[0])
        tx, ty, tz, qx, qy, qz, qw = map(float, a[1:])
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        T[:3, 3] = [tx, ty, tz]
        out[idx] = T
    return out


def _load_widths(p: str) -> Dict[int, float]:
    out: Dict[int, float] = {}
    if not os.path.exists(p):
        return out
    for k, v in json.load(open(p)).items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        l = v.get("l_gripper_finger_joint")
        r = v.get("r_gripper_finger_joint")
        if l is not None and r is not None:
            out[idx] = float(l) + float(r)
    return out


def _load_joints(p: str) -> Dict[int, Any]:
    if not os.path.exists(p):
        return {}
    return {int(k): v for k, v in json.load(open(p)).items()}


def _load_dets(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    out: List[Dict[str, Any]] = []
    for det in data.get("detections", []):
        mb = det.get("mask", "")
        if not mb:
            continue
        try:
            mb_bytes = base64.b64decode(mb)
            m = np.array(Image.open(BytesIO(mb_bytes)).convert("L"))
            m = (m > 128).astype(np.uint8)
        except Exception:
            continue
        out.append({
            "id": int(det.get("object_id")),
            "label": det.get("label", "unknown"),
            "labels": det.get("labels", {}),
            "mask": m,
            "score": float(det.get("score", 0.0)),
            "mean_score": float(det.get("mean_score", 0.0)),
            "n_obs": int(det.get("n_obs", 0)),
            "box": det.get("box"),
        })
    return out


def _trajectory_paths(traj_name: str) -> Dict[str, str]:
    """Resolve the standard path layout for a parity trajectory."""
    data = os.path.join(_ROOT, "datasets", traj_name)
    viz = os.path.join(_ROOT, "tests", "visualization_pipeline", traj_name)
    det_dir = os.path.join(viz, "perception", "detection_h")
    if not os.path.isdir(det_dir):
        det_dir = os.path.join(data, "detection_h")
    return {
        "data": data,
        "viz": viz,
        "det_dir": det_dir,
        "state_dir": os.path.join(viz, "ekf_state"),
        "relation_cache": os.path.join(viz, "relation_cache"),
        "amcl": os.path.join(data, "pose_txt", "amcl_pose.txt"),
        "T_bc": os.path.join(data, "pose_txt", "T_bc.txt"),
        "T_bg": os.path.join(data, "pose_txt", "ee_pose.txt"),
        "joints": os.path.join(data, "pose_txt", "joints_pose.json"),
        "rgb_tpl": os.path.join(data, "rgb", "rgb_{:06d}.png"),
        "dpt_tpl": os.path.join(data, "depth", "depth_{:06d}.npy"),
        "det_tpl": os.path.join(det_dir, "detection_{:06d}_final.json"),
        "state_tpl": os.path.join(viz, "ekf_state", "frame_{:06d}.json"),
    }


def compute_parity_stats(traj_name: str) -> Dict[str, Any]:
    """Drive ``EkfTracker.step()`` over a cached trajectory and compare
    against the cached state dumps frame-by-frame.

    Returns a dict with:
        n_frames, n_compared, n_oid_match, n_oid_mismatch,
        pose_max_err_world, cov_max_err_world, r_max_err,
        mismatched_frames, per_frame_pose_err
    """
    from ekf_tracker.api import EkfTracker  # noqa: E402 — heavy import

    paths = _trajectory_paths(traj_name)

    slam = _load_amcl(paths["amcl"])
    T_bc = _load_idx_pose(paths["T_bc"])
    T_bg = _load_idx_pose(paths["T_bg"])
    widths = _load_widths(paths["joints"])
    joints = _load_joints(paths["joints"])

    n_frames = len(slam)

    tr = EkfTracker(
        K=_K,
        T_bc=None,                         # main() defaults to None too
        relation_backend="llm",
        relation_cache_dir=paths["relation_cache"],
    )

    n_compared = 0
    n_oid_match = 0
    n_oid_mismatch = 0
    pose_max_err_world = 0.0
    cov_max_err_world = 0.0
    r_max_err = 0.0
    mismatched_frames: List[Tuple[int, str, set]] = []
    per_frame_pose_err: List[Tuple[int, float, Any]] = []

    for idx in range(n_frames):
        rgb_p = paths["rgb_tpl"].format(idx)
        dpt_p = paths["dpt_tpl"].format(idx)
        if not (os.path.exists(rgb_p) and os.path.exists(dpt_p)):
            continue
        rgb = np.array(Image.open(rgb_p).convert("RGB"))
        depth = np.load(dpt_p).astype(np.float32)
        dets = _load_dets(paths["det_tpl"].format(idx))

        sv = tr.step(
            detections=dets,
            rgb=rgb,
            depth=depth,
            slam_pose=slam[idx],
            T_bc=T_bc.get(idx),
            T_bg=T_bg.get(idx),
            gripper_width=widths.get(idx),
            joints=joints.get(idx),
        )

        gt_path = paths["state_tpl"].format(idx)
        if not os.path.exists(gt_path):
            continue
        gt = json.load(open(gt_path))
        gt_tracks = gt.get("tracks_post_update", {}) or {}
        gt_oids = sorted(int(o) for o in gt_tracks.keys())
        api_oids = sorted(sv.objects.keys())

        if gt_oids != api_oids:
            n_oid_mismatch += 1
            mismatched_frames.append((idx, "oid_set",
                                       set(gt_oids) ^ set(api_oids)))
            continue
        n_oid_match += 1

        frame_pose_err = 0.0
        worst_oid = None
        for oid in gt_oids:
            gt_tr = gt_tracks[str(oid)]
            api_obj = sv.objects[oid]

            T_gt_w = np.asarray(gt_tr["T_world"], dtype=np.float64)
            T_api_w = api_obj.pose
            err = float(np.max(np.abs(T_gt_w - T_api_w)))
            pose_max_err_world = max(pose_max_err_world, err)
            if err > frame_pose_err:
                frame_pose_err = err
                worst_oid = (oid, gt_tr["label"])

            cov_gt_w = gt_tr.get("cov_world")
            if cov_gt_w is not None:
                cov_gt_w = np.asarray(cov_gt_w, dtype=np.float64)
                cov_err = float(np.max(np.abs(cov_gt_w - api_obj.cov)))
                cov_max_err_world = max(cov_max_err_world, cov_err)

            r_err = abs(float(gt_tr["r"]) - float(api_obj.r))
            r_max_err = max(r_max_err, r_err)

        per_frame_pose_err.append((idx, frame_pose_err, worst_oid))
        n_compared += 1

    return {
        "trajectory": traj_name,
        "n_frames": n_frames,
        "n_compared": n_compared,
        "n_oid_match": n_oid_match,
        "n_oid_mismatch": n_oid_mismatch,
        "pose_max_err_world": pose_max_err_world,
        "cov_max_err_world": cov_max_err_world,
        "r_max_err": r_max_err,
        "mismatched_frames": mismatched_frames,
        "per_frame_pose_err": per_frame_pose_err,
        "T_bc_count": len(T_bc),
        "T_bg_count": len(T_bg),
        "widths_count": len(widths),
        "joints_count": len(joints),
    }


def print_parity_report(stats: Dict[str, Any],
                         *, post_frame_breakdown_at: int = -1) -> None:
    """Print the canonical parity report. Mirrors the pre-refactor
    output of ``parity_apple_drop.py`` and ``parity_apple_in_the_tray.py``.

    ``post_frame_breakdown_at``: if >= 0, also prints a per-oid worst-
    offender breakdown for frames >= that index (apple_drop's release-
    transition tail, originally hardcoded to 274).
    """
    print(f"frames: {stats['n_frames']}, T_bc map: {stats['T_bc_count']}, "
          f"T_bg map: {stats['T_bg_count']}, widths: {stats['widths_count']}, "
          f"joints: {stats['joints_count']}")
    print()
    print("=" * 60)
    print(f"frames compared:        {stats['n_compared']}")
    print(f"oid-set match frames:   {stats['n_oid_match']}")
    print(f"oid-set mismatch frames:{stats['n_oid_mismatch']}")
    print(f"pose (world) max |Δ|:   {stats['pose_max_err_world']:.3e}")
    print(f"cov  (world) max |Δ|:   {stats['cov_max_err_world']:.3e}")
    print(f"r           max |Δ|:   {stats['r_max_err']:.3e}")
    if stats["mismatched_frames"]:
        print()
        print("first oid-set mismatches:")
        for fr, kind, extra in stats["mismatched_frames"][:5]:
            print(f"  fr {fr}: {kind} symdiff={sorted(extra)}")

    pfpe = stats["per_frame_pose_err"]
    first_micro = next((idx for idx, e, _ in pfpe if e > 1e-6), None)
    first_visible = next((idx for idx, e, _ in pfpe if e > 0.01), None)
    print()
    print(f"first frame with pose Δ > 1e-6:  {first_micro}")
    print(f"first frame with pose Δ > 0.01:  {first_visible}")
    print()

    if post_frame_breakdown_at >= 0:
        after = [(i, e, w) for i, e, w in pfpe if i >= post_frame_breakdown_at]
        if after:
            offenders: Dict[Any, List[Tuple[int, float]]] = {}
            for i, e, w in after:
                offenders.setdefault(w, []).append((i, e))
            print(f"post-{post_frame_breakdown_at} worst-offender oids and frame counts:")
            for w, items in sorted(offenders.items(),
                                       key=lambda kv: -len(kv[1])):
                errs = [e for _, e in items]
                print(f"  oid={w}: {len(items)} frames; max Δ {max(errs):.3e}, "
                      f"min {min(errs):.3e}")
