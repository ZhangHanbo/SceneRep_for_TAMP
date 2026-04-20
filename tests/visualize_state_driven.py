#!/usr/bin/env python3
"""
Side-by-side: cached SAM (legacy 2-D IoU reconciliation) vs
state-driven reconciliation (3-D Hungarian + FOV gate).

   +---------------------------+---------------------------+
   |  cached SAM detection     |  state-driven reconciler  |
   |  (detection_h/*.json,     |  final state projected    |
   |   legacy pipeline)        |  through camera model;    |
   |                           |  in-FOV markers, missed   |
   |                           |  observations, new-at-k   |
   +---------------------------+---------------------------+

Right panel shows, per frame:
  - state tracks' world-frame centroid projected through T_cw via K
  - solid circle if the track projects inside the image (in FOV)
  - faded "(out)" marker pinned to the right edge for out-of-FOV tracks
  - each OWL candidate this frame:
      solid box colored by matched state_oid (if any)
      dashed box labelled NEW if it spawned a new prompt this frame
      nothing if it was gated out by label / chi^2

Uses a mock SAM2 client (returns empty propagation) so no network
access to the SAM2 server is needed -- the state evolves purely via
OWL + 3-D Hungarian + FOV, which is the whole point of the new design.

Output: tests/visualization_pipeline/<trajectory>/state_driven/
    frame_NNNNNN.png
    summary.mp4

Run:
    conda run -n ocmp_test python tests/visualize_state_driven.py \
        [--trajectory apple_bowl_2] [--step S] [--no-video]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCENEREP_ROOT not in sys.path:
    sys.path.insert(0, SCENEREP_ROOT)

from tests.test_orchestrator_integration import _load_pose_txt, K
from tests.visualize_sam2_observations import (
    _load_detections, overlay_detections, _palette_color, _pngs_to_mp4,
)
from tests.test_state_driven_reconcile import _MockSAM2
from rosbag2dataset.sam2.state_driven_reconcile import (
    state_driven_reconcile, ReconcileInputs,
    project_in_fov, backproject_centroid,
    hungarian_match_3d, TrackState3D,
)
from rosbag2dataset.sam2.sam2_client import _load_owl_detections


DATA_BASE = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects",
)
IMAGE_SHAPE = (480, 640)


# ─────────────────────────────────────────────────────────────────────
# Panel renderers.
# ─────────────────────────────────────────────────────────────────────

def _draw_cached_panel(ax, rgb: np.ndarray,
                        raw_dets: List[Dict[str, Any]],
                        frame_idx: int) -> None:
    overlay = overlay_detections(rgb, raw_dets)
    ax.imshow(overlay)
    ids_here = sorted(int(d["object_id"]) for d in raw_dets
                       if d.get("object_id") is not None)
    ax.set_title(
        f"Cached SAM (2-D IoU pipeline) — frame {frame_idx:04d}   "
        f"{len(raw_dets)} masks   ids: {ids_here}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_state_driven_panel(ax,
                              rgb: np.ndarray,
                              state: Dict[int, TrackState3D],
                              T_cw: np.ndarray,
                              owl_dets: List,      # list[OwlDet]
                              depth: np.ndarray,
                              frame_idx_video: int,
                              frame_idx_dataset: int) -> None:
    """Render the state-driven reconciler's snapshot at this frame.

    Projects every track whose first_frame <= current through T_cw.
    Runs the Hungarian at this frame to derive match/new/miss labels
    (same logic the reconciler uses internally)."""
    ax.imshow(rgb)
    H, W = rgb.shape[:2]

    # A track is "alive" at every frame from its birth forward -- tracks
    # are never deleted, per the state-driven design (they persist even
    # when out of FOV). Birth is the earliest observation; we fall back
    # to first_frame if scores_by_frame is empty (shouldn't happen in
    # practice but keeps the gate robust).
    def _birth_frame(tr) -> int:
        if tr.scores_by_frame:
            return min(tr.scores_by_frame.keys())
        return tr.first_frame

    alive = {oid: tr for oid, tr in state.items()
             if _birth_frame(tr) <= frame_idx_video}

    # Project each alive track; record FOV status.
    projections: Dict[int, Optional[Tuple[float, float, float]]] = {}
    in_fov_oids: List[int] = []
    for oid, tr in alive.items():
        proj = project_in_fov(tr.mu_w, T_cw, K, IMAGE_SHAPE)
        projections[oid] = proj
        if proj is not None:
            in_fov_oids.append(oid)

    # Build candidates from OWL at this frame (same back-projection as
    # the reconciler) and run Hungarian at this frame so we can label
    # match / new / miss.
    candidates: List[Dict[str, Any]] = []
    for owl in owl_dets:
        x0, y0, x1, y1 = [int(v) for v in owl.box]
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W, x1); y1 = min(H, y1)
        if x1 <= x0 or y1 <= y0:
            candidates.append(None)
            continue
        m = np.zeros((H, W), dtype=np.uint8)
        m[y0:y1, x0:x1] = 1
        cw = backproject_centroid(m, depth, T_cw, K)
        candidates.append({
            "label": owl.label, "score": owl.score, "box": list(owl.box),
            "cw": cw,
        } if cw is not None else None)

    valid_cands = [(l, c) for l, c in enumerate(candidates) if c is not None]
    if valid_cands:
        cand_idxs, cand_dicts = zip(*valid_cands)
        match, unmatched_cands, missed = hungarian_match_3d(
            in_fov_oids, list(cand_dicts), alive)
        # Translate indices back to the original candidate list.
        match_global = {oid: cand_idxs[li] for oid, li in match.items()}
        unmatched_global = [cand_idxs[li] for li in unmatched_cands]
    else:
        match_global: Dict[int, int] = {}
        unmatched_global: List[int] = []
        missed = list(in_fov_oids)

    # Reverse the match map (candidate_idx -> oid) for drawing.
    cand_to_oid = {l: oid for oid, l in match_global.items()}

    # Draw every OWL box this frame: color by matched oid; dashed "NEW"
    # if it's the highest-score unmatched cluster rep (we approximate
    # "newly spawned" by showing all unmatched in red dashed).
    for l, owl in enumerate(owl_dets):
        x0, y0, x1, y1 = [int(v) for v in owl.box]
        if l in cand_to_oid:
            oid = cand_to_oid[l]
            color = tuple(c / 255.0 for c in _palette_color(oid))
            ax.add_patch(plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                edgecolor=color, facecolor="none", linewidth=2))
            ax.text(x0 + 3, y0 + 11,
                    f"id:{oid}  {owl.label}  match",
                    fontsize=6, color=color,
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=1.0))
        elif l in unmatched_global:
            ax.add_patch(plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                edgecolor=(0.85, 0.15, 0.15), facecolor="none",
                linewidth=1.5, linestyle="--"))
            ax.text(x0 + 3, y0 + 11,
                    f"{owl.label}  NEW",
                    fontsize=6, color=(0.85, 0.15, 0.15),
                    bbox=dict(facecolor="white", alpha=0.85,
                              edgecolor="none", pad=1.0))

    # Draw every in-FOV alive track's projection (solid dot + ring).
    for oid in in_fov_oids:
        tr = alive[oid]
        proj = projections[oid]
        if proj is None:
            continue
        u, v, _ = proj
        color = tuple(c / 255.0 for c in _palette_color(oid))
        # Ring size indexed by 2-sigma of the projected translation cov.
        sigma_m = float(np.sqrt(max(
            np.diag(tr.cov_w).max(), 1e-6)))
        z_cam = project_in_fov(tr.mu_w, T_cw, K, IMAGE_SHAPE)[2]
        r_px = float(np.clip(2.0 * sigma_m * K[0, 0] / z_cam, 4.0, 60.0))
        ax.add_patch(Ellipse((u, v), width=r_px, height=r_px,
                              edgecolor=color, facecolor="none",
                              linewidth=1.2, alpha=0.9))
        ax.plot(u, v, "o", color=color, markersize=7,
                markeredgecolor="black", markeredgewidth=0.6)
        hit = "match" if oid in match_global else "miss"
        ax.text(u + 9, v - 9,
                f"id:{oid}  {tr.label}  {hit}",
                fontsize=6, color=color,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=1.0))

    # List out-of-FOV alive tracks as a small sidebar on the right edge.
    out_of_fov = [oid for oid in alive
                   if oid not in in_fov_oids]
    if out_of_fov:
        lines = ["Out of FOV:"]
        for oid in sorted(out_of_fov):
            lines.append(f"  id:{oid} {alive[oid].label}")
        ax.text(
            W - 10, 12, "\n".join(lines),
            fontsize=6, color=(0.3, 0.3, 0.3),
            ha="right", va="top",
            bbox=dict(facecolor="white", alpha=0.85,
                      edgecolor="none", pad=2.0))

    ax.set_title(
        f"State-driven reconciler — frame {frame_idx_dataset:04d}   "
        f"alive={len(alive)}  in-FOV={len(in_fov_oids)}  "
        f"matched={len(match_global)}  missed={len(in_fov_oids) - len(match_global)}  "
        f"new={len(unmatched_global)}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def render_frame(rgb: np.ndarray,
                  cached_dets: List[Dict[str, Any]],
                  state: Dict[int, TrackState3D],
                  T_cw: np.ndarray,
                  owl_dets: List,
                  depth: np.ndarray,
                  frame_idx_video: int,
                  frame_idx_dataset: int,
                  out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=110)
    _draw_cached_panel(axes[0], rgb, cached_dets, frame_idx_dataset)
    _draw_state_driven_panel(axes[1], rgb, state, T_cw, owl_dets, depth,
                              frame_idx_video, frame_idx_dataset)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Driver.
# ─────────────────────────────────────────────────────────────────────

def run(trajectory: str = "apple_bowl_2",
        n_frames: Optional[int] = None,
        step: int = 3,
        max_iters: int = 4,
        make_video: bool = True) -> None:
    data_root = os.path.join(DATA_BASE, trajectory)
    rgb_dir = os.path.join(data_root, "rgb")
    depth_dir = os.path.join(data_root, "depth")
    cached_dir = os.path.join(data_root, "detection_h")
    runtime_dir = os.path.join(data_root, "detection_boxes")
    pose_path = os.path.join(data_root, "pose_txt", "camera_pose.txt")
    if not all(os.path.isdir(d) for d in [rgb_dir, depth_dir,
                                            cached_dir, runtime_dir]):
        print(f"[viz] missing directories under {data_root}",
              file=sys.stderr)
        return

    out_dir = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline",
                            trajectory, "state_driven")
    if os.path.isdir(out_dir):
        for f in glob.glob(os.path.join(out_dir, "frame_*.png")):
            os.remove(f)
    os.makedirs(out_dir, exist_ok=True)

    rgb_files = sorted(f for f in os.listdir(rgb_dir) if f.endswith(".png"))
    indices_all = [int(f[4:10]) for f in rgb_files]
    total = len(indices_all)
    if n_frames is not None:
        indices_all = indices_all[:n_frames]
    indices = indices_all[::step]

    print(f"[viz] {len(indices)} frames from {trajectory} "
          f"(total raw: {total}, step={step})")

    # Load RGB, depth, camera poses for the subsampled indices.
    rgb_list: List[np.ndarray] = []
    depth_list: List[np.ndarray] = []
    for fid in indices:
        rgb_list.append(cv2.cvtColor(
            cv2.imread(os.path.join(rgb_dir, f"rgb_{fid:06d}.png")),
            cv2.COLOR_BGR2RGB))
        depth_list.append(np.load(
            os.path.join(depth_dir, f"depth_{fid:06d}.npy")
        ).astype(np.float32))

    pose_per_fid: Dict[int, np.ndarray] = {}
    with open(pose_path) as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            fid_i, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            pose_per_fid[int(fid_i)] = T
    cam_poses = [pose_per_fid[fid] for fid in indices]

    # Load OWL detections keyed by dataset fid (state-driven reconciler
    # looks them up by fid, not by video index).
    owl_by_fid = _load_owl_detections(runtime_dir, min_score=0.15)

    inputs = ReconcileInputs(
        frame_ids=list(indices), rgb=rgb_list, depth=depth_list,
        cam_poses=cam_poses, K=K, image_shape=IMAGE_SHAPE,
        owl_by_fid=owl_by_fid,
    )

    # Run the reconciler once to convergence (mock SAM2 -> no masks).
    sam2 = _MockSAM2()
    print(f"[viz] running state-driven reconciler (mock SAM2, "
          f"max_iters={max_iters})...")
    state, _prop = state_driven_reconcile(
        inputs, sam2, max_iters=max_iters, verbose=True)
    print(f"[viz] final state: {len(state)} tracks  "
          f"labels: {sorted({t.label for t in state.values()})}")

    # Render per frame.
    for video_idx, fid in enumerate(indices):
        rgb = rgb_list[video_idx]
        depth = depth_list[video_idx]
        T_cw = cam_poses[video_idx]
        cached_raw = _load_detections(
            os.path.join(cached_dir, f"detection_{fid:06d}_final.json"))
        owl_dets = owl_by_fid.get(fid, [])
        out_path = os.path.join(out_dir, f"frame_{fid:06d}.png")
        render_frame(rgb, cached_raw, state, T_cw, owl_dets, depth,
                      video_idx, fid, out_path)
        if video_idx % 10 == 0:
            in_fov = sum(1 for oid, tr in state.items()
                          if tr.first_frame <= video_idx
                          and project_in_fov(tr.mu_w, T_cw, K,
                                              IMAGE_SHAPE) is not None)
            print(f"  [{video_idx+1}/{len(indices)}] frame {fid}: "
                  f"cached={len(cached_raw)}  alive={sum(1 for t in state.values() if t.first_frame<=video_idx)}  in-FOV={in_fov}")

    if make_video:
        _pngs_to_mp4(out_dir, os.path.join(out_dir, "summary.mp4"))

    print(f"[viz] done. Output: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_bowl_2")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--max-iters", type=int, default=4)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    run(trajectory=args.trajectory,
        n_frames=args.frames, step=args.step,
        max_iters=args.max_iters,
        make_video=not args.no_video)
