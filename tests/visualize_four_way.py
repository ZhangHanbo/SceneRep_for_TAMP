#!/usr/bin/env python3
"""
2x2 grid visualization for fast-tier debugging on apple_bowl_2:

   +---------------------------------+---------------------------------+
   |  cached SAM detection           |  degenerate Bernoulli (= legacy)|
   |  (detection_h/*.json)           |  -- bit-exact reproduction      |
   +---------------------------------+---------------------------------+
   |  runtime detection              |  full Bernoulli-EKF             |
   |  (detection_boxes/*.json)       |  (Hungarian + Huber + Phi + r)  |
   +---------------------------------+---------------------------------+

Cached SAM = the per-frame SAM2-tracked masks with stable upstream
object_id (the same input the trackers consume). Runtime detection =
the raw OWL output that arrives BEFORE SAM2 temporal tracking is
applied: many boxes per frame, ranked label predictions per box, no
identity continuity. Showing both reveals how noisy the underlying
detector is and why the SAM2 layer is doing real work.

Both tracker panels (degenerate, full) eat the cached SAM2 detections
(detection_h/*.json) -- exactly as in visualize_three_way.py.

Output: tests/visualization_pipeline/<trajectory>/four_way/
    frame_NNNNNN.png
    summary.mp4

Run:
    conda run -n ocmp_test python tests/visualize_four_way.py \
        [--trajectory apple_bowl_2] [--step S] \
        [--alpha 4.4] [--enforce-labels/--no-enforce-labels] \
        [--runtime-min-score 0.01] [--no-video]
"""

from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import os
import sys
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from PIL import Image

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCENEREP_ROOT not in sys.path:
    sys.path.insert(0, SCENEREP_ROOT)

from tests.test_orchestrator_integration import (
    _build_T_co_from_mask, _gripper_state_from_distance, _load_pose_txt, K,
)
from tests.visualize_sam2_observations import (
    _load_detections, overlay_detections, _palette_color, _pngs_to_mp4,
)
from tests.visualize_pipeline import resolve_held_by_proximity
from tests.visualize_degenerate_vs_sam2 import (
    project_world_point, project_cov_xy,
)
from tests.visualize_three_way import (
    _draw_sam2_panel, _draw_tracker_panel, _build_detections,
    _gripper_phase, _make_full_config,
)
from pose_update.orchestrator import (
    TwoTierOrchestrator, TriggerConfig, BernoulliConfig,
)
from pose_update.slam_interface import PassThroughSlam


DATA_BASE = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects",
)


# ─────────────────────────────────────────────────────────────────────
# Runtime detection loader (raw per-frame OWL output, pre-SAM2).
# ─────────────────────────────────────────────────────────────────────

def _load_runtime_detections(json_path: str,
                              min_score: float = 0.01,
                              ) -> List[Dict[str, Any]]:
    """Load `detection_boxes/detection_NNNNNN.json`.

    Each raw entry has {detection: [{label, score}, ...], box, mask}
    where `detection` is OWL's ranked label list for the bounding box.
    We pick the top label per box and discard entries whose top score
    falls below `min_score`. Returns dicts keyed by:
        label, score, box, mask
    (no object_id -- the runtime detector has no temporal tracking).
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        data = json.load(f)

    out: List[Dict[str, Any]] = []
    for entry in data.get("detections", []):
        ranked = entry.get("detection", [])
        if not ranked:
            continue
        top = max(ranked, key=lambda x: float(x.get("score", 0.0)))
        score = float(top.get("score", 0.0))
        if score < min_score:
            continue
        label = top.get("label", "?")
        bb = entry.get("box")
        mask_b64 = entry.get("mask", "")
        mask = None
        if mask_b64:
            try:
                m = np.array(Image.open(BytesIO(
                    base64.b64decode(mask_b64))).convert("L"))
                mask = (m > 128).astype(np.uint8)
            except Exception:
                mask = None
        out.append({
            "label": label,
            "score": score,
            "box": bb,
            "mask": mask,
        })
    return out


def _color_for_label(label: str) -> Tuple[int, int, int]:
    """Stable per-label color so the same class is the same color across
    frames (runtime detector has no track id, so we colorize by label)."""
    h = hashlib.md5(label.encode("utf-8")).hexdigest()
    return _palette_color(int(h[:6], 16))


def _draw_runtime_panel(ax, rgb: np.ndarray,
                         dets: List[Dict[str, Any]],
                         frame_idx: int) -> None:
    out = rgb.copy()
    h, w = out.shape[:2]
    for det in dets:
        color = _color_for_label(det["label"])
        color_rgb = tuple(int(c) for c in color)
        mask = det.get("mask")
        if mask is not None:
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask.astype(bool)
            if mask_bool.any():
                colored = np.zeros_like(out)
                colored[mask_bool] = color_rgb
                out = np.where(mask_bool[..., None],
                                (0.4 * colored + 0.6 * out).astype(np.uint8),
                                out)
        bb = det.get("box")
        if bb is not None and len(bb) == 4:
            x0, y0, x1, y1 = map(int, bb)
            cv2.rectangle(out, (x0, y0), (x1, y1),
                          color_rgb[::-1], 1)
            tag = f"{det['label']} {det['score']:.2f}"
            cv2.putText(out, tag, (x0 + 2, max(y0 - 2, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        color_rgb[::-1], 1, cv2.LINE_AA)
    ax.imshow(out)
    by_label: Dict[str, int] = {}
    for d in dets:
        by_label[d["label"]] = by_label.get(d["label"], 0) + 1
    ax.set_title(
        f"Runtime detection (raw OWL) — frame {frame_idx:04d}   "
        f"{len(dets)} boxes   labels: {dict(sorted(by_label.items()))}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])


# ─────────────────────────────────────────────────────────────────────
# Combined render.
# ─────────────────────────────────────────────────────────────────────

def render_frame(rgb: np.ndarray,
                  cached_dets: List[Dict[str, Any]],
                  runtime_dets: List[Dict[str, Any]],
                  degen_objects: Dict[int, Dict[str, Any]],
                  full_objects: Dict[int, Dict[str, Any]],
                  full_tentative: Dict[int, Dict[str, Any]],
                  degen_orch: TwoTierOrchestrator,
                  full_orch: TwoTierOrchestrator,
                  T_cw: np.ndarray,
                  frame_idx: int,
                  phase: str,
                  out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=110)
    _draw_sam2_panel(axes[0, 0], rgb, cached_dets, frame_idx)
    _draw_tracker_panel(axes[0, 1], rgb, degen_objects, degen_orch, T_cw,
                         frame_idx, phase,
                         title_prefix="Degenerate Bernoulli (= legacy)",
                         show_r=False)
    _draw_runtime_panel(axes[1, 0], rgb, runtime_dets, frame_idx)
    _draw_tracker_panel(axes[1, 1], rgb, full_objects, full_orch, T_cw,
                         frame_idx, phase,
                         title_prefix="Full Bernoulli-EKF",
                         show_r=True,
                         tentative=full_tentative)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Driver.
# ─────────────────────────────────────────────────────────────────────

def run(trajectory: str = "apple_bowl_2",
        n_frames: Optional[int] = None,
        step: int = 3,
        rng_seed: int = 42,
        alpha: float = 4.4,
        enforce_labels: bool = True,
        runtime_min_score: float = 0.01,
        make_video: bool = True) -> None:
    data_root = os.path.join(DATA_BASE, trajectory)
    rgb_dir = os.path.join(data_root, "rgb")
    cached_dir = os.path.join(data_root, "detection_h")
    runtime_dir = os.path.join(data_root, "detection_boxes")
    if not (os.path.isdir(rgb_dir) and os.path.isdir(cached_dir)):
        print(f"[viz] missing rgb/ or detection_h/ in {data_root}",
              file=sys.stderr)
        return
    if not os.path.isdir(runtime_dir):
        print(f"[viz] missing detection_boxes/ in {data_root}; "
              "the runtime panel will be empty", file=sys.stderr)

    out_dir = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline",
                            trajectory, "four_way")
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

    cam_poses = _load_pose_txt(
        os.path.join(data_root, "pose_txt", "camera_pose.txt"))
    ee_poses = _load_pose_txt(
        os.path.join(data_root, "pose_txt", "ee_pose.txt"))
    l_finger = _load_pose_txt(
        os.path.join(data_root, "pose_txt", "l_gripper_pose.txt"))
    r_finger = _load_pose_txt(
        os.path.join(data_root, "pose_txt", "r_gripper_pose.txt"))
    slam_poses = [cam_poses[i] for i in indices]
    slam_cov = np.diag([1e-4] * 6)

    np.random.seed(rng_seed)
    slam_degen = PassThroughSlam(slam_poses, default_cov=slam_cov)
    degen_orch = TwoTierOrchestrator(
        slam_degen,
        trigger=TriggerConfig(periodic_every_n_frames=30),
        verbose=False, rng_seed=rng_seed,
        bernoulli=BernoulliConfig.degeneracy(),
    )

    np.random.seed(rng_seed)
    slam_full = PassThroughSlam(slam_poses, default_cov=slam_cov)
    full_orch = TwoTierOrchestrator(
        slam_full,
        trigger=TriggerConfig(periodic_every_n_frames=30),
        verbose=False, rng_seed=rng_seed,
        bernoulli=_make_full_config(alpha, enforce_labels,
                                     image_shape=(480, 640)),
    )

    print(f"[viz] {len(indices)} frames from {trajectory} "
          f"(total raw: {total}, step={step})")
    print(f"[viz] full config: alpha={alpha}, "
          f"enforce_label_match={enforce_labels}, "
          f"runtime_min_score={runtime_min_score}")

    last_finger_d_d: Optional[float] = None
    last_finger_d_f: Optional[float] = None
    last_phase_d = "idle"
    last_phase_f = "idle"
    held_d: Optional[int] = None
    held_f: Optional[int] = None

    for local_i, idx in enumerate(indices):
        rgb_path = os.path.join(rgb_dir, f"rgb_{idx:06d}.png")
        depth_path = os.path.join(data_root, "depth", f"depth_{idx:06d}.npy")
        cached_path = os.path.join(cached_dir,
                                    f"detection_{idx:06d}_final.json")
        runtime_path = os.path.join(runtime_dir,
                                     f"detection_{idx:06d}.json")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)
        cached_raw = _load_detections(cached_path)
        runtime_raw = _load_runtime_detections(
            runtime_path, min_score=runtime_min_score)
        dets = _build_detections(cached_raw, depth)

        phase_d, last_finger_d_d = _gripper_phase(
            l_finger[idx], r_finger[idx], last_finger_d_d, last_phase_d)
        phase_f, last_finger_d_f = _gripper_phase(
            l_finger[idx], r_finger[idx], last_finger_d_f, last_phase_f)
        last_phase_d, last_phase_f = phase_d, phase_f

        T_ec = ee_poses[idx]
        ee_cam = T_ec[:3, 3]
        if phase_d == "grasping" and held_d is None and dets:
            held_d = resolve_held_by_proximity(dets, depth, ee_cam, cam_K=K)
        elif phase_d == "idle":
            held_d = None
        if phase_f == "grasping" and held_f is None and dets:
            held_f = resolve_held_by_proximity(dets, depth, ee_cam, cam_K=K)
        elif phase_f == "idle":
            held_f = None

        degen_orch.step(rgb, depth, dets,
                         {"phase": phase_d, "held_obj_id": held_d},
                         T_ec=T_ec, T_bg=T_ec)
        full_orch.step(rgb, depth, dets,
                        {"phase": phase_f, "held_obj_id": held_f},
                        T_ec=T_ec, T_bg=T_ec)

        T_cw = cam_poses[idx]
        degen_objs = degen_orch.objects
        full_objs = full_orch.objects
        full_tent = full_orch.tentative_objects

        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        render_frame(rgb, cached_raw, runtime_raw,
                      degen_objs, full_objs, full_tent,
                      degen_orch, full_orch, T_cw, idx, phase_d, out_path)

        if local_i % 10 == 0:
            print(f"  [{local_i+1}/{len(indices)}] frame {idx}: "
                  f"cached={len(cached_raw)}  runtime={len(runtime_raw)}  "
                  f"degen={len(degen_objs)}  "
                  f"full={len(full_objs)}+{len(full_tent)}t  "
                  f"phase={phase_d}")

    if make_video:
        _pngs_to_mp4(out_dir, os.path.join(out_dir, "summary.mp4"))

    print(f"[viz] done. Output: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_bowl_2")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=4.4)
    ap.add_argument("--enforce-labels", dest="enforce_labels",
                    action="store_true", default=True)
    ap.add_argument("--no-enforce-labels", dest="enforce_labels",
                    action="store_false")
    ap.add_argument("--runtime-min-score", type=float, default=0.01,
                    help="drop runtime detections with top-1 OWL score "
                         "below this threshold")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    run(trajectory=args.trajectory,
        n_frames=args.frames, step=args.step,
        rng_seed=args.seed, alpha=args.alpha,
        enforce_labels=args.enforce_labels,
        runtime_min_score=args.runtime_min_score,
        make_video=not args.no_video)
