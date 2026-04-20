#!/usr/bin/env python3
"""
Side-by-side observation vs tracked state, using the Bernoulli-EKF in
DEGENERACY mode (which reproduces the legacy `_fast_tier` bit-exactly).

Each output PNG has two panels on identical RGB backgrounds:

  * LEFT -- SAM2 detections (the tracker's input): masks + bboxes + text
    `id:N label s=0.NN`. Color keyed on upstream `object_id` modulo 20.

  * RIGHT -- tracked object instances: each track's world-frame pose
    projected back into the image with K + T_cw, with ellipse axes set
    by the projected 2-sigma translation covariance and a label
    `id:N label r=0.NN  fso=F`. Same color palette as LEFT, so the same
    upstream id has the same color on both sides (since oracle mode
    matches tracker oid to upstream id 1:1).

Output: tests/visualization_pipeline/<trajectory>/degenerate_vs_sam2/
    frame_NNNNNN.png
    summary.mp4

Run:
    conda run -n ocmp_test python tests/visualize_degenerate_vs_sam2.py \
        [--trajectory apple_bowl_2] [--frames N] [--step S] [--no-video]
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

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCENEREP_ROOT not in sys.path:
    sys.path.insert(0, SCENEREP_ROOT)

from tests.test_orchestrator_integration import (
    DATA_ROOT as _DEFAULT_DATA_ROOT, K,
)
from tests.visualize_sam2_observations import (
    _load_detections, overlay_detections, _palette_color, _pngs_to_mp4,
)
from tests.test_orchestrator_integration import (
    _build_T_co_from_mask, _gripper_state_from_distance, _load_pose_txt,
)
from pose_update.orchestrator import (
    TwoTierOrchestrator, TriggerConfig, BernoulliConfig,
)
from pose_update.slam_interface import PassThroughSlam


DATA_BASE = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects"
)


# ─────────────────────────────────────────────────────────────────────
# Projection helpers (world -> image).
# ─────────────────────────────────────────────────────────────────────

def project_world_point(p_w: np.ndarray, T_cw: np.ndarray,
                         K_mat: np.ndarray
                         ) -> Optional[Tuple[float, float, float]]:
    T_wc = np.linalg.inv(T_cw)
    p_c = T_wc @ np.append(p_w, 1.0)
    depth = float(p_c[2])
    if depth <= 0:
        return None
    uv = K_mat @ p_c[:3]
    return (float(uv[0] / uv[2]), float(uv[1] / uv[2]), depth)


def project_cov_xy(T_w: np.ndarray, cov: np.ndarray,
                    T_cw: np.ndarray, K_mat: np.ndarray,
                    ) -> Tuple[float, float]:
    cov_t = cov[:3, :3]
    eigs = np.linalg.eigvalsh(0.5 * (cov_t + cov_t.T))
    eigs = np.clip(eigs, 0.0, None)
    sigma_m = float(np.sqrt(max(eigs.max(), 0.0)))
    sigma_m = max(sigma_m, 1e-3)

    T_wc = np.linalg.inv(T_cw)
    p_c = T_wc @ np.append(T_w[:3, 3], 1.0)
    depth = float(p_c[2])
    if depth <= 0:
        return (18.0, 18.0)

    fx = float(K_mat[0, 0])
    sigma_px = 2.0 * sigma_m * fx / depth
    sigma_px = float(np.clip(sigma_px, 4.0, 120.0))
    return (sigma_px, sigma_px)


# ─────────────────────────────────────────────────────────────────────
# Rendering.
# ─────────────────────────────────────────────────────────────────────

def draw_tracks_panel(ax,
                       rgb: np.ndarray,
                       objects: Dict[int, Dict[str, Any]],
                       orch: TwoTierOrchestrator,
                       T_cw: np.ndarray,
                       frame_idx: int,
                       phase: str) -> None:
    ax.imshow(rgb)
    H, W = rgb.shape[:2]
    n_tracks = len(objects)
    ax.set_title(
        f"Degenerate tracker — frame {frame_idx:04d} [{phase}]   "
        f"{n_tracks} tracks",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    legend_handles = []
    for oid in sorted(objects.keys()):
        info = objects[oid]
        T_w = np.asarray(info["T"], dtype=np.float64)
        cov = np.asarray(info["cov"], dtype=np.float64)
        label = info.get("label", "?")
        r_val = orch.existence.get(oid)
        fso = orch.frames_since_obs.get(oid, -1)

        proj = project_world_point(T_w[:3, 3], T_cw, K)
        if proj is None:
            continue
        u, v, _ = proj
        if not (0 <= u < W and 0 <= v < H):
            continue

        color = tuple(c / 255.0 for c in _palette_color(oid))
        a, b = project_cov_xy(T_w, cov, T_cw, K)
        ax.add_patch(Ellipse((u, v), width=a, height=b,
                              edgecolor=color, facecolor="none",
                              linewidth=1.6, alpha=0.9))
        ax.plot(u, v, "o", color=color, markersize=8,
                markeredgecolor="black", markeredgewidth=0.6)

        text = f"id:{oid}  {label}"
        if r_val is not None:
            text += f"  r={r_val:.2f}"
        text += f"  fso={fso}"
        ax.text(u + 10, v - 10, text, fontsize=7, color=color,
                bbox=dict(facecolor="white", alpha=0.8,
                          edgecolor="none", pad=1.0))

        legend_handles.append(mpatches.Patch(
            color=color, label=f"id:{oid}  {label}"))

    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=7,
                  loc="lower right", framealpha=0.85)


def render_frame(rgb: np.ndarray,
                  detections: List[Dict[str, Any]],
                  tracker_objects: Dict[int, Dict[str, Any]],
                  orch: TwoTierOrchestrator,
                  T_cw: np.ndarray,
                  frame_idx: int,
                  phase: str,
                  out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

    # Left: SAM2 observations.
    overlay = overlay_detections(rgb, detections)
    axes[0].imshow(overlay)
    ids_here = sorted(int(d["object_id"]) for d in detections
                       if d.get("object_id") is not None)
    axes[0].set_title(
        f"SAM2 observations — frame {frame_idx:04d}   "
        f"{len(detections)} masks   ids: {ids_here}",
        fontsize=10,
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Right: tracker projections.
    draw_tracks_panel(axes[1], rgb, tracker_objects, orch, T_cw,
                       frame_idx, phase)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Orchestrator driver.
# ─────────────────────────────────────────────────────────────────────

def build_runner(trajectory: str, indices: List[int],
                 rng_seed: int = 42) -> Tuple[TwoTierOrchestrator, dict]:
    """Set up a tracker in DEGENERACY mode, plus the per-frame data
    handles needed to drive it (camera poses, ee poses, fingers)."""
    data_root = os.path.join(DATA_BASE, trajectory)
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
    slam = PassThroughSlam(slam_poses, default_cov=slam_cov)

    orch = TwoTierOrchestrator(
        slam,
        trigger=TriggerConfig(periodic_every_n_frames=30),
        verbose=False,
        rng_seed=rng_seed,
        bernoulli=BernoulliConfig.degeneracy(),
    )

    return orch, {
        "cam_poses": cam_poses,
        "ee_poses": ee_poses,
        "l_finger": l_finger,
        "r_finger": r_finger,
        "data_root": data_root,
    }


def _build_detections_for_step(raw_dets: List[Dict[str, Any]],
                                 depth: np.ndarray) -> List[Dict[str, Any]]:
    """Convert SAM2 JSON -> orchestrator detection dicts."""
    out: List[Dict[str, Any]] = []
    for d in raw_dets:
        oid = d.get("object_id")
        if oid is None:
            continue
        T_co = _build_T_co_from_mask(d["mask"], depth, K)
        if T_co is None:
            continue
        out.append({
            "id": int(oid),
            "label": d.get("label", "unknown"),
            "mask": d["mask"],
            "score": float(d.get("score", 0.0)),
            "T_co": T_co,
            "R_icp": np.diag([1e-4] * 3 + [1e-3] * 3),
            "fitness": float(max(0.3, d.get("score", 0.5))),
            "rmse": 0.005,
            "box": d.get("box"),
        })
    return out


def _gripper_phase(finger_l: np.ndarray, finger_r: np.ndarray,
                    last_d: Optional[float],
                    last_phase: str) -> Tuple[str, float]:
    finger_d = float(np.linalg.norm(finger_l[:3, 3] - finger_r[:3, 3]))
    raw = _gripper_state_from_distance(finger_d, last_d)
    if raw == "grasping":
        phase = "grasping"
    elif raw == "releasing":
        phase = "releasing"
    elif last_phase == "grasping":
        phase = "holding"
    elif last_phase in ("holding", "releasing") and raw == "idle":
        phase = "idle" if last_phase == "releasing" else "holding"
    elif last_phase == "holding" and raw == "idle":
        phase = "holding"
    else:
        phase = "idle"
    return phase, finger_d


# ─────────────────────────────────────────────────────────────────────
# Top-level driver.
# ─────────────────────────────────────────────────────────────────────

def run(trajectory: str = "apple_bowl_2",
        n_frames: Optional[int] = None,
        step: int = 3,
        rng_seed: int = 42,
        make_video: bool = True) -> None:
    data_root = os.path.join(DATA_BASE, trajectory)
    rgb_dir = os.path.join(data_root, "rgb")
    det_dir = os.path.join(data_root, "detection_h")
    if not (os.path.isdir(rgb_dir) and os.path.isdir(det_dir)):
        print(f"[viz] missing rgb/ or detection_h/ in {data_root}",
              file=sys.stderr)
        return

    out_dir = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline",
                            trajectory, "degenerate_vs_sam2")
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

    np.random.seed(rng_seed)
    orch, handles = build_runner(trajectory, indices, rng_seed=rng_seed)

    print(f"[viz] {len(indices)} frames, total raw: {total}, step={step}")
    print(f"[viz] tracker: degenerate Bernoulli (reproduces legacy)")

    last_finger_d: Optional[float] = None
    last_phase = "idle"
    held_obj: Optional[int] = None

    for local_i, idx in enumerate(indices):
        rgb_path = os.path.join(rgb_dir, f"rgb_{idx:06d}.png")
        depth_path = os.path.join(data_root, "depth", f"depth_{idx:06d}.npy")
        det_path = os.path.join(det_dir, f"detection_{idx:06d}_final.json")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)
        raw_dets = _load_detections(det_path)
        dets = _build_detections_for_step(raw_dets, depth)

        phase, finger_d = _gripper_phase(
            handles["l_finger"][idx], handles["r_finger"][idx],
            last_finger_d, last_phase,
        )
        last_finger_d = finger_d
        last_phase = phase

        # Light held-object resolution so the tracker's rigid-attachment
        # predict gets the right object when holding.
        if phase == "grasping" and held_obj is None:
            T_cw = handles["cam_poses"][idx]
            T_ec = handles["ee_poses"][idx]
            T_ew = T_cw @ T_ec
            if dets:
                best = min(
                    ((d["id"], np.linalg.norm(
                        (T_cw @ d["T_co"])[:3, 3] - T_ew[:3, 3]))
                     for d in dets),
                    key=lambda kv: kv[1])
                held_obj = best[0]
        elif phase == "idle":
            held_obj = None

        T_ec = handles["ee_poses"][idx]
        orch.step(rgb, depth, dets,
                   {"phase": phase, "held_obj_id": held_obj}, T_ec=T_ec)

        objects = orch.objects
        T_cw = handles["cam_poses"][idx]
        out_path = os.path.join(out_dir, f"frame_{idx:06d}.png")
        render_frame(rgb, raw_dets, objects, orch, T_cw, idx, phase, out_path)

        if local_i % 10 == 0:
            print(f"  [{local_i+1}/{len(indices)}] frame {idx}: "
                  f"{len(dets)} dets  {len(objects)} tracks  "
                  f"phase={phase}")

    if make_video:
        _pngs_to_mp4(out_dir, os.path.join(out_dir, "summary.mp4"))

    print(f"[viz] done. Output: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", default="apple_bowl_2")
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    run(trajectory=args.trajectory,
        n_frames=args.frames, step=args.step,
        rng_seed=args.seed, make_video=not args.no_video)
