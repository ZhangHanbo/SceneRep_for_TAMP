#!/usr/bin/env python3
"""
Three-panel side-by-side visualization: SAM2 observations, the Bernoulli-EKF
in DEGENERACY mode (reproduces legacy _fast_tier bit-exactly), and the
Bernoulli-EKF in FULL mode (Hungarian association + chi^2 gating + Huber
re-weighting + Phi covariance cap + visibility predicate + r tracking).

Each output PNG has three panels on identical RGB backgrounds:

  * LEFT   — SAM2 detections: masks + bboxes + `id:N label s=0.NN`.
  * MIDDLE — degenerate tracker: each track's world-frame pose projected
             back into the image via K + T_cw, with label `id:N label fso=F`.
  * RIGHT  — full Bernoulli tracker: same layout but each confirmed
             (r >= r_conf) track also shows its existence probability,
             and tentative tracks (r < r_conf) are dashed.

Shared per-frame inputs (RGB, depth, detections, gripper state, SLAM pose)
are computed once and fed to BOTH trackers, so any visual difference is
attributable only to the fast-tier choice.

Output: tests/visualization_pipeline/<trajectory>/three_way/
    frame_NNNNNN.png
    summary.mp4

Run:
    conda run -n ocmp_test python tests/visualize_three_way.py \
        [--trajectory apple_bowl_2] [--frames N] [--step S] [--no-video]
        [--alpha 4.4] [--enforce-labels/--no-enforce-labels]
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
    _build_T_co_from_mask, _gripper_state_from_distance, _load_pose_txt, K,
)
from tests.visualize_sam2_observations import (
    _load_detections, overlay_detections, _palette_color, _pngs_to_mp4,
)
from tests.visualize_pipeline import resolve_held_by_proximity
from tests.visualize_degenerate_vs_sam2 import (
    project_world_point, project_cov_xy,
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
# Panel renderers.
# ─────────────────────────────────────────────────────────────────────

def _draw_sam2_panel(ax, rgb: np.ndarray,
                      raw_dets: List[Dict[str, Any]],
                      frame_idx: int) -> None:
    overlay = overlay_detections(rgb, raw_dets)
    ax.imshow(overlay)
    ids_here = sorted(int(d["object_id"]) for d in raw_dets
                       if d.get("object_id") is not None)
    ax.set_title(
        f"SAM2 observations — frame {frame_idx:04d}   "
        f"{len(raw_dets)} masks   ids: {ids_here}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_tracker_panel(ax, rgb: np.ndarray,
                         objects: Dict[int, Dict[str, Any]],
                         orch: TwoTierOrchestrator,
                         T_cw: np.ndarray,
                         frame_idx: int,
                         phase: str,
                         title_prefix: str,
                         show_r: bool,
                         tentative: Optional[Dict[int, Dict[str, Any]]] = None,
                         ) -> None:
    ax.imshow(rgb)
    H, W = rgb.shape[:2]
    n_conf = len(objects)
    n_tent = len(tentative) if tentative else 0
    extra = f" + {n_tent} tentative" if n_tent else ""
    ax.set_title(
        f"{title_prefix} — frame {frame_idx:04d} [{phase}]   "
        f"{n_conf}{' confirmed' if show_r else ''}{extra}",
        fontsize=10,
    )
    ax.set_xticks([])
    ax.set_yticks([])

    legend_handles: List[mpatches.Patch] = []

    def _draw_one(oid: int, info: Dict[str, Any], dashed: bool) -> None:
        T_w = np.asarray(info["T"], dtype=np.float64)
        cov = np.asarray(info["cov"], dtype=np.float64)
        label = info.get("label", "?")
        r_val = info.get("r")
        if show_r and r_val is None:
            r_val = orch.existence.get(oid) if hasattr(orch, "existence") else None
        fso = orch.frames_since_obs.get(oid, -1) if hasattr(orch, "frames_since_obs") else -1

        proj = project_world_point(T_w[:3, 3], T_cw, K)
        if proj is None:
            return
        u, v, _ = proj
        if not (0 <= u < W and 0 <= v < H):
            return

        color = tuple(c / 255.0 for c in _palette_color(oid))
        a, b = project_cov_xy(T_w, cov, T_cw, K)
        linestyle = "--" if dashed else "-"
        alpha_fill = 0.45 if dashed else 0.9
        ax.add_patch(Ellipse((u, v), width=a, height=b,
                              edgecolor=color, facecolor="none",
                              linewidth=1.4 if dashed else 1.6,
                              linestyle=linestyle, alpha=alpha_fill))
        marker = "x" if dashed else "o"
        ax.plot(u, v, marker, color=color, markersize=8 if dashed else 7,
                markeredgecolor=None if dashed else "black",
                markeredgewidth=1.5 if dashed else 0.6,
                alpha=0.7 if dashed else 1.0)

        text = f"id:{oid}  {label}"
        if show_r and r_val is not None:
            text += f"  r={r_val:.2f}"
        text += f"  fso={fso}"
        if dashed:
            text += "  (tent)"
        ax.text(u + 10, v - 10, text, fontsize=6, color=color,
                alpha=0.85,
                bbox=dict(facecolor="white",
                          alpha=0.75 if not dashed else 0.55,
                          edgecolor="none", pad=1.0))

        legend_handles.append(mpatches.Patch(
            color=color, label=f"id:{oid}  {label}"))

    for oid in sorted(objects.keys()):
        _draw_one(oid, objects[oid], dashed=False)
    if tentative:
        for oid in sorted(tentative.keys()):
            _draw_one(oid, tentative[oid], dashed=True)

    if legend_handles:
        # Deduplicate legend entries while preserving order.
        seen = set()
        unique = []
        for h in legend_handles:
            if h.get_label() not in seen:
                unique.append(h)
                seen.add(h.get_label())
        ax.legend(handles=unique, fontsize=6,
                  loc="lower right", framealpha=0.85, ncol=1)


def render_frame(rgb: np.ndarray,
                  raw_dets: List[Dict[str, Any]],
                  degen_objects: Dict[int, Dict[str, Any]],
                  full_objects: Dict[int, Dict[str, Any]],
                  full_tentative: Dict[int, Dict[str, Any]],
                  degen_orch: TwoTierOrchestrator,
                  full_orch: TwoTierOrchestrator,
                  T_cw: np.ndarray,
                  frame_idx: int,
                  phase: str,
                  out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), dpi=110)
    _draw_sam2_panel(axes[0], rgb, raw_dets, frame_idx)
    _draw_tracker_panel(axes[1], rgb, degen_objects, degen_orch, T_cw,
                         frame_idx, phase,
                         title_prefix="Degenerate Bernoulli (= legacy)",
                         show_r=False)
    _draw_tracker_panel(axes[2], rgb, full_objects, full_orch, T_cw,
                         frame_idx, phase,
                         title_prefix="Full Bernoulli-EKF",
                         show_r=True,
                         tentative=full_tentative)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Runner.
# ─────────────────────────────────────────────────────────────────────

def _build_detections(raw: List[Dict[str, Any]],
                       depth: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    for d in raw:
        oid = d.get("object_id")
        if oid is None:
            continue
        T_co = _build_T_co_from_mask(d["mask"], depth, K)
        if T_co is None:
            continue
        out.append({
            "id": int(oid),
            "label": d.get("label", "?"),
            "mask": d["mask"],
            "score": float(d.get("score", 0.0)),
            "T_co": T_co,
            "R_icp": np.diag([1e-4] * 3 + [1e-3] * 3),
            "fitness": float(max(0.3, d.get("score", 0.5))),
            "rmse": 0.005,
            "box": d.get("box"),
        })
    return out


def _gripper_phase(l: np.ndarray, r: np.ndarray,
                    last_d: Optional[float],
                    last_phase: str) -> Tuple[str, float]:
    d = float(np.linalg.norm(l[:3, 3] - r[:3, 3]))
    raw = _gripper_state_from_distance(d, last_d)
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
    return phase, d


def _make_full_config(alpha: float, enforce_labels: bool,
                      image_shape: Tuple[int, int]) -> BernoulliConfig:
    return BernoulliConfig(
        association_mode="hungarian",
        p_s=1.0, p_d=0.9, alpha=alpha,
        lambda_c=1.0, lambda_b=1.0,
        r_conf=0.5, r_min=1e-3,
        G_in=12.59, G_out=25.0,
        P_max=np.diag([0.0625] * 3 + [(np.pi / 4) ** 2] * 3),
        enable_visibility=True, enable_huber=True,
        init_cov_from_R=False,
        enforce_label_match=enforce_labels,
        K=K,
        image_shape=image_shape,
    )


def run(trajectory: str = "apple_bowl_2",
        n_frames: Optional[int] = None,
        step: int = 3,
        rng_seed: int = 42,
        alpha: float = 4.4,
        enforce_labels: bool = True,
        make_video: bool = True) -> None:
    data_root = os.path.join(DATA_BASE, trajectory)
    rgb_dir = os.path.join(data_root, "rgb")
    det_dir = os.path.join(data_root, "detection_h")
    if not (os.path.isdir(rgb_dir) and os.path.isdir(det_dir)):
        print(f"[viz] missing rgb/ or detection_h/ in {data_root}",
              file=sys.stderr)
        return

    out_dir = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline",
                            trajectory, "three_way")
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

    # Two independent orchestrators, seeded identically. The SLAM backend
    # is stateful (index counter), so we need two instances.
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
          f"enforce_label_match={enforce_labels}")

    last_finger_d_d: Optional[float] = None
    last_finger_d_f: Optional[float] = None
    last_phase_d = "idle"
    last_phase_f = "idle"
    held_d: Optional[int] = None
    held_f: Optional[int] = None

    for local_i, idx in enumerate(indices):
        rgb_path = os.path.join(rgb_dir, f"rgb_{idx:06d}.png")
        depth_path = os.path.join(data_root, "depth", f"depth_{idx:06d}.npy")
        det_path = os.path.join(det_dir, f"detection_{idx:06d}_final.json")
        if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
            continue

        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)
        raw_dets = _load_detections(det_path)
        dets = _build_detections(raw_dets, depth)

        # Shared: gripper phase and held-object via cluster-distance.
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

        # Step BOTH orchestrators on the same inputs.
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
        render_frame(rgb, raw_dets, degen_objs, full_objs, full_tent,
                      degen_orch, full_orch, T_cw, idx, phase_d, out_path)

        if local_i % 10 == 0:
            print(f"  [{local_i+1}/{len(indices)}] frame {idx}: "
                  f"{len(raw_dets)} dets  "
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
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    run(trajectory=args.trajectory,
        n_frames=args.frames, step=args.step,
        rng_seed=args.seed, alpha=args.alpha,
        enforce_labels=args.enforce_labels,
        make_video=not args.no_video)
