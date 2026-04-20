#!/usr/bin/env python3
"""
Side-by-side visualization: legacy `_fast_tier` vs Bernoulli-EKF
`_fast_tier_bernoulli` on apple_bowl_2.

Runs both orchestrators in lockstep over the same trajectory. For each
processed frame, projects every tracked object's world-frame pose back
into the image via the known camera intrinsics and pose, and overlays:

  * Legacy (left panel):
      - filled circle at projected centroid, colored by stable palette
      - text: "oid:N label"
      - faint ellipse of the (x,y)-projected 2-sigma covariance

  * Bernoulli (right panel):
      - filled circle at projected centroid (solid if r >= r_conf,
        dashed / semi-transparent if tentative)
      - text: "oid:N label  r=0.NN"
      - ellipse tinted by r (bright for r ~ 1, pale for r ~ 0)

Output: tests/visualization_pipeline/apple_bowl_2/bernoulli/frame_NNNNNN.png
        tests/visualization_pipeline/apple_bowl_2/bernoulli/summary.mp4
        tests/visualization_pipeline/apple_bowl_2/bernoulli/r_traces.png

Run:
    conda run -n ocmp_test python tests/visualize_bernoulli.py \
        [--frames N] [--step S] [--no-video]
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
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
    TrajectoryRunner, DATA_ROOT, HAS_DATA, K,
)
from tests.test_bernoulli_degeneracy import _BernoulliRunner
from pose_update.orchestrator import (
    TwoTierOrchestrator, TriggerConfig, BernoulliConfig,
)

OUT_DIR = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline",
                        "apple_bowl_2", "bernoulli")


# ─────────────────────────────────────────────────────────────────────
# Colors: stable palette keyed by oid modulo len.
# ─────────────────────────────────────────────────────────────────────

_PALETTE = [
    (0.18, 0.80, 0.44),
    (0.90, 0.30, 0.23),
    (0.20, 0.60, 0.86),
    (0.95, 0.77, 0.06),
    (0.61, 0.35, 0.71),
    (0.90, 0.49, 0.13),
    (0.10, 0.74, 0.61),
    (0.93, 0.44, 0.39),
    (0.36, 0.68, 0.89),
    (0.96, 0.82, 0.25),
]


def _color(oid: int) -> Tuple[float, float, float]:
    return _PALETTE[int(oid) % len(_PALETTE)]


# ─────────────────────────────────────────────────────────────────────
# Projection helpers.
# ─────────────────────────────────────────────────────────────────────

def project_world_point(p_w: np.ndarray, T_cw: np.ndarray,
                         K_mat: np.ndarray) -> Optional[Tuple[float, float, float]]:
    """World-frame 3D point -> (u, v, depth_cam). Returns None if behind
    camera."""
    T_wc = np.linalg.inv(T_cw)
    p_c = T_wc @ np.append(p_w, 1.0)
    depth = float(p_c[2])
    if depth <= 0:
        return None
    uv = K_mat @ p_c[:3]
    return (float(uv[0] / uv[2]), float(uv[1] / uv[2]), depth)


def project_cov_xy(T_w: np.ndarray, cov: np.ndarray,
                    T_cw: np.ndarray, K_mat: np.ndarray,
                    sigma_px_fallback: float = 18.0
                    ) -> Tuple[float, float]:
    """Rough (a, b) ellipse axes in pixels for the translation block of
    cov, projected to the image plane at the object's depth. Uses a
    scalar-depth approximation for simplicity (the translation cov is
    often diag in manipulation scenes so this matches well).
    """
    # Pull the translation 3x3 sub-block.
    cov_t = cov[:3, :3]
    eigs = np.linalg.eigvalsh(0.5 * (cov_t + cov_t.T))
    if np.any(eigs < 0):
        eigs = np.clip(eigs, 0.0, None)
    sigma_m = float(np.sqrt(max(eigs.max(), 0.0)))
    sigma_m = max(sigma_m, 1e-3)

    T_wc = np.linalg.inv(T_cw)
    p_c = T_wc @ np.append(T_w[:3, 3], 1.0)
    depth = float(p_c[2])
    if depth <= 0:
        return (sigma_px_fallback, sigma_px_fallback)

    fx = float(K_mat[0, 0])
    sigma_px = 2.0 * sigma_m * fx / depth
    sigma_px = float(np.clip(sigma_px, 4.0, 120.0))
    return (sigma_px, sigma_px)


# ─────────────────────────────────────────────────────────────────────
# Rendering.
# ─────────────────────────────────────────────────────────────────────

def draw_panel(ax, rgb: np.ndarray,
               objects: Dict[int, Dict[str, Any]],
               tentative: Dict[int, Dict[str, Any]],
               T_cw: np.ndarray,
               title: str,
               show_r: bool) -> None:
    ax.imshow(rgb)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    H, W = rgb.shape[:2]

    # Confirmed tracks.
    for oid, info in sorted(objects.items()):
        T_w = np.asarray(info["T"], dtype=np.float64)
        cov = np.asarray(info["cov"], dtype=np.float64)
        label = info.get("label", "?")
        r_val = info.get("r")

        proj = project_world_point(T_w[:3, 3], T_cw, K)
        if proj is None:
            continue
        u, v, _ = proj
        if not (0 <= u < W and 0 <= v < H):
            continue

        color = _color(oid)
        a, b = project_cov_xy(T_w, cov, T_cw, K)
        ax.add_patch(Ellipse((u, v), width=a, height=b,
                              edgecolor=color, facecolor="none",
                              linewidth=1.5, alpha=0.8))
        ax.plot(u, v, "o", color=color, markersize=7,
                markeredgecolor="black", markeredgewidth=0.6)

        text = f"{oid}:{label}"
        if show_r and r_val is not None:
            text += f"  r={r_val:.2f}"
        ax.text(u + 10, v - 10, text, fontsize=7,
                color=color,
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none", pad=1.0))

    # Tentative tracks (dashed, semi-transparent).
    for oid, info in sorted(tentative.items()):
        T_w = np.asarray(info["T"], dtype=np.float64)
        cov = np.asarray(info["cov"], dtype=np.float64)
        label = info.get("label", "?")
        r_val = info.get("r", 0.0)

        proj = project_world_point(T_w[:3, 3], T_cw, K)
        if proj is None:
            continue
        u, v, _ = proj
        if not (0 <= u < W and 0 <= v < H):
            continue

        color = _color(oid)
        a, b = project_cov_xy(T_w, cov, T_cw, K)
        ax.add_patch(Ellipse((u, v), width=a, height=b,
                              edgecolor=color, facecolor="none",
                              linewidth=1.0, alpha=0.4, linestyle="--"))
        ax.plot(u, v, "x", color=color, markersize=8,
                markeredgewidth=1.5, alpha=0.6)
        text = f"{oid}:{label} r={r_val:.2f} (tent)"
        ax.text(u + 10, v - 10, text, fontsize=6,
                color=color, alpha=0.8,
                bbox=dict(facecolor="white", alpha=0.5,
                          edgecolor="none", pad=1.0))


def render_frame(rgb: np.ndarray,
                  legacy_objects: Dict[int, Dict[str, Any]],
                  bern_objects: Dict[int, Dict[str, Any]],
                  bern_tentative: Dict[int, Dict[str, Any]],
                  T_cw: np.ndarray,
                  frame_idx: int,
                  phase: str,
                  out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=120)

    draw_panel(axes[0], rgb, legacy_objects, {}, T_cw,
               f"Legacy fast tier — frame {frame_idx:04d} [{phase}] "
               f"{len(legacy_objects)} tracks",
               show_r=False)
    draw_panel(axes[1], rgb, bern_objects, bern_tentative, T_cw,
               f"Bernoulli-EKF — frame {frame_idx:04d} [{phase}] "
               f"{len(bern_objects)} confirmed + {len(bern_tentative)} tentative",
               show_r=True)

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Driver.
# ─────────────────────────────────────────────────────────────────────

def run(n_frames: int = 200, step: int = 3, rng_seed: int = 42,
        make_video: bool = True) -> None:
    if not HAS_DATA:
        print(f"[viz] data not found at {DATA_ROOT}", file=sys.stderr)
        return

    if os.path.isdir(OUT_DIR):
        # Clear stale frames so the video stays coherent.
        for f in glob.glob(os.path.join(OUT_DIR, "frame_*.png")):
            os.remove(f)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Build both runners with matching seeds so any difference is purely
    # a consequence of the fast tier, not the particle sampler.
    np.random.seed(rng_seed)
    legacy = TrajectoryRunner(n_frames=n_frames, step=step)
    legacy.orchestrator = TwoTierOrchestrator(
        legacy.slam_backend,
        trigger=TriggerConfig(periodic_every_n_frames=30),
        verbose=False, rng_seed=rng_seed,
    )

    bern_cfg = BernoulliConfig(
        association_mode="hungarian",
        p_s=1.0, p_d=0.9, q_s=0.9,
        lambda_c=1.0, lambda_b=1.0,
        r_conf=0.5, r_min=1e-3,
        G_in=12.59, G_out=25.0,
        P_max=np.diag([0.0625] * 3 + [(np.pi / 4) ** 2] * 3),
        enable_visibility=True, enable_huber=True,
        init_cov_from_R=False, enforce_label_match=True,
        K=K,
        image_shape=(480, 640),
    )
    np.random.seed(rng_seed)
    bern = _BernoulliRunner(n_frames=n_frames, step=step,
                             bernoulli_config=bern_cfg, rng_seed=rng_seed)

    print(f"[viz] writing frames to {OUT_DIR}")
    print(f"[viz] {len(legacy.indices)} frames scheduled (n={n_frames}, step={step})")

    # Run both orchestrators in lockstep over the shared indices,
    # collecting r histories so we can emit a trailing r_traces plot.
    r_history: Dict[int, List[Tuple[int, float]]] = {}

    for local_i, idx in enumerate(legacy.indices):
        # Drive legacy one step.
        rgb, depth, dets, grip, T_ec_leg = _collect_inputs(legacy, idx, local_i)
        if rgb is None:
            continue
        legacy.orchestrator.step(rgb, depth, dets, grip, T_ec=T_ec_leg)

        # Drive Bernoulli with the SAME inputs.
        rgb_b, depth_b, dets_b, grip_b, T_ec_b = _collect_inputs(bern, idx, local_i)
        # If any of the bern-side helpers mutate state (they do: gripper
        # state machine), we end up with two separate copies -- that's why
        # we don't share. rgb/depth are independent of the runner.
        bern.orchestrator.step(rgb_b, depth_b, dets_b, grip_b, T_ec=T_ec_b)

        # Snapshot outputs.
        legacy_objs = legacy.orchestrator.objects
        bern_objs = bern.orchestrator.objects
        bern_tent = bern.orchestrator.tentative_objects

        # Update r history.
        for oid, r_val in bern.orchestrator.existence.items():
            r_history.setdefault(int(oid), []).append((idx, float(r_val)))

        T_cw = legacy.cam_poses[idx]
        out_path = os.path.join(OUT_DIR, f"frame_{idx:06d}.png")
        render_frame(rgb, legacy_objs, bern_objs, bern_tent,
                      T_cw, idx, grip.get("phase", "?"), out_path)

        if local_i % 10 == 0:
            print(f"  [{local_i+1}/{len(legacy.indices)}] "
                  f"frame {idx}  legacy={len(legacy_objs)}  "
                  f"bern_conf={len(bern_objs)}  bern_tent={len(bern_tent)}")

    # r-vs-frame plot.
    fig, ax = plt.subplots(figsize=(10, 5))
    for oid, hist in sorted(r_history.items()):
        if len(hist) < 2:
            continue
        xs = [h[0] for h in hist]
        ys = [h[1] for h in hist]
        label_str = bern.orchestrator.object_labels.get(oid, "?")
        ax.plot(xs, ys, "-", color=_color(oid),
                label=f"oid {oid} ({label_str})", alpha=0.85)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.7,
               label="r_conf = 0.5")
    ax.axhline(1e-3, color="red", linestyle=":", linewidth=0.7,
               label="r_min = 1e-3")
    ax.set_xlabel("frame index")
    ax.set_ylabel("existence probability r")
    ax.set_title("Bernoulli-EKF: r(k) per track on apple_bowl_2")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6, ncol=2, loc="center right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "r_traces.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[viz] r_traces.png saved")

    if make_video:
        mp4_path = os.path.join(OUT_DIR, "summary.mp4")
        _pngs_to_mp4(OUT_DIR, mp4_path)

    print(f"[viz] done. Output directory: {OUT_DIR}")


def _collect_inputs(runner: TrajectoryRunner, idx: int, local_i: int):
    """Pull (rgb, depth, detections, gripper_state, T_ec) for frame idx.

    Replicates what TrajectoryRunner.run() does internally, but here we
    expose the inputs so two runners can step in lockstep on the same
    data. Note: `runner._compute_gripper_state` mutates internal state,
    so each runner must call its own copy (that's why this helper does
    the mutation explicitly per call).
    """
    rgb_path = os.path.join(DATA_ROOT, "rgb", f"rgb_{idx:06d}.png")
    depth_path = os.path.join(DATA_ROOT, "depth", f"depth_{idx:06d}.npy")
    det_path = os.path.join(DATA_ROOT, "detection_h",
                             f"detection_{idx:06d}_final.json")
    if not (os.path.exists(rgb_path) and os.path.exists(depth_path)):
        return (None, None, None, None, None)

    from tests.test_orchestrator_integration import (
        _load_detections, _build_T_co_from_mask,
    )
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path).astype(np.float32)
    raw = _load_detections(det_path)
    dets: List[Dict] = []
    for d in raw:
        if d["id"] is None:
            continue
        T_co = _build_T_co_from_mask(d["mask"], depth, K)
        if T_co is None:
            continue
        dets.append({
            "id": int(d["id"]),
            "label": d["label"],
            "mask": d["mask"],
            "score": d["score"],
            "T_co": T_co,
            "R_icp": np.diag([1e-4] * 3 + [1e-3] * 3),
            "fitness": float(max(0.3, d["score"])),
            "rmse": 0.005,
            "box": d.get("box"),
        })

    grip = runner._compute_gripper_state(idx)
    # Replicate the "resolve held object" logic from TrajectoryRunner so
    # the orchestrator gets the held_obj_id. We only need this side-effect
    # on grasp events; on non-grasping frames the helper leaves
    # self._holding_obj alone.
    if (grip["phase"] == "grasping"
            and grip["held_obj_id"] is None):
        T_cw = runner.cam_poses[idx]
        T_ec = runner.ee_poses[idx]
        T_ew = T_cw @ T_ec
        nearby = []
        for d in dets:
            T_wo = T_cw @ d["T_co"]
            dist = np.linalg.norm(T_wo[:3, 3] - T_ew[:3, 3])
            if dist < 0.15:
                nearby.append((d["id"], dist))
        if nearby:
            best_id = min(nearby, key=lambda kv: kv[1])[0]
        elif dets:
            best_id = min(
                [(d["id"], np.linalg.norm(
                    (T_cw @ d["T_co"])[:3, 3] - T_ew[:3, 3]))
                 for d in dets],
                key=lambda kv: kv[1])[0]
        else:
            best_id = None
        runner._holding_obj = best_id
        grip["held_obj_id"] = best_id

    T_ec = runner.ee_poses[idx]
    return rgb, depth, dets, grip, T_ec


def _pngs_to_mp4(png_dir: str, mp4_path: str, fps: int = 5) -> None:
    frames = sorted(glob.glob(os.path.join(png_dir, "frame_*.png")))
    if not frames:
        print("[viz] no frames to stitch into mp4")
        return
    # Read the first frame to learn size.
    im0 = cv2.imread(frames[0])
    h, w = im0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))
    for f in frames:
        im = cv2.imread(f)
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        writer.write(im)
    writer.release()
    print(f"[viz] summary.mp4 saved ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=200,
                    help="max number of raw frames to cover")
    ap.add_argument("--step", type=int, default=3,
                    help="frame stride")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    run(n_frames=args.frames, step=args.step,
        rng_seed=args.seed, make_video=not args.no_video)
