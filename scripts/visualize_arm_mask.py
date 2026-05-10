#!/usr/bin/env python3
"""Project the Fetch arm onto each RGB frame and overlay a tinted mask.

Driver around :class:`utils.fetch_arm_mask.ArmMaskBuilder`. For each frame:
load joint angles + per-frame ``T_bc``, build the arm mask, tint the
masked region of the RGB, and save the overlay PNG.

Use this to visually verify the URDF + FK + extrinsic pipeline produce
a correct arm silhouette before applying the same mask to depth in the
EKF voxel-integration loop (``visualize_voxel_obs.py --mask-arm``).

Run::

    python scripts/visualize_arm_mask.py --trajectory apple_drop
    python scripts/visualize_arm_mask.py --trajectory apple_drop --dilate-px 12
    python scripts/visualize_arm_mask.py --trajectory apple_drop --start 200 --max-frame 201
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from utils.fetch_arm_mask import (  # noqa: E402
    ArmMaskBuilder, DEFAULT_URDF, DEFAULT_MESH_ROOT,
)

DATASET_DIR = os.path.join(SCENEREP_ROOT, "datasets")
VIZ_BASE = os.path.join(SCENEREP_ROOT, "tests", "visualization_pipeline")

K_DEFAULT = np.array([
    [554.3827, 0.0, 320.5],
    [0.0, 554.3827, 240.5],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def _load_T_bc_poses(path: str) -> Optional[Dict[int, np.ndarray]]:
    if not os.path.exists(path):
        return None
    out: Dict[int, np.ndarray] = {}
    with open(path) as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            try:
                idx = int(arr[0])
                tx, ty, tz, qx, qy, qz, qw = map(float, arr[1:])
            except ValueError:
                continue
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            out[idx] = T
    return out or None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trajectory", default="apple_drop")
    ap.add_argument("--max-frame", type=int, default=10**6)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--urdf-path", default=DEFAULT_URDF)
    ap.add_argument("--mesh-root", default=DEFAULT_MESH_ROOT)
    ap.add_argument("--out-subdir", default="arm_mask_overlay")
    ap.add_argument("--no-mp4", action="store_true",
                    help="skip composing per-frame PNGs into an MP4")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.45,
                    help="overlay opacity (0..1)")
    ap.add_argument("--color", default="0,220,80",
                    help="overlay color B,G,R (default 0,220,80 = green)")
    ap.add_argument("--dilate-px", type=int, default=8,
                    help="morphological dilation kernel size in pixels (0 to "
                         "disable). Default 8 covers boundary slack on the "
                         "convex-hull-per-link silhouette.")
    args = ap.parse_args()

    color_parts = args.color.split(",")
    if len(color_parts) != 3:
        print("[err] --color must be three comma-separated ints (B,G,R)",
              file=sys.stderr)
        return 2
    try:
        color = tuple(int(c) for c in color_parts)
    except ValueError:
        print("[err] could not parse --color", file=sys.stderr)
        return 2

    # ── trajectory data first (so we know image size) ──
    traj = args.trajectory
    ds_root = os.path.join(DATASET_DIR, traj)
    rgb_dir = os.path.join(ds_root, "rgb")
    if not os.path.isdir(rgb_dir):
        print(f"[err] no rgb/ at {rgb_dir}", file=sys.stderr)
        return 4
    T_bc_map = _load_T_bc_poses(os.path.join(ds_root, "pose_txt", "T_bc.txt"))
    if T_bc_map is None:
        print("[err] no T_bc.txt; cannot project arm without per-frame extrinsic",
              file=sys.stderr)
        return 4
    joints_path = os.path.join(ds_root, "pose_txt", "joints_pose.json")
    if not os.path.exists(joints_path):
        print(f"[err] no joints_pose.json at {joints_path}", file=sys.stderr)
        return 4
    with open(joints_path) as f:
        raw = json.load(f)
    joints_map: Dict[int, Dict[str, float]] = {int(k): v for k, v in raw.items()}
    print(f"[traj] {traj}: {len(joints_map)} frames of joint state")

    # Probe one image for size.
    first_rgb_path = None
    for fr in sorted(joints_map.keys()):
        cand = os.path.join(rgb_dir, f"rgb_{fr:06d}.png")
        if os.path.exists(cand):
            first_rgb_path = cand
            break
    if first_rgb_path is None:
        print(f"[err] no rgb_*.png under {rgb_dir}", file=sys.stderr)
        return 4
    sample_rgb = cv2.imread(first_rgb_path)
    H, W = sample_rgb.shape[:2]
    print(f"[traj] image size: {W}x{H}")

    # ── builder ──
    print(f"[urdf] loading from {args.urdf_path}")
    builder = ArmMaskBuilder(
        urdf_path=args.urdf_path,
        mesh_root=args.mesh_root,
        K=K_DEFAULT,
        image_shape=(H, W),
        dilate_px=args.dilate_px,
    )
    print(f"[urdf] {len(builder.fk.joints)} joints, {len(builder.fk.links)} links; "
          f"{len(builder.meshes)} link meshes loaded; dilate_px={args.dilate_px}")

    out_dir = os.path.join(VIZ_BASE, traj, args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    end = min(args.max_frame, max(joints_map.keys()) + 1)
    written = 0
    skipped = 0
    for fr in range(args.start, end, args.step):
        if fr not in joints_map or fr not in T_bc_map:
            skipped += 1
            continue
        rgb_path = os.path.join(rgb_dir, f"rgb_{fr:06d}.png")
        if not os.path.exists(rgb_path):
            skipped += 1
            continue
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            skipped += 1
            continue
        try:
            mask = builder.build(joints_map[fr], T_bc_map[fr])
        except ValueError as e:
            print(f"[warn] FK failed at fr {fr}: {e}")
            continue

        overlay = rgb.copy()
        overlay[mask > 0] = color
        out = cv2.addWeighted(overlay, float(args.alpha),
                                rgb, 1.0 - float(args.alpha), 0.0)
        out_path = os.path.join(out_dir, f"frame_{fr:06d}.png")
        cv2.imwrite(out_path, out)
        written += 1
        if written % 50 == 0:
            print(f"  wrote {written} frames")

    print(f"[done] wrote {written} frames under {out_dir} (skipped {skipped})")

    if not args.no_mp4 and written > 0:
        mp4_path = os.path.join(VIZ_BASE, traj, "arm_mask_overlay.mp4")
        first_fr = next(
            (f for f in range(args.start, end, args.step)
              if os.path.exists(os.path.join(out_dir, f"frame_{f:06d}.png"))),
            None,
        )
        if first_fr is None:
            return 0
        sample = cv2.imread(os.path.join(out_dir, f"frame_{first_fr:06d}.png"))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(mp4_path, fourcc, float(args.fps),
                              (sample.shape[1], sample.shape[0]))
        n_appended = 0
        for fr in range(args.start, end, args.step):
            p = os.path.join(out_dir, f"frame_{fr:06d}.png")
            if not os.path.exists(p):
                continue
            img = cv2.imread(p)
            if img is None:
                continue
            vw.write(img)
            n_appended += 1
        vw.release()
        print(f"[mp4] {mp4_path}  ({n_appended} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
