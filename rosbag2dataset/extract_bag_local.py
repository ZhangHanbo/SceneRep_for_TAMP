#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-platform bag extractor — no ROS install required.

Produces the same directory layout the ROS-native ``rosbag2dataset_5hz.py``
writes, minus the TF-composed world-frame camera/gripper poses (those need
a full TF tree walk which we don't attempt on macOS).

    <out>/
      rgb/rgb_NNNNNN.png                 (CompressedImage decoded to PNG)
      depth/depth_NNNNNN.npy             (Image, float32 meters or raw uint16)
      particles/particles_NNNNNN.npy     ((N, 4): x, y, yaw, weight)
      pose_txt/timestamps.txt            (idx sec nsec)
      pose_txt/ee_pose.txt               (idx x y z qx qy qz qw — base frame)
      pose_txt/amcl_pose.txt             (idx x y z qx qy qz qw — map frame)
      pose_txt/amcl_pose_cov.txt         (idx + 36-float covariance row-major)
      pose_txt/joints_pose.json          (frame_idx → {joint_name: position})

Frame timing: we snap to a target rate (default 5 Hz, matching the ROS script)
by keeping the latest cached value of every topic and emitting one composite
frame each time the sampling clock ticks.

Usage:
    python rosbag2dataset/extract_bag_local.py <bag.bag> <out_dir> [--hz 5]
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

# rosbags (pip install rosbags) — no ROS needed.
from rosbags.highlevel import AnyReader
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_type(t: str) -> str:
    """``sensor_msgs/msg/Image`` → ``sensor_msgs/Image``."""
    return t.replace("/msg/", "/")


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """yaw (z-axis) from a quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _decode_compressed_image(msg) -> np.ndarray:
    """CompressedImage → HxWx3 uint8 RGB."""
    buf = bytes(msg.data)
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _decode_image(msg) -> np.ndarray:
    """sensor_msgs/Image → numpy, preserving the source encoding.

    Handles the encodings the Fetch depth stream actually uses:
      - 32FC1 (meters, float32)
      - 16UC1 (millimeters, uint16)  -> pass through, downstream convert
      - mono16/mono8
    """
    enc = str(msg.encoding)
    h, w = int(msg.height), int(msg.width)
    raw = bytes(msg.data)

    if enc in ("32FC1",):
        arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w)
    elif enc in ("16UC1", "mono16"):
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
    elif enc in ("8UC1", "mono8"):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
    elif enc in ("rgb8", "bgr8"):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[..., ::-1]
    else:
        raise ValueError(f"unsupported image encoding: {enc}")
    return arr.copy()


def _depth_to_meters(arr: np.ndarray) -> np.ndarray:
    """Normalize depth to float32 meters regardless of source encoding."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.dtype == np.uint16:
        # Assume millimetres (Fetch default for 16UC1 depth_registered).
        return (arr.astype(np.float32) / 1000.0)
    if arr.dtype == np.float32:
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Latest-cache, as in the live ROS script
# ---------------------------------------------------------------------------

@dataclass
class Latest:
    """Latest seen message per topic, timestamp-tagged."""
    rgb: Optional[object]        = None
    depth: Optional[object]      = None
    particles: Optional[object]  = None
    ee: Optional[object]         = None
    amcl: Optional[object]       = None
    joints: Optional[object]     = None

    rgb_t: int   = 0
    depth_t: int = 0

    # We only emit a frame when a *new* RGB arrives, so track the last
    # emitted RGB timestamp to avoid duplicates.
    last_emitted_rgb_t: int = -1


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

def extract(bag_path: str, out_dir: str, hz: float = 5.0) -> None:
    bag = Path(bag_path)
    if not bag.is_file():
        raise FileNotFoundError(bag_path)

    # Create output layout.
    rgb_dir       = os.path.join(out_dir, "rgb")
    depth_dir     = os.path.join(out_dir, "depth")
    particles_dir = os.path.join(out_dir, "particles")
    pose_dir      = os.path.join(out_dir, "pose_txt")
    for d in (rgb_dir, depth_dir, particles_dir, pose_dir):
        os.makedirs(d, exist_ok=True)

    # Append-friendly pose files (we overwrite on new run).
    ee_path           = os.path.join(pose_dir, "ee_pose.txt")
    amcl_path         = os.path.join(pose_dir, "amcl_pose.txt")
    amcl_cov_path     = os.path.join(pose_dir, "amcl_pose_cov.txt")
    ts_path           = os.path.join(pose_dir, "timestamps.txt")
    joints_path       = os.path.join(pose_dir, "joints_pose.json")
    for p in (ee_path, amcl_path, amcl_cov_path, ts_path):
        with open(p, "w"):
            pass

    joints_map: Dict[str, Dict[str, float]] = {}

    # Topic → (category) router.
    topic_cat = {
        "/head_camera/rgb/image_raw/compressed":   "rgb",
        "/head_camera/depth_registered/image_raw": "depth",
        "/particlecloud":                          "particles",
        "/end_effector_pose":                      "ee",
        "/amcl_pose":                              "amcl",
        "/joint_states":                           "joints",
    }

    period_ns = int(1e9 / hz)

    latest = Latest()
    save_idx = 0

    def emit_frame(rgb_msg, rgb_t: int) -> None:
        nonlocal save_idx

        # ---- RGB -------------------------------------------------------
        try:
            rgb = _decode_compressed_image(rgb_msg)
            Image.fromarray(rgb).save(
                os.path.join(rgb_dir, f"rgb_{save_idx:06d}.png"))
        except Exception as e:
            print(f"[skip rgb {save_idx}]  {e}")
            return

        # ---- Depth ------------------------------------------------------
        if latest.depth is not None:
            try:
                d = _decode_image(latest.depth)
                d = _depth_to_meters(d)
                np.save(os.path.join(depth_dir, f"depth_{save_idx:06d}.npy"), d)
            except Exception as e:
                print(f"[warn depth {save_idx}]  {e}")

        # ---- Particles --------------------------------------------------
        if latest.particles is not None:
            try:
                p = latest.particles
                poses = list(p.poses)
                if poses:
                    xs, ys, yaws = [], [], []
                    for po in poses:
                        xs.append(po.position.x)
                        ys.append(po.position.y)
                        q = po.orientation
                        yaws.append(_quat_to_yaw(q.x, q.y, q.z, q.w))
                    n = len(poses)
                    arr = np.stack([
                        np.asarray(xs,  dtype=np.float32),
                        np.asarray(ys,  dtype=np.float32),
                        np.asarray(yaws, dtype=np.float32),
                        np.full(n, 1.0 / n, dtype=np.float32),
                    ], axis=1)
                    np.save(os.path.join(
                        particles_dir, f"particles_{save_idx:06d}.npy"), arr)
            except Exception as e:
                print(f"[warn particles {save_idx}]  {e}")

        # ---- Pose files -------------------------------------------------
        if latest.ee is not None:
            po = latest.ee.pose
            with open(ee_path, "a") as f:
                f.write(f"{save_idx} "
                        f"{po.position.x} {po.position.y} {po.position.z} "
                        f"{po.orientation.x} {po.orientation.y} "
                        f"{po.orientation.z} {po.orientation.w}\n")

        if latest.amcl is not None:
            po = latest.amcl.pose.pose
            with open(amcl_path, "a") as f:
                f.write(f"{save_idx} "
                        f"{po.position.x} {po.position.y} {po.position.z} "
                        f"{po.orientation.x} {po.orientation.y} "
                        f"{po.orientation.z} {po.orientation.w}\n")
            cov = latest.amcl.pose.covariance
            with open(amcl_cov_path, "a") as f:
                f.write(f"{save_idx} " + " ".join(f"{v}" for v in cov) + "\n")

        # Joints map -- collect for a single json at the end.
        if latest.joints is not None:
            names = list(latest.joints.name)
            positions = list(latest.joints.position)
            joints_map[f"{save_idx:06d}"] = {
                n: float(p) for n, p in zip(names, positions)
            }

        # Timestamp (seconds, nanoseconds, stored separately)
        sec, nsec = rgb_t // int(1e9), rgb_t % int(1e9)
        with open(ts_path, "a") as f:
            f.write(f"{save_idx} {sec} {nsec}\n")

        save_idx += 1
        if save_idx % 50 == 0:
            print(f"  [{save_idx} frames]")

    # ---- Drive the reader --------------------------------------------------
    with AnyReader([bag]) as reader:
        # Only stream topics we care about.
        conns = [c for c in reader.connections if c.topic in topic_cat]
        start = reader.start_time
        next_emit_t = start + period_ns  # first tick after `start`

        for conn, timestamp, rawdata in reader.messages(connections=conns):
            cat = topic_cat[conn.topic]
            msg = reader.deserialize(rawdata, conn.msgtype)

            if cat == "rgb":
                latest.rgb = msg
                latest.rgb_t = timestamp
                # Emit a frame whenever RGB arrives past the next tick.
                if timestamp >= next_emit_t:
                    emit_frame(msg, timestamp)
                    # Schedule the following tick; skip ahead if we drifted.
                    while next_emit_t <= timestamp:
                        next_emit_t += period_ns
            elif cat == "depth":
                latest.depth = msg
                latest.depth_t = timestamp
            elif cat == "particles":
                latest.particles = msg
            elif cat == "ee":
                latest.ee = msg
            elif cat == "amcl":
                latest.amcl = msg
            elif cat == "joints":
                latest.joints = msg

    # Flush joints map
    with open(joints_path, "w") as f:
        json.dump(joints_map, f, indent=2)

    print(f"[extract] wrote {save_idx} frames to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", help="path to the .bag file")
    ap.add_argument("out_dir", help="output dataset root")
    ap.add_argument("--hz", type=float, default=5.0,
                    help="target sampling rate (default: 5)")
    args = ap.parse_args()
    extract(args.bag, args.out_dir, hz=args.hz)
    return 0


if __name__ == "__main__":
    sys.exit(main())
