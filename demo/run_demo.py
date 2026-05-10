"""Minimal Dynamic Scene Graph demo.

Extracts ``demo/apple_in_the_tray.zip`` (a 37-frame slice of a real
Fetch trajectory) into the repo root if it isn't already extracted,
then runs the EKF tracker on every extracted frame and prints the
per-frame world-frame object poses + Bernoulli existence.

Run from the repo root:

    python demo/run_demo.py

No external services are needed (relation backend is forced to ``none``).
"""
from __future__ import annotations

import base64
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DEMO_DIR = Path(__file__).resolve().parent
ZIP_PATH = DEMO_DIR / "apple_in_the_tray.zip"

DATASET = REPO / "datasets" / "apple_in_the_tray"
DET_DIR = REPO / "tests" / "visualization_pipeline" / "apple_in_the_tray" / "perception" / "detection_h"

K = np.array(
    [[554.3827, 0.0, 320.5],
     [0.0, 554.3827, 240.5],
     [0.0, 0.0, 1.0]],
    dtype=np.float64,
)


def _maybe_extract() -> None:
    """Unzip the demo trajectory next to the repo root, idempotent."""
    if (DATASET / "rgb").is_dir() and (DET_DIR).is_dir():
        return
    print(f"Extracting {ZIP_PATH.name} → {REPO}…")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(REPO)


def _quat_to_T(values: list[float]) -> np.ndarray:
    """Convert ``[tx ty tz qx qy qz qw]`` to a 4×4 SE(3) matrix."""
    tx, ty, tz, qx, qy, qz, qw = values
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    R = np.array(
        [[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
         [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
         [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (tx, ty, tz)
    return T


def _load_pose_table(path: Path) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    with path.open() as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            idx = int(parts[0])
            out[idx] = _quat_to_T([float(x) for x in parts[1:8]])
    return out


def _load_joints(path: Path) -> dict[int, dict[str, float]]:
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.items()}


def _gripper_width(joint_state: dict[str, float] | None) -> float | None:
    if not joint_state:
        return None
    l = joint_state.get("l_gripper_finger_joint")
    r = joint_state.get("r_gripper_finger_joint")
    if l is None and r is None:
        return None
    return float((l or 0.0) + (r or 0.0))


def _decode_mask(b64: str, h: int, w: int) -> np.ndarray:
    import io
    from PIL import Image

    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw))
    return (np.asarray(img) > 0).astype(np.uint8)


def _load_detections(path: Path, h: int, w: int) -> list[dict]:
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    detections = raw.get("detections", raw) if isinstance(raw, dict) else raw
    out = []
    for d in detections:
        if "mask" in d and isinstance(d["mask"], str):
            d = {**d, "mask": _decode_mask(d["mask"], h, w)}
        out.append(d)
    return out


def main() -> int:
    _maybe_extract()

    pose_dir = DATASET / "pose_txt"
    slam = _load_pose_table(pose_dir / "amcl_pose.txt")
    T_bc_map = _load_pose_table(pose_dir / "T_bc.txt")
    T_bg_map = _load_pose_table(pose_dir / "ee_pose.txt")
    joints_map = _load_joints(pose_dir / "joints_pose.json")

    from ekf_tracker import EkfTracker

    rgb_files = sorted((DATASET / "rgb").glob("rgb_*.png"))
    if not rgb_files:
        print("No RGB frames found after extraction.", file=sys.stderr)
        return 1
    indices = [int(p.stem.split("_")[1]) for p in rgb_files]

    tracker = EkfTracker(K=K, T_bc=T_bc_map.get(indices[0]), relation_backend="none")

    print(f"Tracking {len(indices)} frames "
          f"(idx {indices[0]}–{indices[-1]})…\n")
    for idx in indices:
        rgb_path = DATASET / "rgb" / f"rgb_{idx:06d}.png"
        depth_path = DATASET / "depth" / f"depth_{idx:06d}.npy"
        det_path = DET_DIR / f"detection_{idx:06d}_final.json"

        from PIL import Image
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        depth = np.load(depth_path).astype(np.float32)
        h, w = depth.shape
        detections = _load_detections(det_path, h, w)

        T_wb = slam.get(idx)
        T_bc = T_bc_map.get(idx)
        T_bg = T_bg_map.get(idx)
        joints = joints_map.get(f"{idx:06d}") or joints_map.get(idx)
        width = _gripper_width(joints)

        scene = tracker.step(
            detections, rgb, depth,
            slam_pose=T_wb,
            T_bc=T_bc,
            T_bg=T_bg,
            gripper_width=width,
            joints=joints,
        )
        objs = ", ".join(
            f"{oid}:{obj.label}@{obj.pose[:3, 3].round(3).tolist()} r={obj.r:.2f}"
            for oid, obj in scene.objects.items()
        ) or "—"
        print(f"frame {idx:04d}  objects: {objs}")

    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
