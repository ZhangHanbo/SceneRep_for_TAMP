#!/usr/bin/env python3
"""
Sequential visualization of SceneRep API results.

Processes a trajectory frame by frame and produces:
1. Per-frame RGB with tracked object overlays (masks, IDs, labels)
2. Accumulated 3D reconstruction (point clouds + meshes) via Open3D
3. Object poses as coordinate axes
4. Spatial relation graph overlay

Output: a folder of numbered PNG frames + an Open3D 3D viewer at the end.

Usage:
    cd /Volumes/External/Workspace/nus_deliver/SceneRep_for_TAMP
    python tests/visualize_api.py [--frames 50] [--step 5] [--no-3d]

Data: Mobile_Manipulation_on_Fetch/multi_objects/apple_bowl_2/
"""

import os
import sys
import json
import argparse
import base64
from io import BytesIO
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation

# Ensure SceneRep root is on path
SCENEREP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SCENEREP_ROOT)

from api import (
    ObjectReconstructor, ObjectTracker, PoseUpdater, RelationAnalyzer,
    FrameDetections, RelationGraph,
)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

DATA_ROOT = os.path.join(
    os.path.dirname(SCENEREP_ROOT),
    "Mobile_Manipulation_on_Fetch", "multi_objects", "apple_bowl_2"
)

K = np.array([
    [554.3827, 0, 320.5],
    [0, 554.3827, 240.5],
    [0, 0, 1]
], dtype=np.float32)

# Distinct colors for up to 10 objects
COLORS = [
    (46, 204, 113),   # green
    (231, 76, 60),    # red
    (52, 152, 219),   # blue
    (241, 196, 15),   # yellow
    (155, 89, 182),   # purple
    (230, 126, 34),   # orange
    (26, 188, 156),   # teal
    (236, 112, 99),   # salmon
    (93, 173, 226),   # light blue
    (244, 208, 63),   # gold
]


# ─────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────

def load_pose_txt(path):
    poses = []
    with open(path, "r") as f:
        for line in f:
            arr = line.strip().split()
            if len(arr) != 8:
                continue
            _, tx, ty, tz, qx, qy, qz, qw = map(float, arr)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T[:3, 3] = [tx, ty, tz]
            poses.append(T)
    return poses


def load_detection_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    results = []
    for det in data.get("detections", []):
        mask_b64 = det.get("mask", "")
        if mask_b64:
            mask_bytes = base64.b64decode(mask_b64)
            mask_img = Image.open(BytesIO(mask_bytes)).convert("L")
            mask = (np.array(mask_img) > 128).astype(np.uint8)
        else:
            mask = np.zeros((480, 640), dtype=np.uint8)
        results.append({
            "mask": mask,
            "label": det.get("label", "unknown"),
            "score": det.get("score", 0.0),
            "id": det.get("object_id", 0),
            "box": det.get("box", [0, 0, 0, 0]),
        })
    return results


# ─────────────────────────────────────────────────────────────────────
# 2D Visualization helpers
# ─────────────────────────────────────────────────────────────────────

def draw_tracked_overlay(rgb, detections, tracked_objects, relations=None):
    """Draw masks, bounding boxes, labels, IDs, and relations on the RGB image."""
    vis = rgb.copy()
    h, w = vis.shape[:2]

    # Build a map from tracked object label to tracked info
    tracked_by_label = {}
    for obj in tracked_objects:
        tracked_by_label.setdefault(obj.label, []).append(obj)

    # Draw masks and labels from detections
    obj_centers = {}  # id -> (cx, cy) for relation arrows
    for i, det in enumerate(detections):
        mask = det["mask"]
        label = det["label"]
        box = det["box"]  # [x1, y1, x2, y2]
        score = det["score"]

        # Find the matching tracked object
        matched_obj = None
        if label in tracked_by_label and tracked_by_label[label]:
            matched_obj = tracked_by_label[label][0]
            # Don't pop — same label may appear in multiple dets

        obj_id = matched_obj.id if matched_obj else i
        color = COLORS[obj_id % len(COLORS)]

        # Semi-transparent mask overlay
        mask_bool = mask.astype(bool)
        if mask_bool.shape == (h, w):
            overlay = vis.copy()
            overlay[mask_bool] = color
            vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # Bounding box
        if len(box) == 4:
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label + ID + score
            text = f"[{obj_id}] {label} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            obj_centers[obj_id] = (cx, cy)

    # Draw relation arrows
    if relations is not None:
        for obj_id, rels in relations.relations.items():
            if obj_id not in obj_centers:
                continue
            cx1, cy1 = obj_centers[obj_id]
            for rel_type, targets in rels.items():
                for target_id in targets:
                    if target_id not in obj_centers:
                        continue
                    cx2, cy2 = obj_centers[target_id]

                    # Arrow from obj to target
                    arrow_color = {
                        "on": (0, 200, 0),
                        "in": (200, 200, 0),
                        "under": (0, 0, 200),
                        "contain": (200, 0, 200),
                    }.get(rel_type, (128, 128, 128))

                    cv2.arrowedLine(vis, (cx1, cy1), (cx2, cy2), arrow_color, 2, tipLength=0.15)

                    # Relation label at midpoint
                    mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
                    cv2.putText(vis, rel_type, (mx + 5, my - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, arrow_color, 1, cv2.LINE_AA)

    return vis


def draw_frame_info(vis, frame_idx, n_tracked, state_text=""):
    """Draw frame number and status info on the image."""
    h, w = vis.shape[:2]
    info = f"Frame {frame_idx:04d} | Tracked: {n_tracked}"
    if state_text:
        info += f" | {state_text}"
    cv2.rectangle(vis, (0, h - 30), (w, h), (0, 0, 0), -1)
    cv2.putText(vis, info, (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def draw_reconstruction_panel(tracked_objects, panel_size=(320, 240)):
    """Draw a top-down view of object point clouds as a small panel."""
    pw, ph = panel_size
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)

    if not tracked_objects:
        cv2.putText(panel, "No objects", (10, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        return panel

    # Collect all points to compute bounds
    all_pts = []
    obj_pts_list = []
    for obj in tracked_objects:
        if obj.points is not None and len(obj.points) > 0:
            all_pts.append(obj.points)
            obj_pts_list.append((obj.id, obj.label, obj.points, obj.pose))

    if not all_pts:
        cv2.putText(panel, "No point clouds yet", (10, ph // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
        return panel

    all_pts = np.vstack(all_pts)
    # Top-down: X-Y plane
    x_min, y_min = all_pts[:, 0].min(), all_pts[:, 1].min()
    x_max, y_max = all_pts[:, 0].max(), all_pts[:, 1].max()
    x_range = max(x_max - x_min, 0.1)
    y_range = max(y_max - y_min, 0.1)
    margin = 0.05
    x_min -= margin
    y_min -= margin
    x_range += 2 * margin
    y_range += 2 * margin

    def world_to_pixel(x, y):
        px = int((x - x_min) / x_range * (pw - 20)) + 10
        py = int((y - y_min) / y_range * (ph - 40)) + 10
        return px, py

    # Title
    cv2.putText(panel, "Top-down Reconstruction", (5, ph - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    # Draw points
    for obj_id, label, pts, pose in obj_pts_list:
        color = COLORS[obj_id % len(COLORS)]
        # Subsample for performance
        step = max(1, len(pts) // 200)
        for pt in pts[::step]:
            px, py = world_to_pixel(pt[0], pt[1])
            if 0 <= px < pw and 0 <= py < ph:
                cv2.circle(panel, (px, py), 1, color, -1)

        # Draw pose axis (small cross at object center)
        center = pose[:3, 3]
        cpx, cpy = world_to_pixel(center[0], center[1])

        # X-axis (red)
        x_axis = pose[:3, 0] * 0.03
        epx, epy = world_to_pixel(center[0] + x_axis[0], center[1] + x_axis[1])
        cv2.line(panel, (cpx, cpy), (epx, epy), (0, 0, 255), 1)

        # Y-axis (green)
        y_axis = pose[:3, 1] * 0.03
        epx, epy = world_to_pixel(center[0] + y_axis[0], center[1] + y_axis[1])
        cv2.line(panel, (cpx, cpy), (epx, epy), (0, 255, 0), 1)

        # Label
        cv2.putText(panel, f"[{obj_id}]{label}", (cpx + 3, cpy - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return panel


# ─────────────────────────────────────────────────────────────────────
# 3D Visualization (Open3D)
# ─────────────────────────────────────────────────────────────────────

def visualize_3d(tracked_objects, relations=None):
    """Show final 3D scene with Open3D: point clouds + poses + relation edges."""
    import open3d as o3d

    geometries = []

    for obj in tracked_objects:
        color = np.array(COLORS[obj.id % len(COLORS)]) / 255.0

        # Point cloud
        if obj.points is not None and len(obj.points) > 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(obj.points)
            pcd.paint_uniform_color(color)
            geometries.append(pcd)

        # Pose coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(obj.pose)
        geometries.append(frame)

    # Relation edges as line sets
    if relations is not None:
        obj_map = {obj.id: obj for obj in tracked_objects}
        lines = []
        line_colors = []
        points = []

        rel_color_map = {
            "on": [0, 0.8, 0],
            "in": [0.8, 0.8, 0],
            "under": [0, 0, 0.8],
            "contain": [0.8, 0, 0.8],
        }

        for obj_id, rels in relations.relations.items():
            if obj_id not in obj_map:
                continue
            p1 = obj_map[obj_id].pose[:3, 3]
            for rel_type, targets in rels.items():
                for target_id in targets:
                    if target_id not in obj_map:
                        continue
                    p2 = obj_map[target_id].pose[:3, 3]
                    idx = len(points)
                    points.extend([p1, p2])
                    lines.append([idx, idx + 1])
                    line_colors.append(rel_color_map.get(rel_type, [0.5, 0.5, 0.5]))

        if lines:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(np.array(points))
            ls.lines = o3d.utility.Vector2iVector(np.array(lines))
            ls.colors = o3d.utility.Vector3dVector(np.array(line_colors))
            geometries.append(ls)

    if geometries:
        print("\nOpening 3D viewer... (close the window to exit)")
        o3d.visualization.draw_geometries(
            geometries,
            window_name="SceneRep - Final 3D Reconstruction",
            width=1280, height=720,
        )


# ─────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize SceneRep API results")
    parser.add_argument("--data", default=DATA_ROOT, help="Path to trajectory data")
    parser.add_argument("--frames", type=int, default=80, help="Number of frames to process")
    parser.add_argument("--step", type=int, default=3, help="Process every N-th frame")
    parser.add_argument("--output", default=None, help="Output directory (default: tests/vis_output)")
    parser.add_argument("--no-3d", action="store_true", help="Skip Open3D 3D viewer")
    parser.add_argument("--relation-interval", type=int, default=10, help="Compute relations every N processed frames")
    args = parser.parse_args()

    data_root = args.data
    output_dir = args.output or os.path.join(SCENEREP_ROOT, "tests", "vis_output")
    os.makedirs(output_dir, exist_ok=True)

    # Load all poses
    cam_poses = load_pose_txt(os.path.join(data_root, "pose_txt", "camera_pose.txt"))
    ee_poses = load_pose_txt(os.path.join(data_root, "pose_txt", "ee_pose.txt"))
    l_finger_poses = load_pose_txt(os.path.join(data_root, "pose_txt", "l_gripper_pose.txt"))
    r_finger_poses = load_pose_txt(os.path.join(data_root, "pose_txt", "r_gripper_pose.txt"))

    n_total = min(args.frames, len(cam_poses))
    rgb_dir = os.path.join(data_root, "rgb")
    depth_dir = os.path.join(data_root, "depth")
    det_dir = os.path.join(data_root, "detection_h")

    # Initialize API objects
    tracker = ObjectTracker(K=K, voxel_size=0.003)
    reconstructor = ObjectReconstructor(voxel_size=0.003)
    relations = None

    # Track gripper state for status display
    last_finger_d = None

    print(f"Processing {n_total} frames (step={args.step}) from {data_root}")
    print(f"Output: {output_dir}")
    print()

    processed = 0
    for idx in range(0, n_total, args.step):
        rgb_path = os.path.join(rgb_dir, f"rgb_{idx:06d}.png")
        depth_path = os.path.join(depth_dir, f"depth_{idx:06d}.npy")
        det_path = os.path.join(det_dir, f"detection_{idx:06d}_final.json")

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        # Load data
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path).astype(np.float32)
        T_cw = cam_poses[idx]

        # Gripper state
        finger_d = np.linalg.norm(l_finger_poses[idx][:3, 3] - r_finger_poses[idx][:3, 3])
        if last_finger_d is not None:
            d_diff = finger_d - last_finger_d
            if d_diff < -0.002:
                state_text = "GRASPING"
            elif d_diff > 0.002:
                state_text = "RELEASING"
            else:
                state_text = "idle"
        else:
            state_text = "idle"
        last_finger_d = finger_d

        # Load detections
        detections = load_detection_json(det_path) if os.path.exists(det_path) else []

        # --- ObjectTracker: update ---
        if detections:
            fd = FrameDetections(
                labels=[d["label"] for d in detections],
                scores=np.array([d["score"] for d in detections]),
                masks=[d["mask"] for d in detections],
                bboxes=np.array([d["box"] for d in detections]),
            )
            tracked = tracker.update(fd, rgb, depth, T_cw)
        else:
            tracked = tracker._snapshot()

        # --- ObjectReconstructor: fuse each detected object ---
        for det in detections:
            label = det["label"]
            mask = det["mask"]
            # Find the tracked object matching this label
            for obj in tracker.internal_objects:
                if obj.label == label:
                    obj_id = obj.id
                    # Ensure it exists in the reconstructor
                    try:
                        reconstructor.get_points(obj_id)
                    except KeyError:
                        reconstructor.create(
                            pose=obj.pose_cur,
                            label=label,
                            object_id=obj_id,
                        )
                    # Fuse
                    reconstructor.fuse(obj_id, rgb, depth, K, T_cw, mask=mask)
                    break

        # --- RelationAnalyzer: compute periodically ---
        if processed > 0 and processed % args.relation_interval == 0:
            relations = RelationAnalyzer.compute(tracker.internal_objects, tolerance=0.02)

        # --- Build tracked objects with reconstruction data ---
        display_objects = []
        for obj in tracked:
            try:
                pts = reconstructor.get_points(obj.id)
                display_objects.append(type(obj)(
                    id=obj.id, label=obj.label, pose=obj.pose, points=pts
                ))
            except KeyError:
                display_objects.append(obj)

        # --- Visualize ---
        # Main view: RGB with tracked overlays and relations
        vis_main = draw_tracked_overlay(rgb, detections, tracked, relations)
        vis_main = draw_frame_info(vis_main, idx, len(tracked), state_text)

        # Side panel: top-down reconstruction
        recon_panel = draw_reconstruction_panel(display_objects)

        # Composite: main + panel
        main_h, main_w = vis_main.shape[:2]
        panel_h, panel_w = recon_panel.shape[:2]
        # Place panel at top-right corner of main image
        if panel_w <= main_w and panel_h <= main_h:
            vis_main[0:panel_h, main_w - panel_w:main_w] = recon_panel

        # Save
        out_path = os.path.join(output_dir, f"frame_{idx:04d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis_main, cv2.COLOR_RGB2BGR))

        processed += 1
        n_objects = len(tracked)
        n_recon = sum(1 for o in display_objects if len(o.points) > 0)
        rel_str = ""
        if relations:
            n_rels = sum(len(v) for rels in relations.relations.values() for v in rels.values())
            rel_str = f", relations={n_rels}"
        print(f"  [{processed:3d}] Frame {idx:04d}: {n_objects} tracked, {n_recon} reconstructed{rel_str} -> {out_path}")

    print(f"\nDone. {processed} frames saved to {output_dir}/")

    # Summary
    final_tracked = tracker._snapshot()
    print(f"\nFinal scene: {len(final_tracked)} objects")
    for obj in final_tracked:
        try:
            pts = reconstructor.get_points(obj.id)
            n_pts = len(pts)
        except KeyError:
            n_pts = 0
        print(f"  [{obj.id}] {obj.label}: {n_pts} points, pose={obj.pose[:3, 3].tolist()}")

    if relations:
        print(f"\nSpatial relations:")
        for obj_id, rels in relations.relations.items():
            for rel_type, targets in rels.items():
                if targets:
                    print(f"  Object {obj_id} {rel_type} {targets}")

    # 3D viewer
    if not args.no_3d:
        # Build display objects with reconstructed points
        final_display = []
        for obj in final_tracked:
            try:
                pts = reconstructor.get_points(obj.id)
            except KeyError:
                pts = obj.points
            final_display.append(type(obj)(
                id=obj.id, label=obj.label, pose=obj.pose, points=pts
            ))
        visualize_3d(final_display, relations)


if __name__ == "__main__":
    main()
