#!/usr/bin/env python3
import os
import cv2
import numpy as np
import base64
import io
import json
from scipy.spatial.transform import Rotation
from eval_all import PoseEvaluator  # Important: Reuse existing parser logic

class PoseRenderer(PoseEvaluator):
    def __init__(self, dataset_dir):
        # Prevent initialization of global_csv_path in super() if not needed, 
        # but calling super().__init__() is safest.
        super().__init__(dataset_dir)
        
    def render(self, object_id, object_name, output_base_dir):
        # Update IDs based on user input
        self.object_id = object_id
        self.object_name = object_name
        
        self.eval_file = os.path.join(self.dataset_dir, "eval", f"object_{object_id}.txt")
        self.foundation_file = os.path.join(self.dataset_dir, "eval_foundationpose_comp", f"object_{object_id}.txt")
        self.bundle_sdf_file = os.path.join(self.dataset_dir, "eval_bundlesdf_comp", f"object_{object_id}.txt")
        self.midfusion_file = os.path.join(self.dataset_dir, "eval_midfusion", f"object_{object_id}.txt")
        
        # Load all pose lists
        self.estimated_poses, self.evaluation_segments = self.read_estimated_poses()
        self.foundation_poses = self.read_foundation_poses()
        self.bundle_sdf_poses = self.read_bundle_sdf_poses()
        self.midfusion_poses = self.read_midfusion_poses()
        
        # Output dirs setup
        methods_colors = {
            "gt": (0, 255, 0),         # Green
            "ours": (0, 0, 255),       # Red
            "foundationpose": (255, 0, 0), # Blue
            "bundlesdf": (0, 255, 255),# Yellow
            "midfusion": (255, 0, 255) # Magenta
        }
        
        for k in methods_colors.keys():
            os.makedirs(os.path.join(output_base_dir, k), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, "combined"), exist_ok=True)

        # Camera Intrinsics
        fx, fy = 554.3827, 554.3827
        cx, cy = 320.5, 240.5
        
        def project_points(points_3d):
            """ Project 3D points [N, 3] to 2D pixels [N, 2] """
            Z = points_3d[:, 2]
            valid = Z > 0
            points_3d = points_3d[valid]
            Z = Z[valid]
            u = (points_3d[:, 0] * fx / Z) + cx
            v = (points_3d[:, 1] * fy / Z) + cy
            return np.stack([u, v], axis=1).astype(np.int32)

        def draw_points(img, points_2d, color, radius=1):
            out = img.copy()
            for p in points_2d:
                if 0 <= p[0] < img.shape[1] and 0 <= p[1] < img.shape[0]:
                    cv2.circle(out, (p[0], p[1]), radius, color, -1)
            return out
            
        def add_label(img, text, color=(255, 255, 255)):
            out = img.copy()
            cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return out

        def filter_points_percentile(points_3d, lower_p=2.0, upper_p=98.0):
            if points_3d is None or len(points_3d) < 10:
                return points_3d
            # Filter based on percentile distances along each axis to robustly remove outliers
            p_low = np.percentile(points_3d, lower_p, axis=0)
            p_high = np.percentile(points_3d, upper_p, axis=0)
            
            mask = (
                (points_3d[:, 0] >= p_low[0]) & (points_3d[:, 0] <= p_high[0]) &
                (points_3d[:, 1] >= p_low[1]) & (points_3d[:, 1] <= p_high[1]) &
                (points_3d[:, 2] >= p_low[2]) & (points_3d[:, 2] <= p_high[2])
            )
            return points_3d[mask]

        if not self.has_offset:
            # Note: evaluate() checks this but hasn't run. We use default offset as eval_all does for fast test.
            self.has_offset = True

        for i, segment in enumerate(self.evaluation_segments):
            print(f"Renderer: segment {i+1}/{len(self.evaluation_segments)}")
            if len(segment) == 0: continue
            
            first_frame_idx = segment[0]
            # Compute transforming matrix from the first frame 
            self.compute_transformation_matrix(first_frame_idx)
            obj_points_ours = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            
            # Save the transformations since subsequent compute_transformation will overwrite them
            mocap_obj_trans = self.obj_transformation.copy() if hasattr(self, 'obj_transformation') else np.eye(4)
            mocap_cam_trans = self.camera_transformation.copy() if hasattr(self, 'camera_transformation') else np.eye(4)
            
            # --- Initialize FoundationPose ---
            if first_frame_idx in self.foundation_poses:
                self.compute_transformation_matrix_foundation_pose(first_frame_idx)
                obj_points_fp = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            else:
                obj_points_fp = None
                
            # --- Initialize BundleSDF ---
            if first_frame_idx in self.bundle_sdf_poses:
                self.compute_transformation_matrix_bundle_sdf(first_frame_idx)
                obj_points_bs = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            else:
                obj_points_bs = None
                
            # --- Initialize MidFusion ---
            if first_frame_idx in self.midfusion_poses:
                self.compute_transformation_matrix_midfusion(first_frame_idx)
                obj_points_mid = self.obj_points.copy() if getattr(self, "obj_points", None) is not None else None
            else:
                obj_points_mid = None
            
            if obj_points_ours is None:
                print("Failed to get object point cloud from first frame, skipping segment.")
                continue
            # Apply Point Cloud filtering to base points
            if obj_points_ours is not None: obj_points_ours = filter_points_percentile(obj_points_ours)
            if obj_points_fp is not None: obj_points_fp = filter_points_percentile(obj_points_fp)
            if obj_points_bs is not None: obj_points_bs = filter_points_percentile(obj_points_bs)
            if obj_points_mid is not None: obj_points_mid = filter_points_percentile(obj_points_mid)
            
            nearest_idx = None
            
            for frame_idx in segment:
                rgb_path = os.path.join(self.dataset_dir, "rgb", f"rgb_{frame_idx:06d}.png")
                if not os.path.exists(rgb_path):
                    continue
                
                rgb_img = cv2.imread(rgb_path)
                
                # Fetch Mocap nearest idx
                nearest_idx, _ = self.find_nearest_mocap_idx(
                    self.estimated_poses[frame_idx]['timestamp'], nearest_idx
                )
                if nearest_idx is None:
                    continue
                
                # --- GT Pose Mapping --- 
                mocap_obj_pose, mocap_cam_pose = self.extract_mocap_pose(nearest_idx)
                gt_obj_pose = self.mocap_robot @ mocap_obj_pose @ mocap_obj_trans
                gt_cam_pose = self.mocap_robot @ mocap_cam_pose @ mocap_cam_trans
                est_cam_pose = self.camera_poses.get(frame_idx, np.eye(4))
                
                gt_obj_pose_cam = np.linalg.inv(gt_cam_pose) @ gt_obj_pose
                points_gt = self.transform_points(obj_points_ours, gt_obj_pose_cam)
                
                img_gt = draw_points(rgb_img, project_points(points_gt), methods_colors["gt"])
                cv2.imwrite(os.path.join(output_base_dir, "gt", f"frame_{frame_idx:06d}.png"), img_gt)
                
                # --- Ours Pose Mapping --- 
                img_ours = rgb_img.copy()
                if frame_idx in self.estimated_poses and obj_points_ours is not None:
                    our_pose = self.estimated_poses[frame_idx]['transform']
                    our_pose_cam = np.linalg.inv(est_cam_pose) @ our_pose
                    points_ours = self.transform_points(obj_points_ours, our_pose_cam)
                    img_ours = draw_points(rgb_img, project_points(points_ours), methods_colors["ours"])
                cv2.imwrite(os.path.join(output_base_dir, "ours", f"frame_{frame_idx:06d}.png"), img_ours)
                
                # --- FoundationPose ---
                img_fp = rgb_img.copy()
                if frame_idx in self.foundation_poses and obj_points_fp is not None:
                    fp_pose = self.foundation_poses[frame_idx]['transform']
                    # Evaluator logic: est_pose = est_cam_pose @ fp_pose -> cam space logic: est_pose_cam = fp_pose
                    fp_pose_cam = fp_pose
                    points_fp = self.transform_points(obj_points_fp, fp_pose_cam)
                    img_fp = draw_points(rgb_img, project_points(points_fp), methods_colors["foundationpose"])
                cv2.imwrite(os.path.join(output_base_dir, "foundationpose", f"frame_{frame_idx:06d}.png"), img_fp)
                
                # --- BundleSDF ---
                img_bs = rgb_img.copy()
                if frame_idx in self.bundle_sdf_poses and obj_points_bs is not None:
                    bs_pose = self.bundle_sdf_poses[frame_idx]['transform']
                    bs_pose_cam = bs_pose
                    points_bs = self.transform_points(obj_points_bs, bs_pose_cam)
                    img_bs = draw_points(rgb_img, project_points(points_bs), methods_colors["bundlesdf"])
                cv2.imwrite(os.path.join(output_base_dir, "bundlesdf", f"frame_{frame_idx:06d}.png"), img_bs)
                
                # --- MidFusion ---
                img_mid = rgb_img.copy()
                if frame_idx in self.midfusion_poses and obj_points_mid is not None:
                    mid_pose = self.midfusion_poses[frame_idx]['transform']
                    mid_pose_cam = np.linalg.inv(est_cam_pose) @ mid_pose
                    points_mid = self.transform_points(obj_points_mid, mid_pose_cam)
                    img_mid = draw_points(rgb_img, project_points(points_mid), methods_colors["midfusion"])
                cv2.imwrite(os.path.join(output_base_dir, "midfusion", f"frame_{frame_idx:06d}.png"), img_mid)
                
                # --- Combined Plot ---
                h, w = rgb_img.shape[:2]
                combined_canvas = np.zeros((h*2, w*3, 3), dtype=np.uint8)
                
                combined_canvas[0:h, 0:w] = add_label(img_gt, "GT", methods_colors['gt'])
                combined_canvas[0:h, w:w*2] = add_label(img_ours, "Ours", methods_colors['ours'])
                combined_canvas[0:h, w*2:w*3] = add_label(img_fp, "FoundationPose", methods_colors['foundationpose'])
                
                combined_canvas[h:h*2, 0:w] = add_label(img_bs, "BundleSDF", methods_colors['bundlesdf'])
                combined_canvas[h:h*2, w:w*2] = add_label(img_mid, "MidFusion", methods_colors['midfusion'])
                combined_canvas[h:h*2, w*2:w*3] = add_label(rgb_img, "Original RGB")
                
                cv2.imwrite(os.path.join(output_base_dir, "combined", f"frame_{frame_idx:06d}.png"), combined_canvas)
                print(f"Rendered frame {frame_idx:06d}")

if __name__ == "__main__":
    dataset_dir = "/media/wby/6d811df4-bde7-479b-ab4c-679222653ea0/dataset_done/tomato_3"
    object_id = 1
    object_name = "tomato"
    
    # Setup paths 
    output_base_dir = os.path.join(dataset_dir, "render_output")
    os.makedirs(output_base_dir, exist_ok=True)
    
    print(f"Starting renderer for dataset: {dataset_dir}")
    renderer = PoseRenderer(dataset_dir)
    renderer.render(object_id, object_name, output_base_dir)
    print(f"Render completed. Images saved to: {output_base_dir}")
