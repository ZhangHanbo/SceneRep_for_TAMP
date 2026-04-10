"""
SceneRep Public API
===================

Consumer-driven API for robi_butler integration.
All external consumers should import from here.

Usage:
    from SceneRep_for_TAMP.api import ObjectReconstructor, ObjectTracker, PoseUpdater, RelationAnalyzer
    from SceneRep_for_TAMP.api import Mesh, TrackedObject, FrameDetections, RelationGraph
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any


# ─────────────────────────────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────────────────────────────

@dataclass
class Mesh:
    """3D mesh extracted from TSDF reconstruction."""
    vertices: np.ndarray    # (V, 3)
    faces: np.ndarray       # (F, 3)
    normals: np.ndarray     # (V, 3)
    colors: np.ndarray      # (V, 3)

    @property
    def is_empty(self) -> bool:
        return self.vertices.shape[0] == 0


@dataclass
class TrackedObject:
    """A tracked object in the scene with 3D state."""
    id: int
    label: str
    pose: np.ndarray            # 4x4 current SE(3) in world frame
    points: np.ndarray          # (N, 3) accumulated world points
    mesh: Optional[Mesh] = None


@dataclass
class FrameDetections:
    """Detection results for a single frame, used as input to ObjectTracker."""
    labels: List[str]
    scores: np.ndarray          # (N,)
    masks: List[np.ndarray]     # list of (H, W) bool arrays
    bboxes: np.ndarray          # (N, 4) pixel coordinates [x1,y1,x2,y2]


@dataclass
class RelationGraph:
    """Geometric spatial relations between objects."""
    relations: Dict[int, Dict[str, List[int]]]
    # e.g. {obj_id: {"on": [other_ids], "in": [other_ids], "under": [...], "contain": [...]}}


# ─────────────────────────────────────────────────────────────────────
#  ObjectReconstructor
# ─────────────────────────────────────────────────────────────────────

class ObjectReconstructor:
    """TSDF-based single-object 3D reconstruction from multi-view observations.

    Manages SceneObject instances internally and provides a clean interface
    for creating objects, fusing RGBD frames, and extracting meshes.

    Consumer: robi_butler's observe_behavior.
    """

    def __init__(self, voxel_size: float = 0.002):
        """
        Args:
            voxel_size: TSDF voxel resolution in meters.
        """
        self._voxel_size = voxel_size
        self._objects: Dict[int, Any] = {}  # id -> SceneObject
        self._next_id = 0

    def create(self, pose: np.ndarray, label: str,
               object_id: Optional[int] = None) -> int:
        """Create a new tracked object with TSDF volume.

        Args:
            pose: (4, 4) SE(3) initial world pose.
            label: Object class label.
            object_id: Optional explicit ID. Auto-assigned if None.

        Returns:
            Assigned object ID.
        """
        from scene.scene_object import SceneObject

        if object_id is None:
            object_id = self._next_id
            self._next_id += 1

        obj = SceneObject(
            pose=pose,
            id=object_id,
            initial_label=label,
            voxel_size=self._voxel_size,
        )
        self._objects[object_id] = obj
        return object_id

    def fuse(self, object_id: int,
             color: np.ndarray, depth: np.ndarray,
             K: np.ndarray, T_cw: np.ndarray,
             mask: Optional[np.ndarray] = None) -> bool:
        """Integrate one RGBD frame into the object's TSDF.

        IMPORTANT: All geometry is accumulated in the object's *initial pose
        frame* (pose_init), NOT in the current world frame.  This is achieved
        by warping the camera transform:

            T_cw_fix = (pose_init @ inv(pose_cur)) @ T_cw

        This means if the object has moved (pose_cur != pose_init), new
        observations are still registered correctly against the original TSDF.

        Callers MUST NOT call fuse() while the object is being grasped or
        moved — the pose_cur during manipulation is updated by PoseUpdater,
        but TSDF integration during fast motion produces artifacts.  Skip
        fusion during GRASPING / HOLDING / RELEASING states.

        Args:
            object_id: ID of the object to update.
            color: (H, W, 3) RGB image, uint8.
            depth: (H, W) depth image in meters, float32.
            K: (3, 3) camera intrinsic matrix.
            T_cw: (4, 4) camera-to-world transform.
            mask: Optional (H, W) bool mask to limit fusion region.
                  If provided, depth outside mask is zeroed before fusion.

        Returns:
            True if fusion succeeded, False otherwise.
        """
        if object_id not in self._objects:
            raise KeyError(f"Object {object_id} not found. Call create() first.")

        obj = self._objects[object_id]

        # Skip if the object's pose is uncertain (e.g. being manipulated)
        if getattr(obj, 'pose_uncertain', False):
            return False

        # Apply mask: zero out depth outside the object region
        depth_obj = depth.copy()
        color_obj = color.copy()
        if mask is not None:
            depth_obj[~mask.astype(bool)] = 0.0
            color_obj[~mask.astype(bool)] = 0

        # Warp camera transform into the object's initial frame.
        # This ensures all TSDF/point data is accumulated in a consistent
        # coordinate frame even if the object has moved (pose_cur changed).
        #   T_warp = pose_init @ inv(pose_cur)
        #   T_cw_fix = T_warp @ T_cw
        T_warp = obj.pose_init @ np.linalg.inv(obj.pose_cur)
        T_cw_fix = T_warp @ T_cw
        T_wc_fix = np.linalg.inv(T_cw_fix).astype(np.float64)

        # Integrate into object's TSDF in the initial-pose frame
        success = obj.tsdf.integrate(
            color=color_obj,
            depth=depth_obj,
            K=K,
            T=T_wc_fix,
        )

        # Add points to the object's point cloud (also in initial-pose frame)
        if mask is not None:
            from utils.utils import _mask_to_world_pts_colors
            pts, colors_out = _mask_to_world_pts_colors(
                mask, depth, color, K, T_cw_fix
            )
            if pts is not None and len(pts) > 0:
                obj.add_points(pts, colors=colors_out)

        return success

    def get_mesh(self, object_id: int) -> Mesh:
        """Extract mesh from object's TSDF volume.

        Args:
            object_id: ID of the object.

        Returns:
            Mesh with vertices, faces, normals, colors.
        """
        if object_id not in self._objects:
            raise KeyError(f"Object {object_id} not found.")

        obj = self._objects[object_id]
        V, F, N, C = obj.tsdf.get_mesh()
        return Mesh(vertices=V, faces=F, normals=N, colors=C)

    def get_points(self, object_id: int) -> np.ndarray:
        """Get accumulated point cloud in world frame.

        Args:
            object_id: ID of the object.

        Returns:
            (N, 3) numpy array of world-frame points.
        """
        if object_id not in self._objects:
            raise KeyError(f"Object {object_id} not found.")

        obj = self._objects[object_id]
        if hasattr(obj, '_points') and obj._points is not None:
            return obj._points.copy()
        return np.empty((0, 3))

    def get_object(self, object_id: int) -> TrackedObject:
        """Get a TrackedObject snapshot for external use.

        Args:
            object_id: ID of the object.

        Returns:
            TrackedObject dataclass.
        """
        if object_id not in self._objects:
            raise KeyError(f"Object {object_id} not found.")

        obj = self._objects[object_id]
        mesh = None
        try:
            V, F, N, C = obj.tsdf.get_mesh()
            if V.shape[0] > 0:
                mesh = Mesh(vertices=V, faces=F, normals=N, colors=C)
        except Exception:
            pass

        return TrackedObject(
            id=object_id,
            label=obj.label,
            pose=obj.pose_cur.copy(),
            points=self.get_points(object_id),
            mesh=mesh,
        )


# ─────────────────────────────────────────────────────────────────────
#  ObjectTracker
# ─────────────────────────────────────────────────────────────────────

class ObjectTracker:
    """Track object identities across frames using detection matching.

    Uses label-based association with point cloud distance to
    match new detections with existing tracked objects.
    Internally delegates to scene.id_associator.associate_by_id().

    Consumer: robi_butler's SceneManager.
    """

    def __init__(self, K: np.ndarray, voxel_size: float = 0.002):
        """
        Args:
            K: (3, 3) camera intrinsic matrix.
            voxel_size: TSDF voxel size for new objects.
        """
        self._K = K.copy()
        self._voxel_size = voxel_size
        self._objects: List[Any] = []  # internal SceneObject list
        self._frame_count = 0

    def update(self, detections: FrameDetections,
               rgb: np.ndarray, depth: np.ndarray,
               T_cw: np.ndarray,
               integrate: bool = True) -> List[TrackedObject]:
        """Match new detections to existing objects.

        New detections that don't match any existing object create new tracked
        objects. Existing objects that match get their points/TSDF updated.

        Args:
            detections: FrameDetections for the current frame.
            rgb: (H, W, 3) RGB image.
            depth: (H, W) depth image in meters (float32).
            T_cw: (4, 4) camera-to-world transform.
            integrate: Whether to integrate TSDF for matched objects.

        Returns:
            Updated list of TrackedObject snapshots.
        """
        from scene.id_associator import associate_by_id

        # Convert FrameDetections to the list-of-dicts format expected by associate_by_id
        masks_dicts = []
        for i in range(len(detections.labels)):
            masks_dicts.append({
                "mask": detections.masks[i].astype(np.uint8) if detections.masks[i].dtype == bool else detections.masks[i],
                "label": detections.labels[i],
                "score": float(detections.scores[i]),
                "id": i,
            })

        associate_by_id(
            masks=masks_dicts,
            depth=depth,
            rgb=rgb,
            K=self._K,
            T_cw=T_cw,
            objects=self._objects,
            frame_id=self._frame_count,
            voxel_size=self._voxel_size,
            integrate=integrate,
        )
        self._frame_count += 1

        return self._snapshot()

    def _snapshot(self) -> List[TrackedObject]:
        """Convert internal SceneObjects to TrackedObject snapshots."""
        result = []
        for obj in self._objects:
            pts = obj._points.copy() if hasattr(obj, '_points') and obj._points is not None else np.empty((0, 3))
            result.append(TrackedObject(
                id=obj.id,
                label=obj.label,
                pose=obj.pose_cur.copy(),
                points=pts,
            ))
        return result

    def set_held_object(self, obj_id: Optional[int]):
        """Mark an object as held by the gripper (or release it).

        While held, the object is skipped during associate_by_id (its
        pose_uncertain flag is set) and fusion is blocked.

        Args:
            obj_id: ID of the held object, or None to release.
        """
        for obj in self._objects:
            if obj.id == obj_id:
                obj.pose_uncertain = True
            # When releasing, we keep pose_uncertain=True until the
            # object is re-observed and ICP succeeds (handled by icp_reappear).

    def release_object(self, obj_id: int):
        """Mark a previously held object as released.

        The object's pose_uncertain remains True until the next
        successful observation re-localizes it.

        Args:
            obj_id: ID of the released object.
        """
        for obj in self._objects:
            if obj.id == obj_id:
                obj.T_oe = None  # clear EE-object offset

    def detect_held_object(self, T_ew: np.ndarray,
                           box_size: Tuple = (0.07, 0.05, 0.05)) -> Optional[int]:
        """Detect which object (if any) is inside the end-effector grasp box.

        Args:
            T_ew: (4, 4) end-effector to world transform.
            box_size: (length, width, height) of the grasp detection box in meters.

        Returns:
            Object ID of the held object, or None.
        """
        ee_center = T_ew[:3, 3]
        half = np.array(box_size) / 2.0
        ee_axes = T_ew[:3, :3]  # columns are x, y, z axes

        best_id = None
        best_count = 0

        for obj in self._objects:
            if not hasattr(obj, '_points') or len(obj._points) == 0:
                continue
            try:
                V, F, N, C = obj.tsdf.get_mesh()
                if len(V) == 0:
                    continue
                # Transform mesh to current world frame
                world_pts = ((obj.pose_cur @ np.linalg.inv(obj.pose_init))
                             @ np.hstack([V, np.ones((len(V), 1))]).T).T[:, :3]
            except Exception:
                continue

            # Project into EE local frame
            local_pts = (world_pts - ee_center) @ ee_axes  # (N, 3)
            in_box = np.all(np.abs(local_pts) <= half, axis=1)
            count = int(np.sum(in_box))
            if count > best_count:
                best_count = count
                best_id = obj.id

        return best_id if best_count >= 1 else None

    @property
    def internal_objects(self) -> list:
        """Access internal SceneObject list (for PoseUpdater and RelationAnalyzer)."""
        return self._objects


# ─────────────────────────────────────────────────────────────────────
#  PoseUpdater
# ─────────────────────────────────────────────────────────────────────

class PoseUpdater:
    """Update object poses during manipulation.

    Provides static methods for end-effector-based and ICP-based pose updates.

    Consumer: robi_butler's SceneManager.
    """

    @staticmethod
    def update_from_ee(objects: list, obj_id: int,
                       T_cw: np.ndarray, T_ec: np.ndarray) -> bool:
        """Update object pose based on end-effector transform.

        When the robot is holding an object, track its pose via the
        known end-effector-to-camera transform chain.

        Args:
            objects: List of internal SceneObject instances (from ObjectTracker).
            obj_id: ID of the held object.
            T_cw: (4, 4) camera-to-world transform.
            T_ec: (4, 4) end-effector-to-camera transform.

        Returns:
            True if update succeeded.
        """
        from pose_update.object_pose_updater import update_obj_pose_ee
        return update_obj_pose_ee(objects, obj_id, T_cw, T_ec)

    @staticmethod
    def update_from_icp(objects: list, obj_id: int,
                        new_points: np.ndarray,
                        max_correspondence_distance: float = 0.02) -> Optional[np.ndarray]:
        """Refine object pose via ICP against new observation.

        Args:
            objects: List of internal SceneObject instances.
            obj_id: ID of the object to refine.
            new_points: (N, 3) new observation points in world frame.
            max_correspondence_distance: ICP max correspondence distance.

        Returns:
            (4, 4) refined pose matrix, or None if ICP failed.
        """
        from pose_update.object_pose_updater import update_obj_pose_icp
        from utils.utils import find_object_by_id

        obj = find_object_by_id(obj_id, objects)
        if obj is None:
            return None

        return update_obj_pose_icp(
            obj, new_points,
            max_correspondence_distance=max_correspondence_distance,
        )


# ─────────────────────────────────────────────────────────────────────
#  RelationAnalyzer
# ─────────────────────────────────────────────────────────────────────

class RelationAnalyzer:
    """Compute geometric spatial relations between tracked objects.

    Analyzes 3D point clouds to determine spatial relations:
    on, in, under, contain.

    Consumer: robi_butler's SceneManager.
    """

    @staticmethod
    def compute(objects: list, tolerance: float = 0.02) -> RelationGraph:
        """Compute spatial relations between objects.

        Args:
            objects: List of internal SceneObject instances (from ObjectTracker).
            tolerance: Spatial tolerance in meters for relation detection.

        Returns:
            RelationGraph with per-object relation dictionaries.
        """
        from scene.object_relation_graph import compute_spatial_relations

        if len(objects) == 0:
            return RelationGraph(relations={})

        relations_dict = compute_spatial_relations(objects, tolerance=tolerance)
        return RelationGraph(relations=relations_dict)
