# SR_TAMP Codebase Documentation

## 1. Overview
This repository implements a **Scene Representation** system for robotic manipulation (TAMP - Task and Motion Planning). It constructs a 3D semantic understanding of the scene by fusing RGB-D data, tracking objects, and dynamic pose updates based on robot interactions.

The core capability is **real-time object-level scene reconstruction** that accounts for:
*   **Static Objects**: Fusing depth data into TSDF volumes.
*   **Dynamic Interaction**: Tracking objects attached to the robot gripper.
*   **Relational Movement**: Updating "child" objects (e.g., items inside a container) when the parent object moves.
*   **Pose Correction**: Using ICP (Iterative Closest Point) to refine object poses against visual observations.

## 2. Key Entry Points

### 2.1. `data_demo.py` (Offline Pipeline)
*   **Purpose**: Processes pre-recorded datasets (ROS bags converted to images/poses) for testing and algorithms development.
*   **Input**: Directory containing `rgb/`, `depth/`, `pose_txt/` (camera, ee, gripper poses).
*   **Workflow**:
    1.  Loads dataset and config (`configs/data_demo_config.yaml`).
    2.  Iterates through frames.
    3.  **State Machine**: Determines robot state (`idle`, `grasping`, `holding`, `releasing`) based on gripper finger distance.
    4.  **Detection**: Loads/Computes object masks and bounding boxes.
    5.  **Pose Update**:
        *   If `holding`: Lock object pose to End-Effector (EE).
        *   If `moving`: Update child objects via relative transforms.
        *   Refine poses using Colored ICP.
    6.  **Fusion**: Integrates depth into `TSDFVolume` for background and objects.
    7.  **Visualization**: Renders the scene using `pyrender`.

### 2.2. `realtime_app.py` (ROS Application)
*   **Purpose**: Online version for real-time robot operation.
*   **Input**: ROS topics (`/rgb`, `/depth`, `/end_effector_pose`, `/tf`).
*   **Workflow**:
    *   Subscribes to ROS topics.
    *   Uses `LatestCache` to process the most recent available frame (skipping stale frames).
    *   Uses `tf2_ros` to look up transformations (Camera-to-World, EE-to-Camera, etc.).
    *   Reuses core logic from `scene` and `pose_update` modules to maintain the scene graph.
    *   Saves processed frames to disk allows for "online recording".

### 2.3. `rosbag2dataset/`
*   Contains scripts to convert raw ROS bags into the dataset format used by `data_demo.py`.
    *   `rosbag2dataset_5hz.py`: Extracts images and poses.
    *   `owl/`, `sam/`: Scripts for generating 2D semantic masks (OWL-ViT, Segment Anything).

## 3. Core Modules

### 3.1. `scene/` (Scene Representation)
*   **`scene_object.py`**: Defines `SceneObject`.
    *   Attributes: `pose` (SE3), `label`, `confidence`, `tsdf` (voxel volume), `points` (point cloud).
    *   Manages object lifecycle (creation, fusion, tracking).
*   **`tsdf_o3d.py`**: TSDF implementation (likely wrapping Open3D functions) for fusing depth maps into 3D meshes.
*   **`id_associator.py`**: Logic to associate 2D masks from the current frame to existing 3D objects in the scene (Data Association).

### 3.2. `detection/` (Perception)
*   **`hungarian_detection.py`**: Handles matching of new detections to existing objects.
    *   Uses **Hungarian Algorithm** based on IoU (2D/3D) and Point Cloud Overlap.
    *   Logic to decide when to spawn a *new* object vs. update an *existing* one.
*   **`det_server.py`**: A server script (likely HTTP or socket) to offload heavy inference (like OWL-ViT/SAM) to a separate process/machine.

### 3.3. `pose_update/` (Dynamics & Tracking)
*   **`object_pose_updater.py`**:
    *   `update_obj_pose_ee`: Hard-updates object pose based on Gripper transform (rigid attachment).
    *   `update_child_objects_pose_icp`: Updates objects "in" or "on" the held object.
    *   `icp_reappear`: Refines object pose when it reappears or drifts, using Point-to-Plane ICP.
*   **`object_relation_graph.py`**: Infers spatial relations (`on`, `in`, `under`, `contain`) between objects to build a dependency graph.

## 4. Configuration
*   **`configs/`**: Contains YAML files controlling parameters like:
    *   Camera intrinsics.
    *   TSDF voxel sizes.
    *   Detection thresholds.
    *   ICP parameters.
    *   Debugging flags (visualization).

## 5. Summary Flow
1.  **Perception**: Get RGB-D + Camera Pose.
2.  **Detection**: Detect objects (2D Mask) -> Associate with 3D IDs.
3.  **State Check**: Is robot grasping? (Check gripper width).
4.  **Dynamics**:
    *   If grasping → Parent object follows Gripper.
    *   Dependents (children) follow Parent.
    *   Free objects → Refine via ICP if visible.
5.  **Fusion**: Update object shape (TSDF) with new depth data.
6.  **Graph Update**: Re-evaluate spatial relations (e.g., if object was placed into a bin).
