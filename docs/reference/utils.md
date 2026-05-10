# `utils/` — shared helpers

Small, focused modules that don't belong to a tracker but are used across
trackers, scripts, and tests.

## SLAM ↔ Layer-2 boundary

| Module | What it gives you |
|---|---|
| `utils.slam_interface` | `PoseEstimate` (Gaussian) and `ParticlePose` (RBPF) datatypes; `PassThroughSlam` (identity SLAM for tests); the masking protocol that keeps movable-object pixels out of SLAM. |
| `utils.base_pose_backend` | Abstract base class for SLAM backends; required methods: `ensure_object`, `delete_object`, `predict_static`, `rigid_attachment_predict`, `camera_frame_prior`, `innovation_stats`, `absorb_likelihoods`. |

## SE(3) EKF math

`utils.ekf_se3` is the per-object EKF kernel.  Key entry points:

| Function | Purpose |
|---|---|
| `se3_exp(xi) → (4,4)` | Exponential map from Lie-algebra `[v, ω]` to SE(3). |
| `se3_log(T) → (6,)` | Logarithm; inverse of `se3_exp`. |
| `se3_adjoint(T) → (6,6)` | SE(3) adjoint for covariance transport. |
| `ekf_predict(...)` | Standard EKF predict step (mean + Joseph cov update). |
| `ekf_update(...)` | EKF update on a generic measurement. |
| `ekf_update_base_frame(...)` | Frame-agnostic update used by the EKF tracker. |
| `compose_observation_noise(...)` | Builds `R` from ICP fitness/RMSE + base motion. |
| `pose_entropy(P) → float` | Differential entropy of the 6×6 pose covariance. |
| `process_noise_for_phase(phase) → (6,6)` | Q schedule per gripper-FSM phase. |
| `huber_weight(d) → float` | Adaptive-kernel weight for the Mahalanobis residual. |
| `saturate_covariance(P, P_min, P_max) → (6,6)` | Per-axis floor + ceiling clamp. |

## Object belief

`utils.object_belief` packs the per-object Gaussian into a frame-agnostic
container with `predict_ad_conjugate`, `innovation_from_belief`, and
`update_from_innovation`.  Used by both the heuristic tracker (lightly) and
the EKF tracker (centrally).

## Gripper

| Module | Notes |
|---|---|
| `utils.gripper_state` | `GripperPhaseTracker` FSM: `idle → grasping → holding → releasing`.  Per-frame summary: `phase`, `held_oid`, `width`, `is_moving`, `grasp_count`. |
| `utils.gripper_geometry` | Abstract `GripperGeometry` base class.  Concrete subclass for Fetch with hardware-derived AABB and pad volumes. |
| `utils.hand_mask_utils` | Generates hand / end-effector masks (gripper-box AABB projected to image). |
| `utils.robot_models.fetch` | Fetch `GripperGeometry` concrete subclass. |
| `utils.robot_models.__init__` | Factory `create_gripper_geometry(robot_type)`. |

## Fetch arm

| Module | Purpose |
|---|---|
| `utils.fetch_arm_fk` | Lightweight URDF parser for the Fetch arm.  Reads XML, walks the kinematic chain, outputs per-link `T_base_link` from joint angles.  No external URDF loader dependency (only `xml.etree.ElementTree` + scipy). |
| `utils.fetch_arm_mask` | `ArmMaskBuilder` — projects per-link convex hulls onto the image, dilates, returns 2D arm silhouette mask.  Used by `scripts/visualize_arm_mask.py` and `visualize_voxel_obs.py --mask-arm`. |
| `utils.fetch_kinematics` | TF helper for the base→head_camera chain; documents why URDF-hardcoded FK differs from published `/tf` (factory calibration biases).  Main consumer: `scripts/rosbag2dataset/`. |

## Image / mesh

| Module | Purpose |
|---|---|
| `utils.inpaint_utils` | Depth + colour inpainting (`inpaint_color_pyramid` is the multi-scale workhorse).  Used by `data_demo.py` to fill missing depth in object regions. |
| `utils.mesh_filter_fast` | Percentile-box mesh filter; removes outlier vertices along x/y/z axes and re-maps face indices.  Used post-TSDF to clean noisy reconstructions. |

## Object dynamics

`utils.object_dynamics` is a tiny per-label dynamics table (`restitution e`,
`friction μ`, `shape primitive`) used by the gravity-drop predictor.
Lookup: `lookup_dynamics(label, override=None)`.

## Eval helpers

`utils.eval_save_utils` — `ObjectPoseRecorder` saves per-frame object poses
and evaluations to `pose_txt/` and `eval/` directories.

## Misc

`utils.utils` collects small helpers: mesh rendering, mask↔points
conversion, pose math, optional ROS bridge utilities.

```{seealso}
* [Architecture overview](../architecture/overview.md) — where these helpers
  fit into the layered design.
* [`bernoulli_ekf.pdf`](../ekf_tracker/latex/bernoulli_ekf.pdf) Part I —
  derivation of the SE(3) EKF kernel implemented in `ekf_se3.py`.
```
