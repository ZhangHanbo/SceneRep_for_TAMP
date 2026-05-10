# Configuration reference

The repo has **two completely independent** config schemas, one per
tracker family.  They don't share keys.

## EKF tracker — strict two-hierarchy YAML

The EKF pipeline reads parameters from exactly two files:

| File | Role |
|---|---|
| `ekf_tracker/configs/default.yaml` | Canonical defaults — read-only. |
| `configs/ekf_tracker/customization.yaml` | User overrides; `_extends:` the default and deep-merges. |

**Missing keys hard-error at load time.**  No env vars, no constructor
fallbacks, no test-fixture YAMLs.

```python
from ekf_tracker.configs import load_config, to_bernoulli_config, to_trigger_config

cfg  = load_config("configs/ekf_tracker/customization.yaml")
bcfg = to_bernoulli_config(cfg, K=K, image_shape=(480, 640))
tcfg = to_trigger_config(cfg)
```

### Top-level sections

| Section | Maps to | What it controls |
|---|---|---|
| `bernoulli` | {py:class}`ekf_tracker.config.BernoulliConfig` | Fast-tier existence model + EKF gates + birth quality + cov clamping |
| `trigger` | {py:class}`ekf_tracker.config.TriggerConfig` | Slow-tier (factor graph) schedule |
| `ekf_tracker` | `EkfTracker.__init__` runtime knobs | Pose method, image shape, robot type |
| `process_noise` | `utils/ekf_se3.py` | Per-phase Q schedule (idle / grasping / holding / releasing / falling) |
| `fast_tier_noise` | `gaussian_ekf_tracker.py` constants | Centroid std, rotation decouple, tiny-cov |
| `pose_estimator` | `perception/icp_pose.py` PoseEstimator class | ICP voxel size, threshold, fitness floors |
| `perception` | per-call defaults | `visibility_p_v` and `voxelize_mask` |
| `voxel_observability` | `VoxelObservability.__init__` | Workspace AABB, voxel size, integrate kwargs |
| `gripper_phase` | `utils/gripper_state.py` GripperPhaseTracker | FSM thresholds (closed/open width, close delta, transition windows) |
| `grasp_owner` | `manipulation/grasp_owner_detector.py` | Three-tier grasp-owner detector |
| `gravity_predict` | `manipulation/gravity_predict.py` | Release-time landing prior parameters |
| `object_dynamics` | `utils/object_dynamics.py` | Per-label restitution, friction, shape primitive |
| `relation` | `relations/relation_orchestrator.py` | Backend selection + EMA filter + trigger schedule |

### Key Bernoulli fields

| Field | Default | Meaning |
|---|---|---|
| `association_mode` | `hungarian` | `hungarian` (production) or `oracle` (GT eval). |
| `p_s` | 1.0 | Survival probability per frame. |
| `p_d` | 0.9 | Detection probability per frame. |
| `lambda_c` | 1.0 | Clutter rate per frame. |
| `lambda_b` | 1.0 | Birth rate per frame. |
| `gate_mode` | `trans` | Gate dimension: `full` (6-DOF), `trans` (translation only), `trans_and_rot`. |
| `cost_d2_mode` | `sum` | Hungarian cost flavour. |
| `enable_huber` | `true` | Whether to apply the adaptive (Huber) robust kernel. |
| `init_cov_from_R` | `false` | Initialize a new track's cov from `R` (less peaked) or from the prior. |
| `birth_fitness_min` | 0.5 | ICP fitness floor below which a new birth is rejected. |
| `birth_rmse_max` | 0.02 | ICP RMSE ceiling (metres) for a birth. |
| `P_min_diag` | `[2.5e-5×3, 2.5e-3×3]` | Per-axis covariance floor (perception-jitter floor). |
| `P_max` | `[0.0625×3, 0.617×3]` | Per-axis covariance ceiling (numerical sanity clamp). |
| `held_birth_radius_m` | 0.25 | Suppress births within this radius of the held object. |
| `r_held_floor` | 0.5 | Existence probability is clamped to this floor while held. |

### Key trigger fields (slow-tier)

```yaml
trigger:
  on_grasp:                false
  on_release:              false
  on_new_object:           false
  periodic_every_n_frames: -1     # -1 disables; positive value: fire every N frames
```

The default is **all-off** — the slow tier is opt-in.  Enable any flag to
let the GTSAM pose-graph smoother run on that event.

### `_extends:` semantics

`_extends:` may chain.  Each child key overlays its parent via deep merge:

* Scalar / list values — child replaces parent.
* Dict values — recursively merged.

The implementation lives in {py:func}`ekf_tracker.configs.load_config`.

## Heuristic tracker — flat per-scenario YAML

`configs/heuristic_tracker/<scenario>.yaml` files are flat (no `_extends:`).
They are read by `scripts/data_demo.py` and `scripts/realtime_app.py` as
plain dicts.

Available scenarios in the repo:

| File | Purpose |
|---|---|
| `demo.yaml` | Generic offline demo. |
| `test.yaml` | Unit-test fixture. |
| `eval.yaml` / `eval_desk2table.yaml` / `eval_move_apple.yaml` | Evaluation runs. |
| `apple_1.yaml` | The `apple_1` benchmark trajectory. |
| `long_demo.yaml`, `new_long_demo.yaml` | Longer scripted scenes. |
| `multi_obj.yaml` | Multi-object scenes. |
| `tomato_in_bowl.yaml` | Specific scenario. |
| `realtime_app.yaml` | The ROS realtime node. |
| `psm.yaml` | Pre-sets for the PSM (surgical) robot. |
| `my_robot_data.yaml` | Template for a new robot. |

### Schema

```yaml
dataset:
  path: "path/to/dataset"

mask:
  method: "json"            # "json" reads cached detections, "color" segments by color
  color_map: {milkbox: [0, 255, 0]}
  tolerance: 80
  detection_dir: "detection_boxes"
  hungarian_dir: "detection_h"
  score_threshold: 0.0
  clear_bottom_ratio: 0.3
  shrink_mask: false
  shrink_kernel_size: 5
  dilate_kernel_size: 19

camera:
  fx: 554.3827
  fy: 554.3827
  cx: 320.5
  cy: 240.5

tsdf:
  bg_voxel_size:     0.03
  object_voxel_size: 0.003
  inpaint:
    enabled:        true
    depth_iters:    5
    kernel_size:    3
    color_radius:   3
    color_method:   telea       # 'telea' or 'ns'
  pose_refine:
    enabled:           true
    min_objects:       2
    sample_step:       2
    voxel_size:        0.001
    distance_thresh:   0.03
    max_iter:          30
    min_points_each:   20
    min_points_total:  100
    min_fitness:       0.2
  pose_threshold:
    max_rotation:    0.01      # rad
    max_translation: 0.01      # m
  mesh_filter:
    enabled:    true
    trim_ratio: 0.1

visualization:
  viewport_size:        [1024, 768]
  use_raymond_lighting: true
```

### Picking the right config

* New robot? Start from `my_robot_data.yaml`, edit `camera.fx/fy/cx/cy`,
  `mask.method`, and the dataset path.
* Tuning TSDF resolution? `tsdf.bg_voxel_size` and `tsdf.object_voxel_size`
  are the only knobs that change accuracy vs memory.  Keep object voxels
  ≤ 5 mm.
* Mask noise on table edges? Increase `mask.clear_bottom_ratio` or set
  `mask.shrink_mask: true`.

## Where the configs flow

```
configs/ekf_tracker/customization.yaml ──┐
                                         │ load_config()
ekf_tracker/configs/default.yaml ────────┘
                                         │
                                         ▼
                            BernoulliConfig + TriggerConfig
                                         │
                                         ▼
                                EkfTracker(K, T_bc, …)

configs/heuristic_tracker/<scenario>.yaml ──→ data_demo.py / realtime_app.py
                                                       │
                                                       ▼
                                          ObjectTracker(K, voxel_size=…, gripper=…)
```

```{seealso}
* [Architecture overview](../architecture/overview.md) — what each
  subsystem does.
* `ekf_tracker/configs/default.yaml` — the canonical default values, with
  inline comments documenting every field.
```
