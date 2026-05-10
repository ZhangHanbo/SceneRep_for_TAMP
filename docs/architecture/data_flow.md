# Data flow

Three flow diagrams: per-frame inside `step()`, per-trajectory offline, and
the rosbag → tracker → eval round-trip.

## Per-frame: inside `EkfTracker.step()`

```
detections, rgb, depth, slam_pose=T_wb, T_bc, T_bg, gripper_width, joints
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │  Gripper phase FSM                  │  utils/gripper_state.py
                │  → idle / grasping / holding / …    │
                └─────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │  Process noise schedule             │  Q from gripper phase
                │  → Q for predict step               │  utils/ekf_se3.py
                └─────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │  EKF predict (per object, base frame)│
                │  + rigid-attach for held object     │
                │  + gravity prior on release         │  manipulation/gravity_predict
                └─────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │  Detection dedup & birth gate       │  perception/det_dedup
                │                                     │  perception/birth_gating
                └─────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │  Hungarian association              │  perception/association
                │  + visibility & class feasibility   │
                └─────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │  ICP pose update + Bernoulli r update│
                │  + observation chain append         │  state/obs_chain
                └─────────────────────────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │  Birth admissions / track prune     │  birth_gate.py
                │  Relation maybe_update              │  relations/relation_orchestrator
                │  Slow-tier trigger?                 │  TriggerConfig
                └─────────────────────────────────────┘
                                   │
                                   ▼
                          SceneView { objects, relations }
                          (compose T_wo = T_wb · T_bo at boundary)
```

The heuristic tracker follows the same skeleton with TSDF fusion and label-
based association in place of the Bernoulli-EKF + Hungarian.  See
[`ObjectTracker.step`](../heuristic_tracker/api.md).

## Per-trajectory: offline pipeline

```
dataset/                          configs/<tracker>/<scenario>.yaml
└── rgb/, depth/, pose_txt/                  │
                  │                          │
                  ▼                          ▼
             scripts/data_demo.py  ───→  EkfTracker / ObjectTracker
                  │                          │
                  ▼                          ▼
        per-frame poses, masks,       SceneView per frame
        TSDF fusion (heuristic)              │
                  │                          ▼
                  ▼                  ekf_state/<frame>.json
              eval/<traj>/                  (EKF only)
                  │                          │
                  ▼                          ▼
             eval/eval_all.py     scripts/visualize_ekf_tracking.py
                  │                          │
                  ▼                          ▼
        ADD / ADD-S CSV + tables         ekf_debug.mp4
```

Walkthrough: [Offline pipeline](../workflows/offline_pipeline.md) and
[Evaluation](../workflows/evaluation.md).

## End-to-end: rosbag → tracker → eval

```
*.bag
   │
   ▼
scripts/rosbag2dataset/run_pipeline.sh
   │   ├── extract_bag_local.py     (rgb, depth, joints, tf, amcl @ 5 Hz)
   │   ├── icp_amcl.py              (refine T_wb against depth)
   │   ├── sam2/ + owl/ scripts     (per-frame OWLv2 + SAM2 detections)
   │   └── track_object_ids.py      (stable cross-frame oid)
   ▼
datasets/<traj>/
   │
   ▼
scripts/data_demo.py  ──or──  scripts/examples/track_apple_in_the_tray.py
   │
   ▼
SceneView stream  +  ekf_state/  +  ekf_debug.mp4
   │
   ▼
eval/eval_all.py → CSV → eval/generate_*_table.py → tables / plots
```

Walkthroughs: [rosbag → dataset](../workflows/rosbag_to_dataset.md),
[Live detection](../workflows/live_detection.md),
[Evaluation](../workflows/evaluation.md).

## Configuration plumbing

```
configs/ekf_tracker/customization.yaml
              │ _extends:
              ▼
ekf_tracker/configs/default.yaml
              │
              ▼
ekf_tracker.configs.load_config()    →  dict
ekf_tracker.configs.to_bernoulli_config(...)  → BernoulliConfig
ekf_tracker.configs.to_trigger_config(...)    → TriggerConfig
              │
              ▼
EkfTracker(K=..., T_bc=..., bernoulli_cfg=..., trigger=...)
```

For the heuristic tracker, the corresponding files are
`configs/heuristic_tracker/<scenario>.yaml`; they are flat (no `_extends:`).

```{seealso}
* [Configs](../reference/configs.md) — full schema.
* [Architecture overview](overview.md) — why the layers and tiers exist.
```
