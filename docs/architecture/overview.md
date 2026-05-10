# Architecture overview

Dynamic Scene Graph is built around a **strict two-layer hierarchy** with an
**event-triggered slow tier**:

```
┌────────────────────────────────────────────────────────────────────┐
│  Layer 1 — SLAM                                                    │
│  Inputs:  RGB, depth (movables masked out), IMU/wheel odom         │
│  Owns:    T_wb (world-from-base) and Σ_wb                           │
│  Source:  external (e.g. AMCL, ORB-SLAM3); shimmed by              │
│           utils/slam_interface.py::PassThroughSlam for tests       │
└────────────────────────────────────────────────────────────────────┘
                          │ T_wb every frame
                          ▼
┌────────────────────────────────────────────────────────────────────┐
│  Layer 2 — Object tracking (the trackers in this repo)             │
│  Owns:    per-object pose, geometry, existence, relations           │
│  Inputs:  RGB-D, T_wb, gripper width + joints + T_bg                │
│  Output:  SceneView { objects, relations }                          │
│                                                                    │
│  Two-tier inside Layer 2 (EKF only):                                │
│    Fast tier:  per-frame predict / Hungarian / EKF update / birth  │
│                / prune in the BASE frame                            │
│    Slow tier:  GTSAM pose graph over many frames + relation         │
│                priors, triggered on grasp / release / new object   │
│                / every N frames                                    │
└────────────────────────────────────────────────────────────────────┘
```

## Why two layers, not one big SLAM

SLAM fuses depth into a single rigid map. If you let it ingest pixels from
moving objects, those pixels poison the map and pull the camera estimate.
The fix is contractually simple:

> **Movable-object pixels are masked out of the depth before it reaches
> SLAM.** Layer 2 owns those pixels.

That single rule is why this repo has two layers, why the heuristic and EKF
trackers can share one SLAM backend, and why the masking protocol in
`utils/slam_interface.py` is non-negotiable.  See
[`ekf_tracker/DISCUSSION.md`](../ekf_tracker/DISCUSSION.md) for the longer
form.

## Why two tiers inside Layer 2

A per-frame Bayes filter is fast but myopic — it can't undo a bad birth ten
frames later, and it can't apply a "the apple is in the tray" relation.
A pose graph is global but expensive.  Splitting the cost:

| Tier | Frequency | Optimizes | Implementation |
|---|---|---|---|
| Fast | Every frame | Per-object Bernoulli + Gaussian on SE(3) | `ekf_tracker.gaussian_ekf_tracker.GaussianEkfTracker` |
| Slow | On grasp/release/new-object/every-N frames | Joint pose graph over movable objects + relation factors | `ekf_tracker.factor_graph.PoseGraphOptimizer` (GTSAM) |

The {py:class}`ekf_tracker.config.TriggerConfig` dataclass captures the slow-tier
schedule.  Set every `on_*` flag to `false` and `periodic_every_n_frames=0`
to disable the slow tier entirely; the fast tier is fully self-contained.

## The three trackers, mapped onto the layered model

| Tracker | Layer-1 producer | Layer-2 fast tier | Layer-2 slow tier |
|---|---|---|---|
| `heuristic_tracker` | external SLAM (or pass-through) | `ObjectTracker.step` (TSDF + Hungarian + ICP) | n/a |
| `ekf_tracker` | external SLAM (or pass-through) | Bernoulli-EKF (`GaussianEkfTracker`) | GTSAM pose graph (`PoseGraphOptimizer`), optional |
| `baselines.VisualOnlyTracker` | n/a (consumes `T_cw` directly) | Direct ICP composition | n/a |

The EKF tracker's facade {py:class}`ekf_tracker.api.EkfTracker` composes:

* {py:class}`ekf_tracker.gaussian_ekf_tracker.GaussianEkfTracker` (fast tier)
* {py:class}`ekf_tracker.orchestrator_gaussian.TwoTierOrchestratorGaussian` (slow tier wrapper)
* {py:class}`ekf_tracker.perception_pipeline.LiveDetectionPipeline` (OWLv2 + SAM2 streaming)
* `ekf_tracker.manipulation` (gripper FSM, grasp owner, gravity predict)
* `ekf_tracker.relations.RelationOrchestrator` (event-triggered, EMA-smoothed)
* `perception.birth_gating`, `perception.icp_pose`, `perception.visibility` (shared with the heuristic tracker)

## Frames

Five rigid frames matter:

| Symbol | Meaning |
|---|---|
| `W` | World (SLAM map origin) |
| `B` | Robot base |
| `C` | Camera (RGB-D) |
| `G` | Gripper (held-object frame after grasp) |
| `O_i` | Object `i` |

Per-frame inputs to {py:meth}`ekf_tracker.api.EkfTracker.step`:
`slam_pose = T_wb`, `T_bc` (extrinsic, usually static), `T_bg` (FK of the
gripper from base), `gripper_width`, `joints`.  See
[Frame conventions](frame_conventions.md) for sign conventions and the
exact transformation chain.

## Key design decisions

* **Base-frame fusion.** EKF state lives in the robot's base frame, not the
  world frame.  `Σ_wb` from SLAM never enters the recursion; it's composed in
  only at output time.  This avoids inflating object covariance every time
  the base moves.  Detail: [`cov_anisotropy_explained`](../ekf_tracker/cov_anisotropy_explained.md).

* **Bernoulli existence per track.** Every object carries a scalar `r ∈
  [0,1]`.  Births raise it, misses decay it, association evidence updates it.
  Below threshold the track is pruned.  Removes the need for a hard
  death/birth heuristic and gives the slow tier a probabilistic input.

* **Append-only observation chains.** Camera-frame ICP results are stored as
  an append-only chain (`ekf_tracker.state.obs_chain`) so a retroactive SLAM
  correction can be re-projected to world without losing object information.

* **Strict config contract.** Two YAML files (`default.yaml` and your
  `customization.yaml` extending it) hold every parameter.  Missing keys
  hard-error at load time.  See [Configs](../reference/configs.md).

```{seealso}
* [Frame conventions](frame_conventions.md) — exact transform chain.
* [Data flow](data_flow.md) — per-frame and per-trajectory paths through
  the system.
* [`ekf_tracker/DISCUSSION.md`](../ekf_tracker/DISCUSSION.md) — design
  rationale for the layering.
* [`ekf_tracker/improvements.md`](../ekf_tracker/improvements.md) — how the
  per-object EKF and pose graph were retrofitted onto the heuristic
  pipeline.
* [`ekf_tracker/latex/bernoulli_ekf.pdf`](../ekf_tracker/latex/bernoulli_ekf.pdf)
  — full derivation; Part III is the algorithm-to-code map.
```
