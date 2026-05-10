# Glossary

Quick-reference for symbols and terms used across the docs.

## Frames

| Symbol | Frame |
|---|---|
| `W` | World (SLAM map origin). |
| `B` | Robot base. |
| `C` | Camera (RGB-D). |
| `G` | Gripper TCP / held-object anchor. |
| `O_i` | Object `i` body frame. |

`T_AB` reads "A from B": `p_A = T_AB · p_B`.  See [Frame conventions](../architecture/frame_conventions.md).

## Variables

| Symbol | Meaning |
|---|---|
| `T_wb` | World-from-base SLAM pose; input to every `step()`. |
| `T_bc` | Camera-from-base extrinsic; usually static. |
| `T_bg` | Gripper-from-base FK; changes every frame. |
| `T_wo` | World-from-object pose; what `EkfObject.pose` and `TrackedObject.pose` carry. |
| `T_co` | Camera-from-object pose; intermediate, reconstructed from the others. |
| `Σ_wb` | SLAM covariance — *not* fed into the EKF recursion (that's the point of base-frame fusion). |
| `P_bo` | Per-object 6×6 covariance in the base frame; `EkfObject.cov`. |
| `r` | Bernoulli existence probability ∈ `[0, 1]`. |
| `Q` | Process noise (per gripper-phase schedule). |
| `R` | Measurement noise; built from ICP fitness/RMSE. |

## Concepts

**Bernoulli existence model.** Each track carries a scalar probability that
the object exists.  Births raise it, misses decay it, association evidence
updates it.  Below threshold the track is pruned.  References: Mahler 2014,
Vo & Vo 2009.

**Two-tier orchestration.** Layer-2 tracking is split into a fast tier (per
frame, per object) and a slow tier (event-triggered global pose graph).
The fast tier is a Bernoulli-EKF on SE(3); the slow tier is a GTSAM factor
graph that bundles object priors with relation factors.

**Fast tier.** Per-frame predict → associate → update → birth → prune cycle
inside the robot's base frame.  The `InstrumentedTracker` in
`ekf_tracker/gaussian_ekf_tracker.py` is the canonical implementation.

**Slow tier.** GTSAM `PoseGraphOptimizer` over many frames; runs on grasp /
release / new-object / every-N-frames triggers.  Optional — the fast tier
is fully self-contained without it.

**Base-frame fusion.** EKF state lives in the base frame `B`, not the world
frame `W`.  `Σ_wb` from SLAM never enters the recursion; world poses are
composed in only at output.  See [`cov_anisotropy_explained`](../ekf_tracker/cov_anisotropy_explained.md).

**Observation chain.** Append-only buffer of camera-frame ICP results
(`ekf_tracker.state.obs_chain`).  Lets a retroactive SLAM correction be
re-projected to world without losing object information.

**Detection dict.**  Per-object dict produced by OWLv2 + SAM2:

```python
{
    "id":         17,                  # SAM2 tracklet id (stable across frames)
    "object_id":  17,                  # alias of id
    "label":      "apple",
    "score":      0.94,                # this frame's OWL score
    "mean_score": 0.91,                # rolling mean over n_obs frames
    "n_obs":      12,
    "box":        [x1, y1, x2, y2],
    "mask":       "<base64 PNG, decoded to (H,W) {0,1} uint8>",
}
```

See [Detection pipeline](../perception/detection_pipeline.md).

**SceneView.** Per-frame output of {py:meth}`ekf_tracker.api.EkfTracker.step`:
`{objects: Dict[int, EkfObject], relations: List[Dict]}`.

**ICP backends.** Three pose-estimation methods in `perception/icp_pose.py`:

| Backend | What it does |
|---|---|
| `centroid` | Translation-only update from the masked centroid. |
| `icp_chain` | Continuous tracking: ICP with persistent reference cloud, append to obs chain. |
| `icp_anchor` | State-free anchor: ICP from current observation against the object's accumulated points. |

**ADD / ADD-S.** Standard 6-DOF pose error metrics.  ADD averages closest-
point distance over the model surface; ADD-S is the symmetric version
needed for axis-symmetric objects.  Used in [Evaluation](../workflows/evaluation.md).

**`_extends:`.** YAML deep-merge inheritance; child values overlay parent
values via deep merge.  Implemented in
{py:func}`ekf_tracker.configs.load_config`.

**Dead key.** A YAML field with no runtime reader.  The repo aggressively
prunes these from `customization.yaml`; see the comment block at the top
of `configs/ekf_tracker/customization.yaml` for the canonical list.

**Parity test.** Tests under `tests/integration/test_parity_*.py` that
re-run the EKF tracker on cached data and check that pose error vs the
stored JSON dumps is *exactly* zero.  Pass criterion:
`pose_max_err_world == 0.0` and `r_max_err == 0.0`.

**Held object.** The object currently grasped by the gripper, as identified
by either:

* Geometric containment (gripper inside-jaws AABB intersects object points).
* Perception override (a SAM2 tracklet labeled "held" in the scene graph).
* Fallback heuristics (last-seen near-gripper).

The three signals are combined by
{py:class}`ekf_tracker.manipulation.grasp_owner_detector.GraspOwnerDetector`.
While held, the object's pose is rigidly attached to the gripper's FK
(`T_wo = T_wg · T_go_at_grasp`).
