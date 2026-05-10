# Perception layer

Shared perception primitives consumed by every tracker.  All of these are
small, focused modules with no global state — pure functions or thin classes.

```python
from perception import (
    hungarian_associate, oracle_associate, AssociationResult,
    PoseEstimator, centroid_cam_from_mask,
    visibility_p_v,
)
```

```{toctree}
:caption: Perception
:maxdepth: 1

detection_pipeline
api_reference
```

## What's in here

| Module | Role |
|---|---|
| `perception.association` | Hungarian data association with Mahalanobis cost, per-class feasibility, optional SAM2 tracklet bonuses, plus an `oracle` mode for GT eval. |
| `perception.icp_pose` | Three pose-estimation backends — `centroid`, `icp_chain`, `icp_anchor` — with shared mask cleanup and ICP gates. |
| `perception.visibility` | `visibility_p_v(...)` predicate via depth-image z-buffer ray tracing — projects object surface samples to image space, tests against actual depth. |
| `perception.birth_gating` | Rejects duplicate births when a centroid lies near a same-label live track or near the held-object's location. |
| `perception.det_dedup` | Pre-Hungarian dedup of overlapping sub-part detections via voxelization + set intersection.  Absorbs label histories. |
| `perception.voxel_observability` | Dense 3D voxel grid (UNSEEN/EMPTY/OCCUPIED) for gravity-drop predictions. Updated from depth rays per frame. |
| `perception.adaptive_kernel` | Generalized Barron robust-loss kernel with truncated partition function and IRLS weights. Drives the EKF's Mahalanobis update. |
| `perception.camera_pose_refiner` | Refines camera pose from masks of known objects; back-projects points and minimizes reprojection error. |

## Detection sub-stack

The detection pipeline (OWLv2 + SAM2) lives under
{py:mod}`perception.detection` (clients) and `perception.det_pipeline`
(server-side OWL/SAM2 implementations).  See [Detection pipeline](detection_pipeline.md)
for usage; see [Live detection workflow](../workflows/live_detection.md) for
a full setup walkthrough.

## Who consumes what

* {py:class}`heuristic_tracker.api.ObjectTracker` consumes
  `id_associator.associate_by_id` (label-based; *not* the Hungarian flavour
  in `perception.association`).
* {py:class}`ekf_tracker.api.EkfTracker` consumes Hungarian association,
  birth gating, ICP pose, visibility, voxel observability, and adaptive
  kernels.
* `baselines.visual_only_tracker` reuses ICP back-projection helpers
  (`_back_project`, `_voxelize`) from `perception.icp_pose`.

```{seealso}
* [Detection pipeline](detection_pipeline.md) — `detect_objects_on_image` API
* [API reference](api_reference.rst) — auto-generated module docs
```
