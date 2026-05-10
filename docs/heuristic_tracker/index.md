# Heuristic tracker

Deterministic TSDF + Hungarian + ICP scene tracker.  No probabilistic state,
no covariance, but you get a reconstructed mesh per object and the same
spatial-relation API as the EKF.  This is the production path used by
`robi_butler`.

```python
from heuristic_tracker import (
    ObjectReconstructor, ObjectTracker, PoseUpdater, RelationAnalyzer,
    Mesh, TrackedObject, FrameDetections, RelationGraph,
)
```

```{toctree}
:caption: Heuristic tracker
:maxdepth: 1

api
api_reference
```

## Four classes you actually use

| Class | What it does |
|---|---|
| {py:class}`heuristic_tracker.api.ObjectReconstructor` | Per-object TSDF reconstructor.  Create an object, fuse RGB-D frames, extract a mesh. |
| {py:class}`heuristic_tracker.api.ObjectTracker` | Track object identities across frames; matches detections to existing objects, creates new ones, fuses TSDF. |
| {py:class}`heuristic_tracker.api.PoseUpdater` | Updates object poses while held (from end-effector transform) or after release (ICP). |
| {py:class}`heuristic_tracker.api.RelationAnalyzer` | Computes spatial relations (`on`, `in`, `under`, `contain`) from world-frame point clouds. |

## End-to-end shape

```python
import numpy as np
from heuristic_tracker import ObjectTracker, FrameDetections, RelationAnalyzer
from utils.robot_models import create_gripper_geometry

K       = np.array([[554.38, 0, 320.5], [0, 554.38, 240.5], [0, 0, 1]])
gripper = create_gripper_geometry("fetch")
tracker = ObjectTracker(K=K, voxel_size=0.002, gripper=gripper)

for rgb, depth, T_cw, dets, joints, T_ew in stream:
    objs = tracker.update(
        FrameDetections(labels=dets["labels"],
                        scores=dets["scores"],
                        masks=dets["masks"],
                        bboxes=dets["bboxes"]),
        rgb, depth, T_cw,
    )
    held = tracker.detect_held_object(T_ew, joint_state=joints)
    if held is not None:
        tracker.set_held_object(held)

    relations = RelationAnalyzer.compute(tracker.internal_objects)

    for o in objs:
        print(o.id, o.label, o.pose[:3, 3])
```

## Public dataclasses

```python
@dataclass
class Mesh:
    vertices: np.ndarray        # (V, 3)
    faces:    np.ndarray        # (F, 3)
    normals:  np.ndarray        # (V, 3)
    colors:   np.ndarray        # (V, 3)

@dataclass
class TrackedObject:
    id:     int
    label:  str
    pose:   np.ndarray          # 4×4 world-frame T_wo
    points: np.ndarray          # (N, 3) accumulated world points
    mesh:   Optional[Mesh] = None

@dataclass
class FrameDetections:
    labels:  List[str]
    scores:  np.ndarray         # (N,)
    masks:   List[np.ndarray]   # list of (H, W) bool arrays
    bboxes:  np.ndarray         # (N, 4) pixel coords [x1, y1, x2, y2]

@dataclass
class RelationGraph:
    relations: Dict[int, Dict[str, List[int]]]
    # {obj_id: {"on": [other_ids], "in": [...], "under": [...], "contain": [...]}}
```

## Key design rules

* **TSDF fusion uses the *initial* pose frame, not the current one.**
  `ObjectReconstructor.fuse` warps the camera transform so that observations
  taken after the object has moved still register against the original TSDF
  volume.  Call sites must skip fusion during grasping / holding / releasing
  states — see the docstring on
  {py:meth}`heuristic_tracker.api.ObjectReconstructor.fuse`.

* **Held objects skip association.**  `ObjectTracker.set_held_object(oid)`
  marks an object as `pose_uncertain=True`; it stays there until a successful
  ICP re-localizes it (`object_pose_updater.icp_reappear`).

* **Grasp containment uses the URDF-derived gripper volume.**  Pass the
  appropriate `GripperGeometry` (e.g. `create_gripper_geometry("fetch")`) to
  `ObjectTracker.__init__` so `detect_held_object` uses the inside-jaws AABB
  rather than an ad-hoc EE-centered box.

```{seealso}
* [API walkthrough](api.md)
* [API reference](api_reference.rst)
* [Architecture overview](../architecture/overview.md)
* [Configs reference](../reference/configs.md)
```
