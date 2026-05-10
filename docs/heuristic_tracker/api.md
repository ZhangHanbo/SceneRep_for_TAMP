# Heuristic tracker — API reference

Hand-written tour of the four public classes.  For a strict autodoc-rendered
reference, see [`api_reference`](api_reference.rst).

---

## `ObjectReconstructor(voxel_size=0.002)`

TSDF-based single-object 3D reconstruction from multi-view observations.
Owns one TSDF volume per object internally and exposes a clean 4-method
surface.

```python
from heuristic_tracker import ObjectReconstructor

recon = ObjectReconstructor(voxel_size=0.002)
oid = recon.create(pose=np.eye(4), label="apple")
ok  = recon.fuse(oid, rgb, depth, K=K, T_cw=T_cw, mask=mask)
mesh = recon.get_mesh(oid)
snap = recon.get_object(oid)            # TrackedObject snapshot
```

**Methods:**

| Method | Returns | Notes |
|---|---|---|
| `create(pose, label, object_id=None)` | `int` | Auto-assign or override the oid. |
| `fuse(object_id, color, depth, K, T_cw, mask=None) -> bool` | `bool` | TSDF integration. **Skip while held**. |
| `get_mesh(object_id) -> Mesh` | `Mesh` | Empty mesh if no fusion has happened. |
| `get_object(object_id) -> TrackedObject` | snapshot | Includes mesh and accumulated points. |

```{warning}
`fuse()` accumulates geometry in the object's *initial* pose frame, not the
current one.  The transform is warped internally:
``T_cw_fix = (pose_init @ inv(pose_cur)) @ T_cw``.  If the object is moving
(pose_cur ≠ pose_init), call sites **must** suppress fusion during
GRASPING / HOLDING / RELEASING — otherwise blurred reconstructions result.
```

---

## `ObjectTracker(K, voxel_size=0.002, gripper=None)`

Tracks object identities across frames using detection matching.  Internally
delegates to `heuristic_tracker.id_associator.associate_by_id`.

```python
from heuristic_tracker import ObjectTracker, FrameDetections
from utils.robot_models import create_gripper_geometry

tracker = ObjectTracker(
    K=K, voxel_size=0.002,
    gripper=create_gripper_geometry("fetch"),
)

objs = tracker.update(detections, rgb, depth, T_cw)
held = tracker.detect_held_object(T_ew, joint_state=joints)
if held is not None:
    tracker.set_held_object(held)
# … later …
tracker.release_object(held)
```

**Methods:**

| Method | What it does |
|---|---|
| `update(detections, rgb, depth, T_cw, integrate=True)` | Run association + (optional) TSDF fusion; return updated `List[TrackedObject]`. |
| `detect_held_object(T_ew, box_size=(0.07,0.05,0.05), joint_state=None)` | Returns the oid of the object inside the gripper jaws, or `None`.  Uses URDF-derived volume when `gripper` is set. |
| `set_held_object(obj_id)` | Mark `obj_id` as held.  Suppresses association + fusion until released. |
| `release_object(obj_id)` | Clear the held flag; pose stays uncertain until ICP re-localizes. |
| `internal_objects` (property) | Live list of internal `SceneObject` instances — for `PoseUpdater` and `RelationAnalyzer`. |

---

## `PoseUpdater`

Static methods that update object poses during manipulation.  Operates on
the internal `SceneObject` list returned by `tracker.internal_objects`.

```python
from heuristic_tracker import PoseUpdater

# While held: track via end-effector kinematics.
ok = PoseUpdater.update_from_ee(
    tracker.internal_objects, obj_id=held, T_cw=T_cw, T_ec=T_ec,
)

# After release: refine via ICP against the new observation.
new_pose = PoseUpdater.update_from_icp(
    tracker.internal_objects, obj_id=released,
    new_points=world_pts, max_correspondence_distance=0.02,
)
```

| Method | Returns |
|---|---|
| `update_from_ee(objects, obj_id, T_cw, T_ec)` | `bool` |
| `update_from_icp(objects, obj_id, new_points, max_correspondence_distance=0.02)` | `Optional[np.ndarray]` (4×4 pose) |

---

## `RelationAnalyzer`

Static method that computes geometric spatial relations between objects.

```python
from heuristic_tracker import RelationAnalyzer

graph = RelationAnalyzer.compute(tracker.internal_objects, tolerance=0.02)
# graph.relations: {obj_id: {"on": [...], "in": [...], "under": [...], "contain": [...]}}
```

`tolerance` is a spatial slack (metres) used by the underlying
`object_relation_graph.compute_spatial_relations` to absorb noisy point-cloud
boundaries.

---

## Configuration

Heuristic tracker parameters live in `configs/heuristic_tracker/<scenario>.yaml`
(no inheritance, flat schema).  See [Configs reference](../reference/configs.md).

```{seealso}
* [Heuristic tracker overview](index.md)
* [Auto-generated reference](api_reference.rst)
* [`scripts/examples/heuristic_offline.py`](../reference/examples.md) — minimal end-to-end recipe.
```
