# What We Can Borrow from Khronos

Khronos (Schmid et al., RSS 2024, Outstanding Systems Paper) solves a harder problem — 4D spatio-temporal mapping with unknown dynamic objects. SceneRep solves a narrower problem — scene tracking during *known* robot manipulation. But several of Khronos's design patterns directly address weaknesses in our system.

---

## 1. The Three-Speed Architecture

**Khronos:** Active Window (22 FPS, O(1)) → Global Optimization (seconds, at loop closures) → Reconciliation (minutes, after optimization).

**SceneRep currently:** Everything runs at the same speed — detection, tracking, ICP refinement, scene graph, all per-frame. This is wasteful: camera pose refinement ICP (~50ms) runs every frame even when the camera barely moved, while scene graph computation (~1ms) also runs every frame even though relations rarely change.

**What to borrow:** Separate into two speeds:

| Process | Frequency | What it does |
|---------|-----------|-------------|
| **Fast** (every frame) | ~30 FPS | Gripper state machine, EE propagation for held objects, TSDF fusion for visible static objects. This is already O(1) per frame in our code. |
| **Slow** (triggered) | On events | Factor graph optimization + scene graph recomputation. Trigger conditions: (a) grasping onset, (b) release, (c) camera has moved >5cm since last optimization, (d) new object detected. |

This is almost free to implement — we already have the fast path (the main loop in `data_demo.py`). The slow path is the factor graph from Improvement 2. We just need to not call it every frame, only on trigger events. Estimated change: ~10 lines (add trigger conditions around the `optimize_frame()` call).

---

## 2. Fragment-as-Hypothesis

**Khronos:** Objects are not detected once — they accumulate evidence as Tracks/Fragments. Each fragment has temporal bounds (`first_seen`, `last_seen`), a confidence, and a set of observations. Fragments are the atomic units; the hard question "are these two fragments the same object?" is deferred to global optimization.

**SceneRep currently:** `associate_by_id()` immediately commits — a new detection either matches an existing object (by label/ID) or creates a new object. There's no "maybe this is a new object, maybe it's the old one seen from a different angle" hypothesis. Once an object is created, it's permanent. Once a label is assigned, it can only be refined, not split.

**What to borrow:** Add a lightweight hypothesis layer between detection and committed objects:

```python
@dataclass
class ObjectHypothesis:
    observations: List[Observation]  # (timestamp, mask, score, T_cw)
    bbox_3d: np.ndarray              # accumulated bounding box
    label_belief: Dict[str, Tuple[float, float]]  # from Improvement 1
    confidence: float                # accumulated evidence
    first_seen: float
    last_seen: float
```

New detections create or extend hypotheses. A hypothesis is promoted to a committed `SceneObject` only when `confidence > threshold` (e.g., seen in >3 frames with mean score >0.3). This prevents the current problem of noisy single-frame detections creating phantom objects that then pollute the scene graph and factor graph.

**Estimated change:** ~60 lines — a new `ObjectHypothesisManager` class that sits between detection and `associate_by_id()`. The existing `associate_by_id()` receives only promoted hypotheses.

---

## 3. Evidence of Absence vs. Absence of Evidence

**Khronos:** The Ray Verificator stores observation rays and later checks whether they pass through empty space (= object was removed) or are occluded (= can't tell). This is the geometric distinction between "I looked there and it's gone" vs. "I couldn't see."

**SceneRep currently:** After release, an object's `pose_uncertain` is set to True. It stays uncertain until the next successful ICP re-localization. But the system has no way to tell whether the object is (a) still there but unobserved, (b) moved by the robot to a known location, or (c) gone entirely (e.g., placed in a closed container). All three cases look the same: `pose_uncertain = True`.

**What to borrow for manipulation:** We don't need the full ray verificator (Khronos's is designed for long-term autonomous mapping). But we can borrow the core idea cheaply:

After releasing an object, check whether the object's expected location (from EE-propagated pose) is visible in the current depth image:

```python
def check_object_visible(obj, depth, K, T_cw):
    """Check if the object's expected position is visible or occluded."""
    # Project object center into image
    p_cam = np.linalg.inv(T_cw) @ np.append(obj.pose_cur[:3, 3], 1)
    if p_cam[2] <= 0:
        return "not_in_view"  # behind camera
    
    u = int(K[0,0] * p_cam[0] / p_cam[2] + K[0,2])
    v = int(K[1,1] * p_cam[1] / p_cam[2] + K[1,2])
    
    if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
        return "not_in_view"  # outside image
    
    measured_depth = depth[v, u]
    expected_depth = p_cam[2]
    
    if measured_depth < 0.1:
        return "no_depth"  # no measurement
    elif measured_depth < expected_depth - 0.05:
        return "occluded"  # something is in front → absence of evidence
    elif measured_depth > expected_depth + 0.05:
        return "absent"    # ray passes through → evidence of absence
    else:
        return "present"   # depth matches → object is there
```

This is ~20 lines and gives us three distinct outcomes instead of one boolean. In the factor graph, these map to different actions:
- `present` → run ICP, add observation factor with normal noise
- `occluded` → keep prior, don't update (absence of evidence)
- `absent` → the object moved; inflate covariance dramatically or mark for re-detection
- `not_in_view` → keep prior (absence of evidence)

---

## 4. What NOT to Borrow

**Khronos's motion detection (free-space check):** This detects *unknown* moving objects by checking if new points fall in previously-free voxels. SceneRep doesn't need this — we know exactly which object is moving (the one in the gripper). Our state machine is simpler and more robust for manipulation.

**Khronos's full deformation graph:** The mesh deformation at loop closure (background control points + fragment poses) is for correcting large-scale drift over minutes of exploration. SceneRep operates over shorter trajectories (typically 1-5 minutes of manipulation) where AMCL drift is bounded. The ICP camera refinement + factor graph is sufficient.

**Khronos's reconciliation at the minute-to-hour timescale:** The temporal reasoning about when objects appeared/disappeared is designed for autonomous patrol scenarios. For manipulation, we know exactly when objects move because we moved them. The gripper state machine handles this.

**Khronos's C++ complexity:** The multi-threaded architecture with spatial hashing, voxel block grids, and lock-free buffers is for real-time performance on a CPU. Our Python system uses Open3D's TSDF which already handles this internally. We don't need to reimplement it.

---

## Summary: Minimal Changes to Borrow

| Idea | Lines to add | Where | Impact |
|------|-------------|-------|--------|
| Two-speed trigger | ~10 | `data_demo.py` | Factor graph runs only when needed, not every frame |
| Hypothesis layer | ~60 | New `hypothesis_manager.py` | Prevents phantom objects from noisy single-frame detections |
| Visibility check | ~20 | `pose_update/visibility.py` | Distinguishes "gone" from "can't see" after release |

Total: ~90 lines. No new dependencies. All three are independent and can be added incrementally.

The biggest lesson from Khronos is not any single technique — it's the **factorization discipline**: decompose the problem by timescale, make each process only solve what's appropriate to its frequency, and defer hard decisions (fragment association, change detection) to slower processes that have more information. Our Improvement 1 (EKF) handles the fast timescale, and Improvement 2 (factor graph) handles the slow one. The three additions above fill the gaps between them.
