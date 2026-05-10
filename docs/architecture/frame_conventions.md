# Frame conventions

Every transform in the codebase follows the **`T_AB` reads "A from B"**
convention.  A point `p_B` in frame `B` is mapped to frame `A` as
`p_A = T_AB · p_B`.  Composition reads left-to-right:
`T_AC = T_AB · T_BC`.

## Frames

| Symbol | Description | Where it comes from |
|---|---|---|
| `W` | World / map origin | SLAM (Layer 1) |
| `B` | Robot base | Layer 1 → Layer 2 boundary |
| `C` | Camera (RGB-D) | Static URDF extrinsic, occasionally re-calibrated |
| `G` | Gripper TCP / held-object anchor | URDF FK |
| `O_i` | Object `i` body frame | Layer 2 (EKF or heuristic) |

## Transforms used in `step()`

Per-frame inputs to {py:meth}`ekf_tracker.api.EkfTracker.step` and
{py:class}`heuristic_tracker.api.ObjectTracker`:

| Argument | Meaning | Typical shape |
|---|---|---|
| `slam_pose` | `T_wb`, world-from-base | `(4, 4)` |
| `T_bc`      | camera-from-base extrinsic | `(4, 4)` |
| `T_bg`      | gripper-from-base FK | `(4, 4)` |
| `joints`    | dict of joint angle (rad), keys depend on URDF | `Dict[str, float]` |
| `gripper_width` | finger gap in metres | `float` |

The composed camera-from-world used internally is

```
T_cw = T_cb · T_bw  =  inv(T_bc) · inv(T_wb)
```

You don't construct `T_cw` yourself; the trackers do it.

## Object pose

Every public `EkfObject.pose` and `TrackedObject.pose` is **world-from-object**:
`T_wo`.  To project the object's body-frame point cloud `pts_o` into the
camera image you compose:

```python
T_co = inv(T_bc) @ inv(T_wb) @ T_wo
pts_c = (T_co[:3, :3] @ pts_o.T + T_co[:3, 3:4]).T
uv    = (K @ (pts_c / pts_c[:, 2:3]).T).T[:, :2]
```

## Why the EKF lives in the base frame

The fast tier stores `T_bo`, not `T_wo`.  The world pose is computed at output
time as `T_wo = T_wb · T_bo`.  This matters because the SLAM covariance
`Σ_wb` is not zero — composing it into the per-frame recursion would inflate
every object's covariance every time the base moves, even if the object is
perfectly visible.  Detail: [`cov_anisotropy_explained`](../ekf_tracker/cov_anisotropy_explained.md).

## Sign conventions for the 6×6 covariance

The EKF tangent state is ordered `[v, ω]` — translation first, rotation
second — both expressed in the **base frame at the linearization point**.
Updates use the SE(3) right-trivialization (Joseph form for symmetry).
The mathematics is in `utils/ekf_se3.py` and Part I of
`ekf_tracker/latex/bernoulli_ekf.pdf`.

## Common gotchas

* **`T_bc` vs `T_cb`.** Most config files store the camera-to-base
  *extrinsic*, which is `T_bc` (base-from-camera). The trackers expect
  `T_bc` directly.

* **`T_bg` is FK output.** It changes every frame as the arm moves, so
  cache the URDF-loader result (`utils/fetch_arm_fk.py`) but recompute the
  transform per frame from joint angles.

* **`gripper_width` is finger gap.** Half-width or single-finger displacement
  will misfire the gripper FSM; see `utils/gripper_state.py`.

* **`joints` is a dict, not a list.** Keys must match the URDF.  Missing keys
  default to zero, which silently breaks grasp-owner geometric containment.

```{seealso}
[Architecture overview](overview.md) for why the frames are split this way.
```
