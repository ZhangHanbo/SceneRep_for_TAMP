# Baselines

Visual-only ICP tracker — **no filter**, **no proprioception**.  Use only as
an ablation reference; not part of the production tracking surface.

```python
from baselines import VisualOnlyTracker

tracker = VisualOnlyTracker(K=K, mode="last_frame")
T_wo, accepted, fitness, rmse = tracker.update(oid, mask, depth, T_cw)
```

```{toctree}
:caption: Baselines
:maxdepth: 1

api_reference
```

## Two reference-update policies

| `mode` | Behaviour | Drift | Failure mode |
|---|---|---|---|
| `"first_frame"` | Reference cloud captured on first sighting and never updated. | Drift-free for static objects near initial viewpoint. | As the camera moves, ICP fitness collapses → updates rejected → world pose freezes at last accepted `T_wo`. |
| `"last_frame"` | Reference replaced on every accepted frame. | Compounds across frames. | Tolerant of long camera motion; tracks moved objects naturally because the reference follows them. |

Both modes share:

* Voxel size 5 mm, ICP correspondence radius 2 cm, max 30 iterations.
* Acceptance gates: fitness ≥ 0.90, RMSE ≤ 15 mm.
* On rejection: cached `T_wo` is returned unchanged (constant-velocity-zero
  hold).

## Composition

```
T_wo(t) = T_cw(t) · T_co(t)
T_co(t) = ICP(reference_cloud → current_cloud, init = (I, centroid_now))
```

That's it — no filter, no Σ, no gripper FSM, no relations.  The deliberate
absence is the point.  See the docstring at the top of
`baselines/visual_only_tracker.py` for the longer rationale.

## When to use

* You want a baseline number to compare the EKF or heuristic tracker against.
* You're debugging whether a regression is in ICP itself or in the filter.
* You want to verify a pose pipeline end-to-end without bringing in
  proprioception.

## When *not* to use

* Production manipulation. The visual-only tracker has no concept of grasp
  attachment, gravity, occlusion handling, or relations.

```{seealso}
* [Choosing a tracker](../getting_started/choosing_a_tracker.md)
* [`scripts/examples/visual_only_baseline.py`](../reference/examples.md)
* [Auto-generated reference](api_reference.rst)
```
