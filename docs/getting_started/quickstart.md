# Quickstart

Five lines of Python that turn the cached `apple_in_the_tray` trajectory into
world-frame object poses with covariance and existence probabilities.

## Prerequisites

* The package installed ([Installation](installation.md)).
* The cached trajectory present at
  `tests/visualization_pipeline/apple_in_the_tray/`. It ships with the
  repo — no download needed.

## Run

```bash
python scripts/examples/track_apple_in_the_tray.py
```

That script loads RGB-D + SLAM poses + cached detections and walks the
canonical `step()` loop:

```python
from ekf_tracker import EkfTracker

tracker = EkfTracker(K=K, T_bc=T_bc)

for rgb, depth, T_wb, T_bg, width, joints, dets in stream:
    scene = tracker.step(
        dets, rgb, depth,
        slam_pose=T_wb, T_bg=T_bg,
        gripper_width=width, joints=joints,
    )
    for oid, obj in scene.objects.items():
        print(oid, obj.label, obj.pose[:3, 3], "r =", obj.r)
```

You should see one block of object lines per frame, like:

```
0 apple [-0.12  0.41  0.78] r = 0.97
1 tray  [-0.05  0.39  0.74] r = 0.99
```

Each `obj` carries:

* `pose` — 4×4 world-frame `T_wo` (mean).
* `cov`  — 6×6 tangent covariance, `[v, ω]` ordering.
* `r`    — Bernoulli existence probability ∈ [0, 1].

## What just happened?

1. **Detections** were loaded from
   `tests/visualization_pipeline/apple_in_the_tray/perception/detection_h/*.json`
   — pre-computed OWLv2 + SAM2 outputs in the
   {ref}`detection-dict-format`.
2. **The fast tier** (Bernoulli-EKF in the robot's base frame) ran for every
   frame: predict the per-object Gaussian forward, Hungarian-associate to
   detections, update with ICP, gate births, decay misses, prune.
3. **Manipulation logic** consumed `gripper_width` + `joints` to detect grasp
   onset, attach the held object rigidly to the gripper while held, and
   apply a parametric gravity-bounce-roll prior on release.
4. **Relations** ("apple in tray", etc.) were re-evaluated on event triggers
   and EMA-smoothed.

Composing every frame's base-frame mean with `T_wb` gives world poses; that
composition lives entirely in {py:meth}`ekf_tracker.api.EkfTracker.step`.

## Next steps

| If you want to … | Read |
|---|---|
| Pick between heuristic / EKF / baseline | [Choosing a tracker](choosing_a_tracker.md) |
| See covariance and Bernoulli `r` rendered as a video | [Debugging visualizers](../workflows/debugging_visualizers.md) |
| Run on a fresh rosbag | [rosbag → dataset](../workflows/rosbag_to_dataset.md) → [Offline pipeline](../workflows/offline_pipeline.md) |
| Run live detection (no caches) | [Live detection](../workflows/live_detection.md) |
| Understand the math | [`bernoulli_ekf.pdf`](../ekf_tracker/latex/bernoulli_ekf.pdf) and [Architecture overview](../architecture/overview.md) |
