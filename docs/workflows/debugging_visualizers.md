# Workflow: debugging visualizers

Per-frame visualizations that turn the tracker's internal state into
something you can eyeball.  Every visualizer reads the same dataset format
described in [rosbag → dataset](rosbag_to_dataset.md).

## EKF state — `scripts/visualize_ekf_tracking.py`

The canonical EKF entry point.  Replicates the production fast tier (the
`InstrumentedTracker` path through `ekf_tracker.gaussian_ekf_tracker`),
dumps the per-frame state, and renders a 2×3 debug grid.

```bash
python scripts/visualize_ekf_tracking.py \
    --trajectory apple_drop \
    --start 0 --max-frame 200 --step 1 \
    --config-path configs/ekf_tracker/customization.yaml
```

Outputs (under `tests/visualization_pipeline/<traj>/`):

* `ekf_state/<frame>.json` — full state per frame (machine-readable).
* `ekf_debug/frame_*.png` — six-panel debug image:
  1. Detections + masks overlaid on the RGB frame.
  2. Top-down view of object positions in world frame.
  3. Top-down view in base frame.
  4. Covariance ellipses (3 σ).
  5. Bernoulli `r` evolution.
  6. Per-track event log (births, prunes, associations).
* `ekf_debug.mp4` — the PNGs assembled into a video.

```{tip}
This is the fastest way to find what the EKF *thinks* is happening.  When
covariance balloons unexpectedly, panel 4 tells you which axis; when an
object dies, panel 5 + the event log tell you whether the cause was misses,
gating, or association loss.
```

## Voxel observability — `scripts/visualize_voxel_obs.py`

Open3D viewer for the `VoxelObservability` grid used by the gravity-drop
predictor.

```bash
python scripts/visualize_voxel_obs.py \
    --trajectory apple_drop \
    --animate-every 5 \
    --show-unseen \
    --with-tsdf \
    --column 0.0,0.6
```

Renders the workspace as a coloured point cloud:

| Color | Voxel state |
|---|---|
| Grey | Unobserved |
| Green | Empty (rays passed through) |
| Red | Occupied (rays terminated) |

Use this to debug "why did the gravity prediction land here?" — if the
support voxel is grey, the predictor never saw the surface and fell back
to the floor plane.

## Gripper geometry — `scripts/render_gripper_overlay.py`

Projects the URDF-derived gripper geometry (inside-jaws AABB, finger pads)
onto an RGB frame so you can verify the FK + extrinsic chain.

```bash
python scripts/render_gripper_overlay.py 488 \
    --trajectory apple_in_the_tray \
    --out gripper_overlay.png
```

Run this **before** trusting any tracker run on a fresh dataset.  If the
boxes don't sit on the actual gripper in the image, your `T_bc` is wrong
or the joint state is malformed.

## Arm masking — `scripts/visualize_arm_mask.py`

Renders per-link Fetch arm silhouettes onto RGB using
{py:class}`utils.fetch_arm_mask.ArmMaskBuilder`.  Verifies the arm-mask
pipeline before it's applied to the depth (so the SLAM doesn't see the
arm).

```bash
python scripts/visualize_arm_mask.py \
    --trajectory apple_in_the_tray \
    --start 0 --max-frame 50 --dilate-px 12
```

Each output frame shows the input RGB with the projected arm silhouette in
semi-transparent red.  Tune `--dilate-px` until every visible link is
covered with a ~5-pixel margin.

## Diagnostic-only visualizers (`tests/diagnostics/`)

Heavy or single-purpose visualizers that aren't meant for batch use:

| Script | Purpose |
|---|---|
| `tests/diagnostics/diagnose_ekf_births.py` | Replay birth events; show why each candidate was admitted or rejected. |
| `tests/diagnostics/diagnose_icp_methods.py` | Compare ICP backends (`centroid`, `icp_chain`, `icp_anchor`) on the same frame. |
| `tests/diagnostics/visualize_full.py` | All-in-one render: TSDF + EKF + relations. Slow. |
| `tests/diagnostics/visualize_trajectories.py` | World-frame trajectory plot for every tracked object. |
| `tests/diagnostics/visualize_uncertainty.py` | Stand-alone covariance evolution renderer. |
| `tests/diagnostics/render_api_videos.py` | Renders demo videos from saved API output. |

These are not pytest fixtures — run them directly with `python <path>`.

## Suggested debugging order

1. `render_gripper_overlay.py` — is the FK chain correct?
2. `visualize_arm_mask.py` — is the arm being masked from depth?
3. `visualize_ekf_tracking.py` — what is the EKF doing per frame?
4. `visualize_voxel_obs.py` — only when investigating gravity-drop or
   visibility surprises.
5. `tests/diagnostics/diagnose_*` — only when one of the above implicates
   a specific subsystem.

```{seealso}
* [Offline pipeline](offline_pipeline.md) — produce the cached state these
  visualizers read.
* [Architecture overview](../architecture/overview.md) — what each panel of
  the EKF debug grid corresponds to in the algorithm.
```
