# Workflow: offline pipeline

Run a tracker over a full dataset directory and dump per-frame state.

## Prerequisites

* A dataset directory in the format produced by
  [rosbag → dataset](rosbag_to_dataset.md) (or the cached
  `tests/visualization_pipeline/<traj>/`).
* A YAML config under `configs/<tracker>/`.

## EKF tracker (canonical entry point)

The full Bernoulli-EKF pipeline with the per-frame visualization is in
`scripts/visualize_ekf_tracking.py`:

```bash
python scripts/visualize_ekf_tracking.py \
    --trajectory apple_drop \
    --max-frame 200 --start 0 --step 1 \
    --config-path configs/ekf_tracker/customization.yaml
```

Outputs land under `tests/visualization_pipeline/<traj>/`:

```
ekf_state/<frame>.json     # per-object pose, cov, r per frame (machine-readable)
ekf_debug/frame_*.png      # 2×3 debug grid: detections, top-down state,
                           # covariance evolution, etc. (one PNG per frame)
ekf_debug.mp4              # PNGs assembled into a video
```

## Heuristic tracker

```bash
python scripts/data_demo.py \
    --config configs/heuristic_tracker/demo.yaml \
    --tracker heuristic \
    --dataset datasets/<traj>
```

Outputs:

```
datasets/<traj>/
├── tsdf_meshes/<oid>.ply      # per-object TSDF mesh
├── pose_txt/predicted_*.txt   # per-frame world pose
└── eval/                       # for downstream eval/eval_all.py
```

## Switching trackers

`scripts/data_demo.py` supports both:

```bash
python scripts/data_demo.py --tracker heuristic --config configs/heuristic_tracker/demo.yaml --dataset datasets/<traj>
python scripts/data_demo.py --tracker ekf       --config configs/heuristic_tracker/demo.yaml --dataset datasets/<traj>  # consumes the same dataset
```

Tracker output formats are compatible with `eval/eval_all.py` — see
[Evaluation](evaluation.md).

## Detection: cached vs live

By default, the offline pipeline reads cached detections from
`<dataset>/perception/detection_h/`.  To re-run detection live, drop the
cache and supply OWL/SAM2 servers:

```bash
rm -rf datasets/<traj>/perception/detection_h
python scripts/data_demo.py \
    --tracker ekf \
    --config configs/ekf_tracker/customization.yaml \
    --owl-server http://localhost:7860 \
    --sam2-server http://localhost:7861 \
    --dataset datasets/<traj>
```

See [Live detection](live_detection.md) for server setup.

## Multi-trajectory sweep

```bash
python scripts/eval_run.py \
    --config configs/heuristic_tracker/eval.yaml \
    --tracker ekf \
    --ekf-backend gaussian
```

This iterates over every dataset under the configured root, calling
`data_demo.py` once per trajectory.  Use `--ekf-backend rbpf` to swap in the
Rao-Blackwellized particle filter (research backend; see
{py:mod}`ekf_tracker.state.rbpf_state`).

## Configs

* `configs/ekf_tracker/customization.yaml` — extends the package-internal
  `default.yaml`; only the keys you change live here.
* `configs/heuristic_tracker/<scenario>.yaml` — flat per-scenario configs;
  no inheritance.

Schema details: [Configs reference](../reference/configs.md).

## Reproducing a parity test

```bash
pytest tests/integration/test_parity_apple_in_the_tray.py -v
```

That test drives the EKF tracker on the cached `apple_in_the_tray` data and
checks that pose error vs the stored JSON dumps is *exactly* zero — useful
as a regression smoke test after any change to the fast tier.

```{seealso}
* [Debugging visualizers](debugging_visualizers.md) — what the per-frame
  PNGs and `ekf_debug.mp4` show.
* [Evaluation](evaluation.md) — turn pose dumps into ADD / ADD-S tables.
* [Realtime ROS](realtime_ros.md) — the same trackers under a live ROS node.
```
