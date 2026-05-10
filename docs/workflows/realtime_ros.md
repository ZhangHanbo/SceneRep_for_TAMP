# Workflow: realtime ROS

Run the heuristic tracker as a ROS1 node, consuming live RGB-D + joints +
AMCL pose, with a pyrender viewer for the reconstructed mesh and pose.

```{warning}
`scripts/realtime_app.py` is **heuristic-only** today.  Passing
`--tracker ekf` raises `NotImplementedError`.
```

## Prerequisites

* ROS1 noetic (parallel to the conda env; not pip-installable).
* `cv_bridge`, `tf2_ros`, `sensor_msgs`, `nav_msgs` available on the ROS
  Python path.
* A robot publishing the topics from
  [rosbag → dataset](rosbag_to_dataset.md) — they're identical.
* The OWL/SAM2 detection servers (or a cached detection pipeline; see below).

## Run

```bash
roscore &

# Optionally start the OWL + SAM2 servers
python perception/det_pipeline/det_server.py &
# … plus the SAM2 server, see Live detection workflow

python scripts/realtime_app.py \
    _config:=configs/heuristic_tracker/realtime_app.yaml
```

The node:

1. Subscribes to camera RGB-D, joints, and AMCL pose.
2. Pulls the most-recent message per topic (no time-sync — see Caveats).
3. Runs the heuristic tracker pipeline (mask extraction → ID association →
   TSDF fusion → relation graph).
4. Updates a pyrender viewer in a background thread.

## Configuration

`configs/heuristic_tracker/realtime_app.yaml` controls:

* Camera intrinsics, TSDF voxel sizes, mask extraction method.
* Topic names (override defaults if your robot uses different names).
* Pyrender viewer options (window size, follow camera, etc.).

See [Configs reference](../reference/configs.md) for the per-field schema.

## Caveats

* **No time-synchronization.**  The node grabs the latest message on each
  topic on every tick; under heavy load, you can get a depth frame from
  100 ms ago and an RGB frame from now.  In practice the heuristic tracker
  is robust to this; the EKF would not be — that's a major reason
  `--tracker ekf` is not yet wired in.
* **Single-process pyrender.**  The viewer runs in a daemon thread; closing
  the window does not always shut down the ROS spinner.  Ctrl-C twice if it
  hangs.
* **No persistence.**  Output is in memory only.  To persist, run the
  offline pipeline against a recorded bag instead.

## Debugging

* No detections coming through?  Run `probe_servers.py` from the rosbag
  pipeline to confirm the OWL/SAM2 servers respond.
* Gripper overlay misaligned in the viewer?  Re-check the URDF FK chain
  and the `T_bc` extrinsic; `scripts/render_gripper_overlay.py` is a
  faster way to validate this offline against a saved frame.
* TSDF blurry?  The robot or camera is moving during fusion of a held
  object — confirm `set_held_object` is being called when the FSM enters
  the holding state.

```{seealso}
* [Offline pipeline](offline_pipeline.md) — same trackers, recorded data.
* [Live detection](live_detection.md) — the OWL/SAM2 server stack.
* [`scripts/realtime_app.py`](https://github.com/) — source for the node.
```
