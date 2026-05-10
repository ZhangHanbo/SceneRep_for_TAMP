# Workflow: rosbag → dataset

Convert a recorded ROS1 bag into the directory layout consumed by the
trackers and the eval pipeline.

## What you start with

A `.bag` file containing at minimum:

| Topic | Type | Used for |
|---|---|---|
| `/head_camera/rgb/image_raw` | `sensor_msgs/Image` | RGB stream |
| `/head_camera/depth/image_raw` | `sensor_msgs/Image` | Depth stream |
| `/joint_states` | `sensor_msgs/JointState` | URDF FK for `T_bg` and arm masking |
| `/tf`, `/tf_static` | `tf2_msgs/TFMessage` | `T_bc` extrinsic |
| `/amcl_pose` | `geometry_msgs/PoseWithCovarianceStamped` | Layer-1 SLAM prior |

## What you get out

```
datasets/<traj>/
├── rgb/
│   ├── rgb_000000.png
│   ├── rgb_000001.png
│   └── …
├── depth/
│   ├── depth_000000.npy
│   └── …
├── pose_txt/
│   ├── amcl_pose.txt        # T_wb per frame
│   ├── T_bc.txt             # static base→camera extrinsic
│   ├── ee_pose.txt          # T_we per frame (forward kinematics)
│   └── joints_pose.json     # joint angles per frame
├── perception/
│   └── detection_h/         # OWLv2 + SAM2 detections per frame
└── eval/
    └── *.txt                # ground-truth pose files (optional)
```

This is the format consumed by `scripts/data_demo.py`, every
`tests/visualization_pipeline/` parity test, and `eval/eval_all.py`.

## Pipeline

The full chain is automated by `scripts/rosbag2dataset/run_pipeline.sh`:

```bash
cd scripts/rosbag2dataset
./run_pipeline.sh /path/to/recording.bag <traj-name>
```

Internally it runs four stages:

| Stage | Script | What it does |
|---|---|---|
| 1. Extract @ 5 Hz | `rosbag2dataset_5hz.py` | Sync RGB/depth/joints/TF/AMCL at 5 Hz; write `rgb/`, `depth/`, `pose_txt/`. |
| 2. ICP refine | `icp_amcl.py` | Refine `T_wb` against the depth cloud per frame to remove AMCL latency drift. |
| 3. Detect | `sam2/` + `owl/` scripts | Per-frame OWLv2 + SAM2 → `perception/detection_h/*.json`. |
| 4. Track IDs | `track_object_ids.py` | Stable cross-frame `object_id` (SAM2 tracklets aligned across re-detections). |

## Running stages individually

If you only need to refresh one stage (typical: re-run detection with a
new vocabulary), call the script directly:

```bash
# Re-extract from a different time range
python rosbag2dataset_5hz.py --bag /path/to.bag --out datasets/<traj> --start 12.0 --end 65.0

# Re-detect with a custom vocabulary
python sam2/detect_with_sam2.py --dataset datasets/<traj> \
    --vocabulary "apple,bowl,table"

# Re-track identities only
python track_object_ids.py --dataset datasets/<traj>
```

## Diagnostic helpers

| Script | What it does |
|---|---|
| `inspect_bags.py` | Lists topic names, message counts, time ranges. Run before extraction to confirm the bag has what you need. |
| `probe_servers.py` | Pings the OWL and SAM2 servers; useful before kicking off detection. |
| `extract_bag_local.py` | Direct rosbag reader that writes raw images / TF / joints to disk without the 5 Hz sync — handy for debugging. |

## Sanity checks

After a run, before pushing the dataset into a tracker:

```bash
# Counts should match across rgb / depth / pose_txt / detections
ls datasets/<traj>/rgb           | wc -l
ls datasets/<traj>/depth         | wc -l
wc -l datasets/<traj>/pose_txt/amcl_pose.txt
ls datasets/<traj>/perception/detection_h | wc -l

# Eyeball the gripper alignment with the dataset
python scripts/render_gripper_overlay.py 0 --trajectory <traj>
python scripts/visualize_arm_mask.py --trajectory <traj> --start 0 --max-frame 20
```

If the gripper overlay is offset from the actual gripper in the RGB image,
your `T_bc` is wrong — re-check the TF chain extracted by stage 1.

```{seealso}
* [Offline pipeline](offline_pipeline.md) — feed the dataset to a tracker.
* [Live detection](live_detection.md) — what the OWL/SAM2 servers do.
* [Debugging visualizers](debugging_visualizers.md) — render gripper, arm,
  and EKF state overlays.
```
