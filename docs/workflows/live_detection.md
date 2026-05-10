# Workflow: live detection (OWLv2 + SAM2)

Stand up the open-vocabulary detection stack used by both the offline
pipeline and the realtime ROS node.

## What runs where

```
RGB frame
    │
    ▼
[ OWLv2 server  ]  perception/det_pipeline/det_server.py
    │ detections (boxes + label scores)
    ▼
[ SAM2 server   ]  scripts/rosbag2dataset/sam2/…
    │ refined masks + stable tracklet IDs
    ▼
detections/*.json (the tracker's input)
```

## Prerequisites

* JAX-CUDA installed (see [Installation](../getting_started/installation.md)
  step 3).
* OWLv2 and SAM ViT-B checkpoints on disk (step 4 of installation).

## Start the OWL server

```bash
python perception/det_pipeline/det_server.py \
    --owl-checkpoint scripts/rosbag2dataset/owl/<owl-ckpt-dir> \
    --port 7860
```

Wait until you see `Uvicorn running on http://0.0.0.0:7860`.  First-time
startup takes ~30 s while JAX warms the model.

## Start the SAM2 server

```bash
cd scripts/rosbag2dataset/sam2
python streaming_sam2_server.py --port 7861 \
    --checkpoint ../sam/sam_vit_b_01ec64.pth
```

The two servers are **independent processes** so each can be restarted
without dragging the other down.

## Smoke test

```bash
python perception/det_pipeline/client_demo.py \
    --image scripts/rosbag2dataset/test.png \
    --owl-server http://localhost:7860
```

Should print a list of detections and write a debug overlay to `client_demo_out.png`.

`scripts/rosbag2dataset/probe_servers.py` is the lightest probe — it just
pings both servers and prints their version strings.

## Wire the servers into a tracker

### Direct API call

```python
from ekf_tracker import EkfTracker

tracker = EkfTracker(
    K=K, T_bc=T_bc,
    owl_server="http://localhost:7860",
    sam2_server="http://localhost:7861",
)

dets, hist = tracker.detect(rgb, vocabulary=["apple", "tray"])
scene = tracker.step(dets, rgb, depth, slam_pose=T_wb, T_bg=T_bg, ...)
```

### Offline pipeline

```bash
python scripts/data_demo.py \
    --tracker ekf \
    --config configs/ekf_tracker/customization.yaml \
    --dataset datasets/<traj> \
    --owl-server http://localhost:7860 \
    --sam2-server http://localhost:7861
```

### Caching

If your offline run needs to reproduce identical detections every time,
cache the per-frame JSON:

```bash
ls datasets/<traj>/perception/detection_h/
# detection_000000.json
# detection_000001.json
# …
```

Subsequent runs read these files instead of hitting the servers.  The
schema is documented in [Detection pipeline](../perception/detection_pipeline.md)
and {ref}`detection-dict-format`.

## Vocabulary

The default vocabulary is

```
milkbox, cola, cup, apple, pot, flowerpot
```

Override per request by passing a `vocabulary` list to
{py:meth}`ekf_tracker.api.EkfTracker.detect`.  The server treats the prompt
as ranked; longer lists slow inference roughly linearly.

## GPU memory

* OWL ≈ 35 % of an A100's HBM at fp16 (model + activations).
* SAM2 ≈ 25 %.
* Run both on the same GPU only if you have ≥ 24 GB; otherwise put SAM2 on a
  second device with `CUDA_VISIBLE_DEVICES`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `connection refused` | Server not up. | Check the process and port. |
| `out of memory` mid-stream | Two servers on one GPU. | Split GPUs or lower batch size. |
| Detections come back but no masks | SAM2 server unreachable. | Restart the SAM2 server; OWL alone returns boxes only. |
| First-frame detections sparse, later ones fine | Default first-frame threshold (0.2) is intentional. | Pass `score_threshold=0.05` if your scene's first frame is hard. |

```{seealso}
* [Detection pipeline](../perception/detection_pipeline.md) — function shape
  and JSON schema.
* [rosbag → dataset](rosbag_to_dataset.md) — pipeline that calls these
  servers for every extracted frame.
```
