(detection-dict-format)=
# Detection pipeline

The detection sub-stack runs **OWLv2** (open-vocabulary object detection)
followed by **SAM2** (mask refinement and cross-frame tracklet IDs).  The
output is the same per-frame JSON consumed by every tracker.

This page covers the public function shape; for an end-to-end setup see
[Live detection](../workflows/live_detection.md).

## Function

```python
def detect_objects_on_image(rgb_image, frame_idx, datapath, score_threshold=None):
    """Run OWL-ViT detection on a single RGB frame.

    Args:
        rgb_image (np.ndarray): RGB image, shape ``(H, W, 3)``, dtype uint8.
        frame_idx (int): Frame index, used to build the output filename
            (``detection_000000.json`` etc.).
        datapath (str | Path): Output directory.  Both the JSON result and a
            rendered overlay PNG are written here.
        score_threshold (float | None): Optional score floor.  When None,
            uses 0.2 for the first frame and 0.02 for every subsequent
            frame (the OWL-ViT model warms up on the first call).

    Returns:
        dict:  ``{"detections": [...]}`` — see schema below.
    """
```

## Output schema

```python
{
    "detections": [
        {
            "detection": [
                {"label": "milkbox", "score": 0.85},
                {"label": "cola",    "score": 0.12},
            ],
            "box": [x1, y1, x2, y2],          # pixel coordinates
        },
        # ...
    ]
}
```

The same schema (extended with `mask` and stable `id` fields once SAM2 has
run) is what every tracker reads.  See {ref}`detection-dict-format` in the
EKF API doc for the post-SAM2 form, including the base64-PNG mask payload.

## Files written

For each call, two files are produced under ``datapath``:

| File | Contents |
|---|---|
| `detection_{frame_idx:06d}.json` | The dict above. |
| `detection_{frame_idx:06d}.png`  | RGB image with detection boxes + labels rendered. |

## Default vocabulary

When no override is supplied, the detector ranks the following labels:

```
milkbox, cola, cup, apple, pot, flowerpot
```

You can override per call by passing a custom vocabulary list to
{py:meth}`ekf_tracker.api.EkfTracker.detect`, or by editing the OWL config in
``perception/det_pipeline/``.

## Usage

### Single image

```python
import numpy as np
from PIL import Image
from perception.det_pipeline.detect_objects import detect_objects_on_image

rgb = np.array(Image.open("frame.png").convert("RGB"))
out = detect_objects_on_image(rgb, frame_idx=0, datapath="./detections")

print(f"{len(out['detections'])} detections")
for d in out["detections"]:
    print(d["box"], d["detection"][0]["label"], d["detection"][0]["score"])
```

### Folder batch

```python
import os
from PIL import Image
import numpy as np
from perception.det_pipeline.detect_objects import detect_objects_on_image

src = "path/to/images"
dst = "path/to/detections"
os.makedirs(dst, exist_ok=True)

for i, fname in enumerate(sorted(os.listdir(src))):
    if not fname.endswith(".png"):
        continue
    rgb = np.array(Image.open(os.path.join(src, fname)).convert("RGB"))
    out = detect_objects_on_image(rgb, frame_idx=i, datapath=dst)
    print(fname, len(out["detections"]))
```

## Implementation notes

* The model is loaded lazily on the first call and reused thereafter, so
  call ``detect_objects_on_image`` repeatedly rather than re-importing.
* GPU is used automatically when available; the GPU cache is cleared after
  each call to keep memory bounded.
* The first frame uses a higher score threshold (0.2) than subsequent
  frames (0.02) to suppress spurious early-frame detections before the
  cross-frame tracklet logic stabilizes.

## Errors

* `ValueError` — input image is not `(H, W, 3)` uint8.
* Anything else — model loading or inference failure; check the OWL/SAM2
  checkpoints listed in [Installation](../getting_started/installation.md).

## Dependencies

`jax`, `numpy`, `opencv-python`, `Pillow`, `scipy`, `scikit-image`,
`scenic` (OWL-ViT), `torch`.

```{seealso}
* [Live detection workflow](../workflows/live_detection.md) — full server setup.
* [Perception API reference](api_reference.rst) — auto-generated module docs.
* [`scripts/rosbag2dataset/`](../workflows/rosbag_to_dataset.md) — the
  rosbag pipeline calls this for every extracted frame.
```
