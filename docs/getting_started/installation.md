# Installation

The repo runs on Linux + Python 3.11. macOS works for everything except the
ROS realtime path and JAX-CUDA detection.

## 1. Clone & make a fresh environment

```bash
git clone git@github.com:ZhangHanbo/dynamic_scene_graph.git
cd dynamic_scene_graph

conda create -n dynamic_scene_graph python=3.11
conda activate dynamic_scene_graph
```

## 2. Core dependencies

```bash
pip install -r requirements.txt
pip install -e .       # registers the four packages on the import path
```

`requirements.txt` pins the exact set used in development; if you only need
the trackers (no live OWL/SAM2 detection, no eval renderer), the lean install
is:

```bash
pip install numpy scipy "open3d>=0.18" opencv-python pyyaml filterpy
```

## 3. Optional: JAX with CUDA (for live detection only)

The OWLv2 detector runs on JAX. Install the CUDA build only if you intend to
call {py:meth}`ekf_tracker.api.EkfTracker.detect` or run
``perception/det_pipeline/det_server.py``:

```bash
pip install --upgrade "jax[cuda12]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

If you only want to track on cached detections (the typical case), skip this
step entirely — JAX is `autodoc_mock_imports`-mocked at doc build time and
not touched by the tracker fast path.

## 4. Optional: model checkpoints

Live detection needs two checkpoints under `scripts/rosbag2dataset/`:

| Checkpoint | Path | Source |
|---|---|---|
| OWLv2 ST/FT-ens | `scripts/rosbag2dataset/owl/` | [`owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b`](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b) |
| SAM ViT-B | `scripts/rosbag2dataset/sam/sam_vit_b_01ec64.pth` | [Meta's SAM repo](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |

Update `checkpoint_path` in the OWL config to point at the local copy.

## 5. Optional: GTSAM (slow-tier pose graph)

The factor-graph smoother in {py:meth}`ekf_tracker.api.EkfTracker.smooth`
requires GTSAM:

```bash
pip install gtsam
```

Without it, `smooth()` raises at import time but `step()` keeps working.
``tests/conftest.py`` stubs ``gtsam`` so the unit suite collects without it.

## 6. Optional: ROS1 (realtime app only)

`scripts/realtime_app.py` is a ROS1 noetic node. It needs a parallel
ROS environment (`roscpp`, `cv_bridge`, `tf2_ros`) on top of the conda env.
Skip if you only run offline pipelines.

## 7. Verify

```bash
pip install -r requirements-docs.txt   # if you also want to build the docs
python scripts/examples/heuristic_offline.py
python scripts/examples/ekf_offline.py
```

Both print one line per synthetic frame and exit `0` on success. If they
do, the install is good.

```{seealso}
[Quickstart](quickstart.md) — track a real cached trajectory in five lines.
```
