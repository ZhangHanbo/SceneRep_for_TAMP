# scenerep

Object-centric 3D scene representation for robotic manipulation. Given a stream
of RGB-D frames, SLAM poses, and gripper proprioception, the package tracks
every object's world-frame pose, geometry, existence, and pairwise spatial
relations across grasp / lift / drop interactions. Two interchangeable trackers
are exposed (a deterministic TSDF + Hungarian variant and a probabilistic
Bernoulli-EKF variant) plus a visual-only ICP baseline.

## Quickstart

```bash
conda create -n scenerep python=3.11 && conda activate scenerep
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

For the live detection path, also fetch the
[OWLv2 ST/FT-ens](https://storage.googleapis.com/scenic-bucket/owl_vit/checkpoints/owl2-b16-960-st-ngrams-curated-ft-lvisbase-ens-cold-weight-05_209b65b)
and [SAM ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
checkpoints into `scripts/rosbag2dataset/{owl,sam}/` and update
`checkpoint_path` in the OWL config. Tracking on cached detections does not
need either checkpoint.

## APIs

Three top-level Python packages export the user-facing surface.

### `heuristic_tracker` — TSDF + Hungarian + ICP

Deterministic, no probabilistic state; the production path used by
`robi_butler` and the offline / realtime demos. Use this for fast deterministic
pose updates with TSDF geometry.

```python
from heuristic_tracker import (
    ObjectReconstructor, ObjectTracker, PoseUpdater, RelationAnalyzer,
)
```

### `ekf_tracker` — two-tier Bernoulli-EKF

Probabilistic tracker on SE(3) with a Bernoulli existence model and an
optional slow-tier pose graph. Output carries 6×6 covariance and existence
probability per object. Use this for research, occlusion-heavy scenes, or
anywhere uncertainty matters.

```python
from ekf_tracker import EkfTracker

tracker = EkfTracker(K=K, T_bc=T_bc)
scene   = tracker.step(detections, rgb, depth, slam_pose=T_wb,
                       T_bg=T_bg, gripper_width=w, joints=joints)
```

Five public methods: `detect`, `step`, `get_scene`, `get_points`, `smooth`.
Full reference at [`docs/ekf_tracker/index.html`](docs/ekf_tracker/index.html).

### `baselines` — visual-only ICP

Direct ICP composition with no filter and no proprioception. For ablation
only.

```python
from baselines import VisualOnlyTracker
```

## See also

- [`docs/ekf_tracker/index.html`](docs/ekf_tracker/index.html) — EKF API reference
- [`docs/ekf_tracker/DISCUSSION.md`](docs/ekf_tracker/DISCUSSION.md) — architectural rationale
- [`docs/ekf_tracker/latex/bernoulli_ekf.pdf`](docs/ekf_tracker/latex/bernoulli_ekf.pdf) — algorithm derivation
- [`docs/survey_and_analysis.md`](docs/survey_and_analysis.md) — comparative survey
- [`scripts/examples/`](scripts/examples/) — runnable recipes
- [`scripts/rosbag2dataset/`](scripts/rosbag2dataset/) — ROS-bag → dataset conversion
