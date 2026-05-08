# EKF tracker — API reference

Per-frame Bernoulli-EKF object tracker. Five-method Python API, single-import:
`from ekf_tracker import EkfTracker`.

---

## Install

```bash
conda create -n scenerep python=3.11 && conda activate scenerep
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -r requirements.txt
```

## End-to-end shape

```python
from ekf_tracker import EkfTracker

tracker = EkfTracker(K=K, T_bc=T_bc)

for rgb, depth, T_wb, T_bg, width, joints in stream:
    dets, hist = tracker.detect(rgb, ["apple", "bowl"])           # or use cached dets
    scene = tracker.step(dets, rgb, depth,
                         slam_pose=T_wb, T_bg=T_bg,
                         gripper_width=width, joints=joints)
    for oid, obj in scene.objects.items():
        print(oid, obj.label, obj.pose[:3, 3], "r =", obj.r)
```

---

## The five APIs

### `EkfTracker(K, *, T_bc=None, ...)`

Constructs a tracker. Defaults are loaded from `ekf_tracker/configs/default.yaml`; pass `K` (3×3 intrinsics) and the `T_bc` camera-from-base extrinsic if you have it.

**In:** camera intrinsics & extrinsics. **Out:** tracker instance.

```python
import numpy as np
from ekf_tracker import EkfTracker

K    = np.array([[554.38, 0, 320.5], [0, 554.38, 240.5], [0, 0, 1]])
T_bc = np.eye(4)
tracker = EkfTracker(K=K, T_bc=T_bc)
```

<details>
<summary>All constructor keyword args</summary>

| kwarg | default | meaning |
|---|---|---|
| `T_bc` | `None` | camera-from-base extrinsic; can also be passed per-frame to `step` |
| `robot_type` | `"fetch"` | URDF lookup key; `"panda"` etc. are valid if the geometry exists |
| `relation_backend` | `None` | overrides the YAML's `relation.backend`: `"llm"` \| `"rest"` \| `"none"` |
| `relation_cache_dir` | `None` | filesystem cache for the relation client |
| `pose_method` | `"icp_chain"` | fast-tier pose method |
| `owl_server` | `None` | OWL-ViT server URL for `detect()` |
| `sam2_server` | `None` | SAM2 streaming server URL for `detect()` |
| `bernoulli_cfg` | `None` | explicit `BernoulliConfig` (otherwise from YAML) |
| `voxel_obs` | `None` | explicit `VoxelObservability` (otherwise from YAML) |
| `image_shape` | `(480, 640)` | `(H, W)` for the in-frame border birth gate |
| `trigger` | `None` | explicit `TriggerConfig` for slow-tier scheduling |

Source: `ekf_tracker/api.py::EkfTracker.__init__`.

</details>

---

### `tracker.detect(rgb, vocabulary, history=None)`

Runs the live OWL-ViT + SAM2-streaming detection pipeline. Pass `history=None` on the first call; thread the returned session handle through subsequent calls. Skip this method if you have cached detections.

**In:** `rgb (H,W,3) uint8`, `vocabulary: List[str]`, optional session handle.
**Out:** `(detections: List[Dict], history)`.

```python
dets, hist = tracker.detect(rgb, vocabulary=["apple", "bowl"], history=None)
# next frame:
dets, hist = tracker.detect(rgb_next, ["apple", "bowl"], hist)
```

<details>
<summary>Detection dict format</summary>

```python
{
    "id":         17,                  # SAM2 tracklet id (stable across frames)
    "object_id":  17,                  # alias of id
    "label":      "apple",
    "score":      0.94,                # this frame's OWL score
    "mean_score": 0.91,                # rolling mean over n_obs frames
    "n_obs":      12,
    "box":        [x1, y1, x2, y2],
    "mask":       "<base64 PNG, decoded to (H,W) {0,1} uint8>",
}
```

The cached files at `tests/visualization_pipeline/<traj>/perception/detection_h/*.json` use this format; you can feed them straight into `step()` without ever calling `detect()`.

</details>

---

### `tracker.step(detections, rgb, depth, *, slam_pose, T_bc=None, T_bg=None, gripper_width=None, joints=None) → SceneView`

One frame of the canonical pipeline (predict → update → birth/prune). `slam_pose` is the world-from-base SLAM pose `T_wb`.

**In:** detections, `rgb`, `depth (H,W) float32 m`, `slam_pose (4,4)`, gripper extrinsics & proprio.
**Out:** `SceneView`.

```python
scene = tracker.step(
    dets, rgb, depth,
    slam_pose=T_wb, T_bc=T_bc, T_bg=T_bg,
    gripper_width=width, joints=joints,
)
print(len(scene.objects), "objects;", len(scene.relations), "edges")
```

<details>
<summary>All <code>step()</code> arguments</summary>

| arg | type | meaning |
|---|---|---|
| `detections` | `List[Dict]` | per-frame detections (see format above) |
| `rgb` | `(H,W,3) uint8` | RGB image |
| `depth` | `(H,W) float32` | depth in metres; 0/NaN for invalid |
| `slam_pose` | `(4,4)` | world-from-base SLAM pose `T_wb` |
| `T_bc` | `(4,4)` \| `None` | camera-from-base; falls back to constructor value |
| `T_bg` | `(4,4)` \| `None` | gripper-from-base; needed for gripper FSM & gravity predict |
| `gripper_width` | `float` \| `None` | finger gap, metres |
| `joints` | `Dict[str,float]` \| `None` | joint state for grasp-owner geometric containment |

Source: `ekf_tracker/api.py::EkfTracker.step` — mirrors `scripts/visualize_ekf_tracking.py:main()`.

</details>

---

### `tracker.get_scene() → SceneView`

Read-only snapshot of currently tracked objects + relations, without stepping.

**In:** none. **Out:** `SceneView`.

```python
scene = tracker.get_scene()
for oid, obj in scene.objects.items():
    print(oid, obj.label, obj.pose[:3, 3], "r =", obj.r)
for edge in scene.relations:
    print(edge["parent"], edge["type"], edge["child"], edge["score"])
```

<details>
<summary>SceneView / EkfObject fields</summary>

```python
@dataclass
class SceneView:
    objects:   Dict[int, EkfObject]      # oid → snapshot
    relations: List[Dict[str, Any]]      # [{parent, child, type, score}, …]

@dataclass
class EkfObject:
    id:    int                            # stable oid assigned at birth
    label: str
    pose:  np.ndarray                     # (4,4) world-frame mean T_wo
    cov:   np.ndarray                     # (6,6) tangent covariance, [v, ω]
    r:     float                          # Bernoulli existence ∈ [0, 1]
```

</details>

---

### `tracker.get_points(object_id) → np.ndarray`

Accumulated ICP reference cloud for one tracked object, transformed to world coordinates.

**In:** `object_id: int`.
**Out:** `(N, 3) float32`; empty `(0, 3)` if the oid is unknown or has no points yet.

```python
pts = tracker.get_points(0)
print(pts.shape)                          # (N, 3)
```

---

### `tracker.smooth() → SceneView`

Runs the slow-tier `PoseGraphOptimizer` over the cached priors + relation graph and returns the refreshed scene. Re-uses the SLAM pose, extrinsics, and held-state from the most recent `step()`. No-op (returns empty `SceneView`) if `step()` has not been called.

**In:** none. **Out:** `SceneView`.

```python
scene = tracker.smooth()
```

---

## Reference

### Configuration (two-hierarchy YAML)

Every parameter the pipeline reads lives in one of two files:

| file | role |
|---|---|
| [`ekf_tracker/configs/default.yaml`](../../ekf_tracker/configs/default.yaml) | canonical defaults — read-only |
| [`configs/ekf_tracker/customization.yaml`](../../configs/ekf_tracker/customization.yaml) | your overrides; `_extends:` the default and deep-merges |

Missing keys hard-error at load time. No env vars, no constructor fallbacks.

```python
from ekf_tracker.configs import load_config, to_bernoulli_config, to_trigger_config
cfg  = load_config()                      # default.yaml
bcfg = to_bernoulli_config(cfg, K=K, image_shape=(480, 640))
tcfg = to_trigger_config(cfg)
```

<details>
<summary>Top-level YAML sections</summary>

| section | maps to |
|---|---|
| `bernoulli` | `BernoulliConfig` — fast-tier gates, weights, birth params |
| `trigger` | `TriggerConfig` — slow-tier scheduling |
| `ekf_tracker` | `EkfTracker.__init__` runtime knobs |
| `process_noise` | per-phase Q schedule (7 regimes) |
| `fast_tier_noise` | centroid std, rotation decouple, tiny-cov |
| `pose_estimator` | ICP voxel size / threshold / fitness floors |
| `perception` | visibility & det-dedup |
| `voxel_observability` | workspace AABB, voxel size, integrate kwargs |
| `gripper_phase` | FSM thresholds (closed/open width, close delta, …) |
| `grasp_owner` | 3-tier grasp-owner detector |
| `gravity_predict` | release-time landing prior |
| `object_dynamics` | per-label physical properties |
| `relation` | backend + EMA filter + trigger |

</details>

### Subsystems (advanced — bypass the facade)

The classes `EkfTracker` composes are individually public and configurable:

| subsystem | class / function |
|---|---|
| Birth gating | `perception/birth_gating.py` — `BirthGateConfig`, `is_near_live_track` |
| Gripper phase FSM | `utils/gripper_state.py` — `GripperPhaseTracker.step` |
| Grasp owner | `ekf_tracker/manipulation/grasp_owner_detector.py` — `GraspOwnerDetector.select` |
| Gravity predict | `ekf_tracker/manipulation/gravity_predict.py` — `predict_landing_pose` |
| Relations | `ekf_tracker/relations/relation_orchestrator.py` — `RelationOrchestrator` |
| Low-level orchestrator | `ekf_tracker/orchestrator_gaussian.py` — `TwoTierOrchestratorGaussian` |

See the source files for full signatures. [`bernoulli_ekf.pdf`](./latex/bernoulli_ekf.pdf) Part III maps every algorithm component to the file/function that implements it.

### CLI scripts

```bash
# Canonical EKF entry point: cached perception → ekf_state/*.json + ekf_debug.mp4
python scripts/visualize_ekf_tracking.py --trajectory apple_drop \
    [--max-frame N] [--start 0] [--step 1] \
    [--config-path configs/ekf_tracker/customization.yaml]

# Offline batch (dispatches to the above for known trajectories)
python scripts/data_demo.py --config configs/demo.yaml --tracker ekf \
    [--ekf-backend gaussian|rbpf] [--max-frames N]

# Multi-trajectory eval
python scripts/eval_run.py --config configs/eval.yaml --tracker ekf
```

`scripts/realtime_app.py` is currently heuristic-only — passing `--tracker ekf` raises `NotImplementedError`.

### Examples on disk

| file | shows |
|---|---|
| [`track_apple_in_the_tray.py`](../../scripts/examples/track_apple_in_the_tray.py) | full `EkfTracker` on a cached trajectory — the most representative end-to-end |
| [`ekf_offline.py`](../../scripts/examples/ekf_offline.py) | low-level `TwoTierOrchestratorGaussian` on synthetic data — runs anywhere |
| [`compare_trackers.py`](../../scripts/examples/compare_trackers.py) | heuristic + EKF side-by-side, prints poses per frame |

### Further reading (theory & rationale)

- [`latex/bernoulli_ekf.pdf`](./latex/bernoulli_ekf.pdf) — full algorithm derivation; Part III is the algorithm-to-code map
- [`DISCUSSION.md`](./DISCUSSION.md) — architectural rationale (two-tier hierarchy, base-frame fusion)
- [`PLAN.md`](./PLAN.md) — implementation roadmap with status per task
- [`improvements.md`](./improvements.md) — original EKF + factor-graph design notes
- [`cov_anisotropy_explained.md`](./cov_anisotropy_explained.md) — covariance anisotropy under base motion
- [`../survey_and_analysis.md`](../survey_and_analysis.md) — comparative survey vs other systems
- [`../../README.md`](../../README.md) — top-level installation, ROS-bag conversion, tracker comparison
