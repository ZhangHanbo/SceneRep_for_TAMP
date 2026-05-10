# EKF tracker

Per-frame Bernoulli-EKF object tracker on SE(3) with optional GTSAM
pose-graph smoothing.  Probabilistic, occlusion-aware, the canonical research
path.

```python
from ekf_tracker import EkfTracker

tracker = EkfTracker(K=K, T_bc=T_bc)
scene   = tracker.step(detections, rgb, depth, slam_pose=T_wb,
                       T_bg=T_bg, gripper_width=w, joints=joints)
```

Five public methods: `__init__`, `detect`, `step`, `get_scene`, `get_points`,
`smooth`.  See [API walkthrough](api.md) for the hand-written tour and
[API reference](api_reference.rst) for the auto-generated reference.

```{toctree}
:caption: EKF tracker
:maxdepth: 1

api
api_reference
DISCUSSION
cov_anisotropy_explained
PLAN
improvements
```

## What you get out

A {py:class}`ekf_tracker.api.SceneView` per frame:

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

## What's inside

```
ekf_tracker/
├── api.py                       # EkfTracker facade (5-method public API)
├── gaussian_ekf_tracker.py      # Fast tier: GaussianEkfTracker, InstrumentedTracker
├── orchestrator_gaussian.py     # Two-tier orchestrator (fast + GTSAM smoother)
├── perception_pipeline.py       # OWLv2 + SAM2 streaming detection pipeline
├── factor_graph.py              # GTSAM PoseGraphOptimizer (slow tier)
├── birth_gate.py                # Pending-birth buffer + admission policy
├── config.py                    # BernoulliConfig, TriggerConfig dataclasses
├── configs/                     # default.yaml + load_config()
├── state/                       # bernoulli, gaussian_state, obs_chain, rbpf_state
├── manipulation/                # gripper FSM, grasp owner, gravity predict
└── relations/                   # client + filter + orchestrator + utils
```

## Where to read more

| Topic | File |
|---|---|
| Hand-written API tour | [`api.md`](api.md) |
| Auto-generated reference | [`api_reference`](api_reference.rst) |
| Architectural rationale (two-tier, base-frame) | [`DISCUSSION.md`](DISCUSSION.md) |
| Why covariance ellipses look anisotropic | [`cov_anisotropy_explained.md`](cov_anisotropy_explained.md) |
| Original improvement roadmap (historical) | [`PLAN.md`](PLAN.md) |
| Per-object EKF + factor-graph design notes (historical) | [`improvements.md`](improvements.md) |
| Algorithm derivation (paper) | [`latex/bernoulli_ekf.pdf`](latex/bernoulli_ekf.pdf) |

```{seealso}
* [Architecture overview](../architecture/overview.md)
* [Choosing a tracker](../getting_started/choosing_a_tracker.md)
* [Configs reference](../reference/configs.md)
```
