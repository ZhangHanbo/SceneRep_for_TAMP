# Choosing a tracker

The repo ships three trackers behind clean public APIs.  Pick by the column
that matches your constraint.

| Property | `heuristic_tracker` | `ekf_tracker` | `baselines.VisualOnlyTracker` |
|---|---|---|---|
| Style | TSDF + Hungarian + ICP | Bernoulli-EKF on SE(3) (+ optional GTSAM smoother) | Direct ICP, frame-to-frame |
| State per object | Pose + accumulated TSDF | Mean pose, 6×6 cov, Bernoulli `r` | Pose only |
| Uncertainty output | None | Yes (covariance + existence) | None |
| Mesh / TSDF output | Yes (`Mesh`, `TrackedObject.points`) | Point cloud only (`get_points`) | None |
| Spatial relations | Yes (`RelationAnalyzer`) | Yes (`RelationOrchestrator`, returned in `SceneView`) | None |
| Handles occlusion | Identity hold while invisible | Predict-only step, cov inflates, `r` decays | No (rejects updates) |
| Handles grasp/release | Yes (gravity drop heuristic) | Yes (rigid-attach + parametric release prior) | No |
| Determinism | Deterministic | Deterministic | Deterministic |
| External deps | open3d, scipy | open3d, scipy, gtsam (smoother only) | open3d |
| Used by | `robi_butler`, offline + ROS demos | Research, paper artifacts, parity tests | Ablation only |
| Entry point | `from heuristic_tracker import ObjectTracker` | `from ekf_tracker import EkfTracker` | `from baselines import VisualOnlyTracker` |

```{admonition} TL;DR
:class: tip

* Shipping a robot? **Heuristic.**
* Need uncertainty, want occlusions handled, doing research? **EKF.**
* Comparing against "what does ICP alone do?" **Visual-only.**
```

## Decision questions

**Do you need a mesh out?**
Only the heuristic tracker reconstructs TSDF meshes today.  EKF emits per-object
point clouds (the ICP reference cloud) via
{py:meth}`ekf_tracker.api.EkfTracker.get_points`; you can run TSDF over those
externally if needed.

**Do you need covariance / existence?**
Only EKF.  The 6×6 covariance is in the tangent space at the mean, ordered
`[translation, rotation]`; the existence is a scalar Bernoulli probability
in `[0, 1]`.  See [`cov_anisotropy_explained`](../ekf_tracker/cov_anisotropy_explained.md)
for what the covariance actually expresses under base motion.

**How heavy is the slow tier?**
The EKF's pose-graph smoother runs only when triggered (grasp / release / new
object / every N frames; see {py:class}`ekf_tracker.config.TriggerConfig`).
The fast tier alone is comparable to the heuristic tracker in cost.

**Can I switch trackers without rewriting?**
Yes for the offline pipeline — `scripts/data_demo.py` accepts
`--tracker {heuristic,ekf}` and they consume the same dataset format.
Real-time (`scripts/realtime_app.py`) is heuristic-only today.

```{seealso}
* [`scripts/examples/compare_trackers.py`](../reference/examples.md) prints
  per-frame `T_wo` from heuristic + EKF side by side.
* [Architecture overview](../architecture/overview.md) explains the layered
  design that makes tracker swaps cheap.
```
