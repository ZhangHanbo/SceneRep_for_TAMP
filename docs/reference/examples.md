# Examples

Self-contained ~100-LOC recipes that exercise each tracker's public API
with a tiny synthetic dataset generated inline (so they double as smoke
tests).  The cached-trajectory recipe needs the
`tests/visualization_pipeline/apple_in_the_tray/` directory shipped with
the repo.

| File | Tracker | Purpose |
|---|---|---|
| [`scripts/examples/heuristic_offline.py`](https://github.com/) | `heuristic_tracker.ObjectTracker` | TSDF + ID-association on 5 synthetic frames; prints world-frame poses. |
| [`scripts/examples/ekf_offline.py`](https://github.com/) | `ekf_tracker.TwoTierOrchestratorGaussian` | Gaussian-backend EKF on the same 5 frames; compares with the heuristic. |
| [`scripts/examples/visual_only_baseline.py`](https://github.com/) | `baselines.VisualOnlyTracker` | ICP-only reference baseline. |
| [`scripts/examples/compare_trackers.py`](https://github.com/) | Both | Heuristic + EKF side-by-side; prints per-frame `T_wo` discrepancy. |
| [`scripts/examples/track_apple_in_the_tray.py`](https://github.com/) | `ekf_tracker.EkfTracker` | End-to-end on the cached `apple_in_the_tray` trajectory; exercises every `RelationOrchestrator`, `GripperPhaseTracker`, and `GraspOwnerDetector` path. |

## Run any example from the repo root

```bash
python scripts/examples/heuristic_offline.py
python scripts/examples/ekf_offline.py
python scripts/examples/visual_only_baseline.py
python scripts/examples/compare_trackers.py
python scripts/examples/track_apple_in_the_tray.py     # needs cached trajectory
```

The first four print one line per frame and exit `0` on success — they
generate their inputs inline so they work from a clean clone.

## What each one shows

### `heuristic_offline.py`

Constructs a 5-frame synthetic dataset (cube on a table, mask + depth
hand-built per frame) and walks the full
{py:class}`heuristic_tracker.api.ObjectTracker` cycle: `update` →
`detect_held_object` → `set_held_object`.  Output: per-frame `(oid,
label, T_wo[:3,3])`.

### `ekf_offline.py`

Same five frames, but uses the lower-level
{py:class}`ekf_tracker.orchestrator_gaussian.TwoTierOrchestratorGaussian`
directly (no facade).  Useful as the minimal example of the EKF pipeline
without OWL/SAM2.

### `visual_only_baseline.py`

Drives {py:class}`baselines.visual_only_tracker.VisualOnlyTracker` on the
same five frames so you can see how a no-filter ICP behaves under camera
motion.  The output diverges from the EKF when the object is occluded.

### `compare_trackers.py`

Heuristic and EKF run in lockstep on the same input.  Useful when a
regression appears in one tracker but not the other.

### `track_apple_in_the_tray.py`

The most representative example: drives the full
{py:class}`ekf_tracker.api.EkfTracker` facade (all five public methods)
against a cached real-robot trajectory.  Reads detections from
`tests/visualization_pipeline/apple_in_the_tray/perception/detection_h/`
and prints {py:class}`ekf_tracker.api.SceneView` per frame.  The basis of
[Quickstart](../getting_started/quickstart.md).

```{seealso}
* [Quickstart](../getting_started/quickstart.md) — the same example with
  more narration.
* [Choosing a tracker](../getting_started/choosing_a_tracker.md) — when to
  use each.
```
