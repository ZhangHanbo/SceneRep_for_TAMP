# Dynamic Scene Graph

Object-centric, dynamic 3D scene-graph representation for robotic
manipulation. Given a stream of RGB-D frames, SLAM poses, and gripper
proprioception, the package tracks every object's world-frame pose,
geometry, existence, and pairwise spatial relations across grasp / lift /
drop interactions.

Three trackers ship in the box:

| Package | Style | Use it when |
|---|---|---|
| [`heuristic_tracker`](heuristic_tracker/index.md) | TSDF + Hungarian + ICP, deterministic | Production / `robi_butler` integration, fast deterministic poses with mesh output |
| [`ekf_tracker`](ekf_tracker/index.md) | Bernoulli-EKF on SE(3) with optional pose-graph smoothing | Research, occlusion-heavy scenes, anywhere uncertainty matters |
| [`baselines`](baselines/index.md) | Visual-only ICP, no filter | Ablation / reference baseline only |

```{admonition} New here?
:class: tip

Start with [Quickstart](getting_started/quickstart.md) — five lines that turn
a cached trajectory into world-frame object poses. Then read
[Choosing a tracker](getting_started/choosing_a_tracker.md) and
[Architecture overview](architecture/overview.md) before integrating.
```

```{toctree}
:caption: Getting started
:maxdepth: 2

getting_started/installation
getting_started/quickstart
getting_started/choosing_a_tracker
```

```{toctree}
:caption: Workflows
:maxdepth: 2

workflows/rosbag_to_dataset
workflows/offline_pipeline
workflows/realtime_ros
workflows/live_detection
workflows/evaluation
workflows/debugging_visualizers
```

```{toctree}
:caption: Architecture
:maxdepth: 2

architecture/overview
architecture/frame_conventions
architecture/data_flow
```

```{toctree}
:caption: Trackers
:maxdepth: 2

ekf_tracker/index
heuristic_tracker/index
baselines/index
```

```{toctree}
:caption: Perception layer
:maxdepth: 2

perception/index
perception/detection_pipeline
perception/api_reference
```

```{toctree}
:caption: Reference
:maxdepth: 2

reference/configs
reference/utils
reference/examples
reference/glossary
```

```{toctree}
:caption: Background
:maxdepth: 1

background/survey_and_analysis
background/khronos_lessons
```

## Indices

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
