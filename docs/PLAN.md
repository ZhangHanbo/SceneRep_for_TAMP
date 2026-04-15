# SceneRep Improvement Plan — Condensed

Decoupled, actionable tasks. Each item can be implemented independently. Ordering reflects dependency, not priority.

## Scope

Transform SceneRep from a sequential pipeline into a two-tier architecture: per-object EKF (fast) + joint pose graph over movable objects (slow). Uncertainty from Layer 1 (SLAM) propagates down; movable objects do not feed back into Layer 1.

---

## What the Pose Graph Guarantees

Two robustness properties must hold end-to-end. Both are achieved by composable mechanisms spread across Tasks 2, 4, and 5; stated here explicitly so they are not implicit in the design.

### Guarantee A — Localization uncertainty softens observations

Every factor that depends on the camera pose (observation factors, manipulation factor) uses a composed noise model:

$$\Sigma_f = R_{\text{local}} + J \Sigma_{wb} J^T$$

where `R_local` is the factor's intrinsic measurement noise (e.g., ICP fitness-based), `J` is the Jacobian of the residual w.r.t. `T_wb`, and `Σ_wb` is the SLAM pose covariance from Layer 1. When `Σ_wb` is large, `Σ_f` is inflated, the factor's IRLS weight drops, and the object pose stays near its prior rather than chasing a geometrically unreliable observation. When `Σ_wb` is small, `Σ_f ≈ R_local` and observations dominate. No threshold — continuous degradation.

**Implemented by:** Task 2 (composes the noise for EKF observation update) and Task 4 (same composition inside each factor graph observation/manipulation factor).

### Guarantee B — Sparse outliers are auto-filtered

Every factor in the pose graph is wrapped in the adaptive Barron robust kernel with truncated partition function. Alternating minimization jointly estimates the kernel shape `α` from the current residual distribution and the poses that minimize the weighted least squares. When most residuals are small (inliers) and a few are large (e.g., an object was moved by a human between observations), `α` becomes strongly negative and the outlier factors receive near-zero weight, leaving the inliers to determine the solution.

**Assumption:** outliers are a minority (Paper 2 validated up to ~30-40%). For persistent disagreement (an object actually did move), the EKF prior decays via process noise across frames and consistent re-observations eventually dominate — the system converges to the new state over multiple frames rather than being fooled in a single one.

**Implemented by:** Task 5 (the kernel module) wired into Task 4 (wraps every factor in the graph).

### Composition — Orthogonal mechanisms

The two guarantees address different failure modes and compose without interference:

| Localization | Observation residual | Behavior |
| --- | --- | --- |
| Confident | Small | Observation factor dominates, pose updates toward observation. Standard case. |
| Confident | Large | Adaptive kernel downweights. Pose stays at prior. (Outlier rejected) |
| Uncertain | Small | Observation weak due to inflated `Σ_f`. Prior dominates. |
| Uncertain | Large | Already weak factor, further downweighted by kernel. Factor effectively removed. |

**What is NOT covered:** (1) systematic majority outliers (shelf rearrangement) — handled by eventual consensus switching rather than rejection; (2) slow systematic drift with small consistent residuals — invisible to both mechanisms, requires external correction (loop closure, manual re-init); (3) wrong scene graph relations — handled by the same adaptive kernel but with thin statistics if few relations are active.

---

## Module Interfaces

Shared types across modules:

```python
@dataclass
class PoseEstimate:
    T: np.ndarray        # (4,4) SE(3)
    cov: np.ndarray      # (6,6) in se(3) tangent, ordering [v, omega]

@dataclass
class Observation:
    obj_id: int
    T_co: np.ndarray     # (4,4) camera-to-object from ICP
    R_icp: np.ndarray    # (6,6) ICP measurement noise
    fitness: float       # [0,1]
    rmse: float
    mask: np.ndarray     # (H,W) bool, which pixels belonged to this object

@dataclass
class ManipulationState:
    phase: str           # 'idle' | 'grasping' | 'holding' | 'releasing'
    held_obj_id: Optional[int]
    T_oe: Optional[np.ndarray]   # (4,4) object-to-EE, locked at grasp

@dataclass
class SceneGraphRelation:
    type: str            # 'on' | 'in' | 'contact'
    parent: int
    child: int
    score: float         # [0,1] detection confidence
```

### Layer 1 — SLAM (Task 1)

Interface (fixed, SLAM-backend-agnostic):

```python
class SlamBackend(Protocol):
    def step(self, rgb: np.ndarray, depth: np.ndarray,
             movable_mask: np.ndarray,    # (H,W) bool, True where movable
             odom_prior: Optional[PoseEstimate]) -> PoseEstimate:
        """Returns (T_wb, Σ_wb). Movable-masked pixels are excluded from depth."""
```

Upstream dependency: none. Downstream: Tasks 2, 4, 7 consume `PoseEstimate`.

### Per-object EKF (Task 2)

```python
class ObjectEKF:
    def predict(self, dt: float, manip: ManipulationState) -> None:
        """Inflate cov by Q_t, set by manipulation phase + object's role."""

    def update(self, obs: Observation,
               slam: PoseEstimate,
               manip: ManipulationState) -> None:
        """R_eff = obs.R_icp + J·slam.cov·J^T + ρ(r)·R_robust.
        If object is the held/grasped/releasing target: fuse in base frame (Task 3)."""

    @property
    def posterior(self) -> PoseEstimate: ...

    @property
    def label_belief(self) -> Dict[str, Tuple[float, float]]:
        """Beta-Bernoulli (α, β) per label."""
```

Depends on: `PoseEstimate`, `Observation`, `ManipulationState` from Layer 1 + detection pipeline. Supported by: current `SceneObject` fields (extended with `cov`).

### Base-frame fusion (Task 3)

Not a module — a branch inside `ObjectEKF.update()`. Needs `ManipulationState` to decide fusion frame. No new interface.

### Joint pose graph (Task 4)

```python
class PoseGraphOptimizer:
    def run(self,
            slam: PoseEstimate,                        # fixed parameter
            priors: Dict[int, PoseEstimate],           # from Task 2 EKFs
            observations: List[Observation],           # accumulated since last run
            relations: List[SceneGraphRelation],      # from Task 6
            manip: ManipulationState,
            T_ew: Optional[np.ndarray],                # if manip.held_obj_id set
            active_set: Set[int],
           ) -> Dict[int, PoseEstimate]:
        """Returns optimized posteriors for objects in active_set.
        Non-active objects keep their prior unchanged."""
```

Depends on: `gtsam`, Task 2 posteriors, Task 6 relations, adaptive kernel (Task 5). Supported by: wrapper around GTSAM; current pipeline supplies all required inputs once Tasks 2+6 land.

### Adaptive robust kernel (Task 5)

```python
class AdaptiveRobustKernel:
    def fit_alpha(self, residuals: np.ndarray) -> float:
        """Alternating minimization: given residuals, find α that maximizes likelihood
        under truncated Barron loss. Range [-10, 2]."""

    def weight(self, residual: float, alpha: float) -> float:
        """Returns per-residual IRLS weight for the current α."""
```

Depends on: nothing external. Used by: Task 4's factor graph wraps each observation/relation factor in this kernel.

### Scene graph relation factors (Task 6)

```python
class RelationFactor(Protocol):
    """Implemented per relation type (on, in, contact)."""
    def residual(self, T_parent: np.ndarray, T_child: np.ndarray,
                 shape_parent: BoundingBox,
                 shape_child: BoundingBox) -> np.ndarray:
        """Residual vector. Zero when relation is satisfied."""

    def noise(self, score: float,
              cov_parent: np.ndarray, cov_child: np.ndarray) -> np.ndarray:
        """Noise covariance composed from detection score + pose uncertainties."""
```

Also modifies `object_relation_graph.compute_spatial_relations()` to return `List[SceneGraphRelation]` with soft scores instead of boolean relations. Depends on: Task 2 posteriors (for `cov_*`). Supported by: existing bbox computation in `object_relation_graph.py`.

### Two-tier orchestrator (Task 7)

Not a new module — the main loop in `data_demo.py` / `realtime_app.py`. Responsibilities:

```python
# Per frame:
slam = slam_backend.step(rgb, depth, movable_mask, odom_prior)
manip = manip_state_machine.update(finger_d, T_ec, tracker.objects)
for obj_id in observed_this_frame:
    ekfs[obj_id].predict(dt, manip)
    ekfs[obj_id].update(observations[obj_id], slam, manip)
for obj_id in unobserved_this_frame:
    ekfs[obj_id].predict(dt, manip)

# On trigger:
if trigger_condition(manip, residuals, frame_count):
    active = select_active_set(ekfs, manip, relations)
    posteriors = pose_graph.run(slam, {i: ekfs[i].posterior for i in active},
                                 pending_observations, relations, manip, T_ew, active)
    for i, p in posteriors.items():
        ekfs[i].set_posterior(p)
```

Depends on: all other tasks. Currently: main loops exist and supply most inputs; manipulation state machine already present; missing piece is the EKF and pose graph substitutions.

---

## Compatibility with existing modules

| Needed input | Source in current code | Task that exposes it cleanly |
|---|---|---|
| `T_wb`, `Σ_wb` | Currently raw `T_cw` from rosbag, no covariance | Task 1 wraps SLAM behind the interface; covariance may need a placeholder until the chosen SLAM reports it |
| movable mask | `masks` list in `associate_by_id` | Task 1 collects before SLAM call |
| Observation | Partially in `associate_by_id`'s ICP path | Task 2 formalizes the dataclass |
| Manipulation state | Gripper state machine in `data_demo.py` | Already exists, just needs exposure |
| BoundingBox per object | `object_relation_graph.py` computes it inline | Task 6 promotes to a shared type |
| `T_ew`, `T_oe` | `update_obj_pose_ee()` uses them | Already available |

No blocking gaps. Covariance from the existing SLAM (AMCL / rosbag poses) is the only missing input — can be stubbed with a reasonable constant and upgraded later.

---

## Minimality Audit Against Existing Code

Cross-checked each task against the current implementation. Summary: **the originally stated effort estimates hold**, but several tasks are even smaller than listed because the existing code already has most structural support. One item (Task 3) is subsumed by Task 2 and not a separate module.

### Task 1 — Layer separation

- `data_demo.py:678–703` already builds `depth_bg, color_bg` with movable-object masks zeroed out. This is effectively the movable-mask exclusion we need, but it happens *after* pose consumption, not before.
- **Minimal change:** move the mask-out step to before any pose computation; wrap SLAM behind a `SlamBackend` protocol; drop or restrict `refine_camera_pose()`.
- **Revised effort:** ~20 lines, not 30.

### Task 2 — Per-object EKF

- `SceneObject` at `scene/scene_object.py:73–124` already has `pose_init`, `pose_cur`, `T_oe`, `pose_uncertain`, `_score_sum`. Adding `pose_cov` is literally one new field.
- The EKF update call site is `update_obj_pose_ee()` + `update_obj_pose_icp()` + the integrate branch inside `associate_by_id()`. All three already receive enough information to compute covariance.
- **Minimal change:** new file `pose_update/ekf_se3.py`, add `pose_cov` field, call `ekf_predict`/`ekf_update` at the existing update sites. `pose_uncertain` stays as a derived property.
- **Revised effort:** ~90 new lines + ~10 modified (down from 100+15).

### Task 3 — Base-frame fusion

- This is a one-branch decision inside Task 2's `ekf_update`. It is not a separable task.
- **Revised plan:** merge into Task 2. Remove Task 3 as a standalone item.

### Task 4 — Joint pose graph

- `camera_pose_refiner.py` already does partial joint reasoning via ICP against multi-object clouds. It's 337 lines of the closest-existing approximation to what we want.
- **Decision:** replace `refine_camera_pose()` wholesale with the new `PoseGraphOptimizer`, rather than extending it. The old function's source stays as reference.
- **Revised effort:** ~150 new lines, ~15 modified (unchanged).

### Task 5 — Adaptive robust kernel

- No existing approximation in the codebase. Entirely new.
- **Unchanged:** ~60 new lines.

### Task 6 — Relation factors

- `object_relation_graph.py:29–163` (`compute_spatial_relations`) and `335–361` (`get_relation_graph`) already compute bounding boxes, overlap ratios, and return per-relation structure. Currently discards the soft scores by thresholding.
- **Minimal change:** expose the overlap ratio as a `score` field on the returned relations; add factor classes in the new `factor_graph.py`.
- **Revised effort:** ~60 new lines in factor graph + ~20 modified in relation graph (down from 80+40).

### Task 7 — Orchestration

- `data_demo.py` main loop at lines 305+ already has the state machine and per-frame structure. The integration is substitution of call sites, not restructuring.
- **Unchanged:** ~40 lines modified.

### Net revised budget

| Task | New lines | Modified lines | Notes |
|---|---|---|---|
| 1. Layer separation | 0 | ~20 | Reuses existing mask-out logic |
| 2. Per-object EKF (absorbs Task 3) | ~90 | ~10 | Reuses `SceneObject` fields |
| 4. Joint pose graph | ~150 | ~15 | Replaces `refine_camera_pose` |
| 5. Adaptive kernel | ~60 | 0 | New |
| 6. Relation factors | ~60 | ~20 | Extends existing `compute_spatial_relations` |
| 7. Orchestration | 0 | ~40 | Substitute call sites |
| 8. Verification | ~230 | 0 | Tests + visualization |

**Total:** ~590 new lines, ~105 modified (down from 640 + 140). Task 3 no longer standalone; Tasks 1 and 6 smaller than originally estimated.

---

## Unit Test Plan

Each test module grounds in fixtures extractable from the existing `apple_bowl_2` trajectory under `Mobile_Manipulation_on_Fetch/multi_objects/apple_bowl_2/`. Tests follow the existing `tests/test_api.py` pattern with `requires_data` markers for tests needing the trajectory.

### `tests/test_ekf_se3.py` — Task 2

Purely synthetic, no data dependency.

| Test | Setup | Expected |
|---|---|---|
| `test_exp_log_roundtrip` | Random twist `ξ` in se(3) | `se3_log(se3_exp(ξ))` ≈ ξ |
| `test_predict_grows_cov` | Static object, predict 10 steps | Cov trace monotonically increases by process noise |
| `test_update_with_perfect_obs_shrinks_cov` | Predict then update with R=ε | Posterior cov ≈ ε, mean matches observation |
| `test_update_with_large_R_is_noop` | Predict then update with R=10⁶·I | Posterior ≈ prior |
| `test_manipulation_process_noise` | Same object, toggle manipulation phase | Q in grasping >> Q in idle stable >> Q in long-term static |
| `test_beta_bernoulli_label` | Feed 10 high-score "cup" and 3 high-score "bowl" | label_belief["cup"] mean > 0.8; label_belief["bowl"] mean < 0.5 |
| `test_shared_error_fusion_base_frame` | Mock SLAM drift: apply same `T_wb` error to prior and observation | Posterior cov does not shrink below `Σ_wb` lower bound |
| `test_shared_error_fusion_world_frame` | Independent observations, different base poses | Posterior cov shrinks as expected |

### `tests/test_layer1_interface.py` — Task 1

Uses real trajectory RGBD + masks.

| Test | Setup | Expected |
|---|---|---|
| `test_movable_mask_removed_from_depth` | Frame 0 depth + 5 object masks | Output depth has zeros at all mask pixels |
| `test_slam_backend_protocol_signature` | Mock backend | Returns `PoseEstimate` with correct shapes |
| `test_slam_cov_is_lower_bound_on_object_cov` | Run EKF for 20 frames with constant Σ_wb | Every object's world-frame cov trace ≥ trace(Σ_wb) |

### `tests/test_pose_graph.py` — Task 4

Mix of synthetic and real data.

| Test | Setup | Expected |
|---|---|---|
| `test_single_object_graph_equals_ekf` | 1 object, 1 observation, no relations | Pose graph posterior ≈ EKF posterior within tolerance |
| `test_containment_pulls_inconsistent_poses` | Place apple 10cm outside bowl, add "apple in bowl" factor | Apple's optimized pose is inside bowl |
| `test_manipulation_factor_transports_contained_child` | Held bowl with apple in it; move bowl by 50cm | Apple's optimized pose moves by ≈50cm |
| `test_fixed_camera_invariant` | Run optimization; check camera pose in output | Camera pose equals input `T_wb` exactly (not a free variable) |
| `test_active_set_excludes_stale_objects` | 10 objects, only 2 recently observed | Only 2 + scene-graph-neighbors appear as variables |
| `test_slam_uncertainty_inflates_observation_noise` | Same observation, increasing Σ_wb | Observation factor's effective noise grows monotonically |
| `test_real_apple_bowl_optimization_converges` | Load 5 frames from `apple_bowl_2` | Optimization completes; residuals decrease monotonically |

### `tests/test_adaptive_kernel.py` — Task 5

Purely synthetic.

| Test | Setup | Expected |
|---|---|---|
| `test_alpha_near_2_for_gaussian_residuals` | Residuals ~ N(0, 1) | Estimated α ≈ 2 |
| `test_alpha_negative_for_heavy_tails` | 95% residuals N(0,1), 5% residuals N(0, 100) | Estimated α < 0 |
| `test_weight_monotone_in_residual` | Fixed α | Weight is non-increasing with \|r\| |
| `test_weight_zero_for_extreme_outlier` | α = -10, r = 100c | Weight ≈ 0 |
| `test_alpha_inlier_outlier_decision` | Inlier set + one clear outlier | Outlier receives ≥10× lower weight than inliers |

### `tests/test_relation_factors.py` — Task 6

Mix of synthetic and real.

| Test | Setup | Expected |
|---|---|---|
| `test_compute_spatial_relations_returns_scores` | Two objects with 80% bbox overlap | `score` in returned relation equals overlap ratio |
| `test_on_factor_zero_residual_when_stacked` | Two unit cubes, A bottom touches B top | Residual ≈ 0 |
| `test_on_factor_positive_residual_when_floating` | A is 10cm above B top | Residual = 0.10 |
| `test_in_factor_zero_when_centered` | A at B's center, A smaller | Residual ≈ 0 |
| `test_in_factor_positive_when_outside` | A outside B's bbox | Residual > 0 |
| `test_noise_grows_with_pose_covariance` | Same relation, increasing `cov_parent` | Factor noise trace grows |
| `test_real_apple_in_bowl_detected` | Load `apple_bowl_2` frame 0 | "in" relation between apple and bowl with score > 0.3 |

### `tests/test_orchestration.py` — Task 7

End-to-end with real trajectory.

| Test | Setup | Expected |
|---|---|---|
| `test_full_pipeline_runs_on_trajectory` | Load 20 frames of `apple_bowl_2` | Completes without exception, all objects have finite poses |
| `test_held_object_pose_tracks_ee_during_holding` | Trajectory frames 50–300 | Held object's base-frame pose stays within 2cm of `T_be @ T_oe` |
| `test_no_drift_after_release` | Trajectory frames 300–327 | Apple's world pose standard deviation over last 10 frames < 3cm |
| `test_scene_graph_contains_apple_in_bowl` | At final frame | Relations contain "apple in bowl" with score > 0.5 |
| `test_outer_loop_triggered_on_grasp_release` | Count outer-loop invocations | Invoked at grasp onset and at release, plus periodic triggers |
| `test_ekf_absorbs_pose_graph_updates` | Before and after outer loop | EKF means match pose graph posteriors after absorption |

### Shared fixtures

Add to `tests/conftest.py`:

```python
@pytest.fixture
def synthetic_pose():
    """Random SE(3) pose with small covariance."""

@pytest.fixture
def mock_slam():
    """Deterministic fake SlamBackend returning fixed (T_wb, Σ_wb)."""

@pytest.fixture
def apple_bowl_frames(n_frames=20):
    """Load first n_frames of apple_bowl_2 with RGB, depth, camera poses, detections."""

@pytest.fixture
def grasped_bowl_with_apple():
    """Synthetic: 2-object scene with bowl held by EE and apple in bowl, positions known."""
```

Fixtures are reused across test files to keep the data-loading path consistent with `tests/test_api.py`.

---

## Task 1 — Fix Layer 1/Layer 2 separation

**Goal:** Movable objects must not poison SLAM input.

**Changes:**
- In SLAM pre-processing, mask out movable-object pixels from the depth image before passing to Layer 1.
- Remove `refine_camera_pose()`'s use of movable-object clouds as camera-pose landmarks. Either drop this refinement entirely or restrict it to static-only references.
- Define a fixed Layer 1 interface: input = masked RGBD + prior pose; output = `(T_wb, Σ_wb)`.

**Depends on:** nothing.
**Enables:** Tasks 2, 3, 5.
**Estimated effort:** ~30 lines modified in `data_demo.py` / `camera_pose_refiner.py`.

---

## Task 2 — Per-object EKF on SE(3)

**Goal:** Replace the deterministic `pose_cur` + boolean `pose_uncertain` with a proper posterior `(T_o, Σ_o)`.

**Changes:**
- New file `pose_update/ekf_se3.py` (~80 lines): `se3_exp`, `se3_log`, `ekf_predict`, `ekf_update`, `pose_entropy`.
- Add `pose_cov` (6×6) field to `SceneObject`. Remove `pose_uncertain` boolean; replace with property derived from covariance.
- Observation noise is composed additively: `R_eff = R_ICP + J·Σ_wb·J^T + ρ(r)·R_robust`.
- Process noise `Q_t` between observations is set by manipulation state:
  - Manipulated (held/grasping/releasing): huge
  - Just released, not re-observed: moderate
  - Unobserved + stable history: near zero
  - Unobserved + unstable history: small nonzero
- Label tracking: replace raw score sum with Beta-Bernoulli update per label.

**Depends on:** nothing.
**Enables:** Tasks 3, 4, 5.
**Estimated effort:** ~100 new lines + ~15 modified across `scene_object.py`, `id_associator.py`, `object_pose_updater.py`.

---

## Task 3 — Base-frame fusion for held objects

**Goal:** Prevent Kalman overconfidence when the EE prior and the camera observation share `T_wb`.

**Changes:**
- In the EKF update, check the object's manipulation state:
  - HOLDING / GRASPING / RELEASING target: fuse in **base frame** (`T_bo^EE` vs `T_bo^cam`), then project result to world frame with proper `Σ_wb` inflation.
  - All other cases: fuse in world frame as normal.
- One branch in the EKF update call, no new files.

**Depends on:** Task 2.
**Enables:** Task 4, 5.
**Estimated effort:** ~20 lines in the EKF update path.

---

## Task 4 — Joint pose graph over movable objects

**Goal:** Jointly optimize all movable object poses with scene graph relations as constraints. The camera pose is a fixed parameter, not a variable.

**Changes:**
- New file `pose_update/factor_graph.py` (~150 lines) using GTSAM.
- Variables: `{T_o^k}` for objects in the active set.
- Factors:
  - Object prior from current EKF posterior.
  - Observation factors from accumulated ICP measurements since last optimization, with `R_eff` as noise.
  - Scene graph relation factors (containment, support, contact). Wrap each in an adaptive robust kernel (Barron generalized loss).
  - Manipulation rigid-attachment factors during HOLDING, with noise matching base localization uncertainty (not kinematic precision).
- Active set policy: recently observed objects + manipulated object + their scene-graph neighbors. Others are fixed priors.
- Triggers: grasp/release event, scene-graph topology change, adaptive kernel detects large residuals, periodic safety-net.
- After optimization, write posteriors back to EKF state.

**Depends on:** Task 2.
**Enables:** Task 7.
**Estimated effort:** ~150 new lines + ~15 modified in `data_demo.py`.
**Dependency:** `pip install gtsam`.

---

## Task 5 — Adaptive robust kernel

**Goal:** Automatically downweight outlier observations and wrong scene graph relations.

**Changes:**
- New file `pose_update/adaptive_kernel.py` (~60 lines) implementing Barron generalized loss with truncated partition function.
- Alternating minimization: given current residuals, estimate `α`; given `α`, run the standard factor graph optimization.
- Used inside Task 4's factor graph (wraps observation factors and relation factors).
- Not used in the per-object EKF (too few residuals per object for `α` to be meaningful).

**Depends on:** Task 4.
**Enables:** better outlier handling end-to-end.
**Estimated effort:** ~60 new lines, integrated into `factor_graph.py`.

---

## Task 6 — Scene graph relation factors

**Goal:** Turn scene graph relations from downstream annotations into first-class optimization constraints.

**Changes:**
- In `object_relation_graph.py`, replace boolean relations with soft scores derived from bounding box overlap + pose covariance.
- Each relation type gets a factor class in `factor_graph.py`:
  - Containment: A's center should lie inside B's interior volume.
  - Support: A's bottom face should be at B's top face z.
  - Contact: A and B's surfaces should be within tolerance.
- Factor noise is composed of: relation detection score, pose covariance of both objects, shape uncertainty from TSDF weight.

**Depends on:** Task 4.
**Enables:** manipulation-aware propagation falls out of optimization.
**Estimated effort:** ~80 new lines in `factor_graph.py` + ~40 modified in `object_relation_graph.py`.

---

## Task 7 — Two-tier orchestration

**Goal:** Wire the fast EKF loop and the slow pose graph loop into a coherent pipeline.

**Changes:**
- In the main loop of `data_demo.py` / `realtime_app.py`:
  - Every frame: run per-object EKF updates for observed objects; predict-only for others.
  - On trigger: run joint pose graph optimization; write posteriors back to EKFs.
- Remove the current per-frame `refine_camera_pose()` call (handled by Task 1).
- Move ICP quality metrics (fitness, RMSE) from local variables to outputs that feed the EKF and the pose graph factors.

**Depends on:** Tasks 2, 4.
**Enables:** full system.
**Estimated effort:** ~40 lines modified in `data_demo.py`.

---

## Task 8 — Verification

**Changes:**
- Extend `tests/visualize_full.py` to show per-object covariance ellipses in the top-down panel.
- Run on the `apple_bowl_2` trajectory with and without each task enabled. Compare:
  - Reconstruction drift during HOLDING.
  - Scene graph stability across frames.
  - Convergence time after release.
- Add unit tests for the EKF (Task 2) and the factor graph (Task 4) using the existing `tests/test_api.py` pattern.

**Depends on:** all other tasks.
**Estimated effort:** ~80 lines added to visualization + ~150 lines of tests.

---

## Original Budget (Superseded by "Net revised budget" above)

| Task | New lines | Modified lines | New dep | Depends on |
|---|---|---|---|---|
| 1. Layer separation | 0 | ~30 | none | — |
| 2. Per-object EKF | ~100 | ~15 | none | — |
| 3. Base-frame fusion | ~20 | 0 | none | 2 |
| 4. Joint pose graph | ~150 | ~15 | gtsam | 2 |
| 5. Adaptive kernel | ~60 | 0 | none | 4 |
| 6. Relation factors | ~80 | ~40 | none | 4 |
| 7. Orchestration | 0 | ~40 | none | 2, 4 |
| 8. Verification | ~230 | 0 | pytest | all |

**Total:** ~640 new lines, ~140 modified, 1 new dependency (`gtsam`).

Tasks 1, 2, 3, 5, 6 are independently valuable and can land in any order respecting their dependencies. Task 4 is the largest single chunk. Task 7 is integration.
