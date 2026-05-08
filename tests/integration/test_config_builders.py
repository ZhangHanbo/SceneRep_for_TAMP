"""Bit-exactness contract: every Tier-3 builder's output must equal the
current in-code value for that subsystem.

This is the safety net for the config-first refactor. Each test loads
the canonical default YAML, calls one builder, and asserts that the
returned value exactly equals the production constant / default kwarg /
class attribute it mirrors. Any drift between the YAML and the code
fails fast here --- before the parity tests even run.

Heavy runtime dependencies (gtsam, open3d, cv2, matplotlib, PIL) are
stubbed via ``sys.modules`` so this test can run in a minimal numpy +
PyYAML environment. The stubs are activated only for module-import; no
runtime call uses the stubbed objects.
"""
from __future__ import annotations

import importlib.util
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────
# Stub heavy deps so the EKF-tracker subpackage modules import cleanly
# ─────────────────────────────────────────────────────────────────────

class _Stub:
    def __getattr__(self, _):
        return _Stub()
    def __call__(self, *a, **kw):
        return _Stub()


# Only stub modules that are not installed in the minimal env. scipy /
# open3d / cv2 / PIL are available system-wide; we use them for real.
# ``gtsam`` is the slow-tier backend and is the only blocker.
sys.modules.setdefault("gtsam", _Stub())


# Lazy import of the package via importlib to avoid pulling
# ekf_tracker/__init__.py (which imports api.py → orchestrator_gaussian →
# factor_graph → gtsam). We only need the dataclasses + builders.

def _load_module_by_path(qualname: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(qualname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


# Establish ``ekf_tracker`` as an empty package so subpackage imports
# work without triggering api.py.
_pkg = types.ModuleType("ekf_tracker")
_pkg.__path__ = [str(ROOT / "ekf_tracker")]
sys.modules.setdefault("ekf_tracker", _pkg)

# Same for ``perception``, ``utils``, ``ekf_tracker.manipulation``,
# ``ekf_tracker.relations``.
for _pkgname, _pkgpath in [
    ("perception", ROOT / "perception"),
    ("utils", ROOT / "utils"),
    ("ekf_tracker.manipulation", ROOT / "ekf_tracker" / "manipulation"),
    ("ekf_tracker.relations", ROOT / "ekf_tracker" / "relations"),
    ("ekf_tracker.state", ROOT / "ekf_tracker" / "state"),
]:
    if _pkgname not in sys.modules:
        _p = types.ModuleType(_pkgname)
        _p.__path__ = [str(_pkgpath)]
        sys.modules[_pkgname] = _p


# Load the dataclasses and builders we need.
_config_mod = _load_module_by_path(
    "ekf_tracker.config", ROOT / "ekf_tracker" / "config.py")
BernoulliConfig = _config_mod.BernoulliConfig
TriggerConfig = _config_mod.TriggerConfig

_configs_mod = _load_module_by_path(
    "ekf_tracker.configs", ROOT / "ekf_tracker" / "configs" / "__init__.py")
load_config = _configs_mod.load_config
to_bernoulli_config = _configs_mod.to_bernoulli_config
to_trigger_config = _configs_mod.to_trigger_config


# Load the Tier-3 source modules to extract their in-code defaults.
_ekf_se3 = _load_module_by_path(
    "utils.ekf_se3", ROOT / "utils" / "ekf_se3.py")

_object_dynamics = _load_module_by_path(
    "utils.object_dynamics", ROOT / "utils" / "object_dynamics.py")

_voxel_obs = _load_module_by_path(
    "perception.voxel_observability",
    ROOT / "perception" / "voxel_observability.py")

_visibility = _load_module_by_path(
    "perception.visibility", ROOT / "perception" / "visibility.py")

_det_dedup = _load_module_by_path(
    "perception.det_dedup", ROOT / "perception" / "det_dedup.py")

_birth_gating = _load_module_by_path(
    "perception.birth_gating", ROOT / "perception" / "birth_gating.py")

_grasp_owner = _load_module_by_path(
    "ekf_tracker.manipulation.grasp_owner_detector",
    ROOT / "ekf_tracker" / "manipulation" / "grasp_owner_detector.py")

_relation_utils = _load_module_by_path(
    "ekf_tracker.relations.relation_utils",
    ROOT / "ekf_tracker" / "relations" / "relation_utils.py")

# RelationFilter imports factor_graph (gtsam-stubbed), so this should
# succeed via the stub.
try:
    _relation_filter = _load_module_by_path(
        "ekf_tracker.relations.relation_filter",
        ROOT / "ekf_tracker" / "relations" / "relation_filter.py")
except Exception:
    _relation_filter = None

# Gripper FSM imports the GraspOwnerDetector (already loaded).
try:
    _gripper_state = _load_module_by_path(
        "utils.gripper_state",
        ROOT / "utils" / "gripper_state.py")
except Exception:
    _gripper_state = None

# Gravity predict imports voxel_observability + object_dynamics.
try:
    _gravity_predict = _load_module_by_path(
        "ekf_tracker.manipulation.gravity_predict",
        ROOT / "ekf_tracker" / "manipulation" / "gravity_predict.py")
except Exception:
    _gravity_predict = None

# PoseEstimator imports open3d --- stubbed --- and may fail.
try:
    _icp_pose = _load_module_by_path(
        "perception.icp_pose",
        ROOT / "perception" / "icp_pose.py")
except Exception:
    _icp_pose = None


# Defer loading gaussian_ekf_tracker and orchestrator_gaussian; both
# depend on factor_graph (gtsam). We only need their module constants;
# read them via small file-level inspection to bypass the import.

def _read_module_constants(path: Path,
                            names: tuple[str, ...]) -> dict:
    """Run only the assignments we care about by exec'ing a stripped
    snippet. Avoids transitively pulling gtsam.
    """
    src = path.read_text()
    # Take everything up to the first ``class`` definition --- module
    # constants live at the top of the file.
    cut = src.find("\nclass ")
    if cut > 0:
        src = src[:cut]
    namespace: dict = {"np": np}
    # Minimal subset of imports the constants need.
    exec(compile(src, str(path), "exec"), namespace)
    return {n: namespace[n] for n in names if n in namespace}


# Module constants for D8/D9 were stripped --- nothing left to read.


# ─────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_config()


@pytest.fixture(scope="module")
def K() -> np.ndarray:
    return np.array([[554.3827, 0.0, 320.5],
                     [0.0, 554.3827, 240.5],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _get_default(func, kwarg: str):
    sig = inspect.signature(func)
    return sig.parameters[kwarg].default


def _eq(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(a, b)
    return a == b


# ─────────────────────────────────────────────────────────────────────
# Tier-3 builder bit-exactness tests
# ─────────────────────────────────────────────────────────────────────

def test_birth_gate_config(cfg):
    """Post-Phase-D2 BirthGateConfig has no dataclass defaults; the YAML
    is the source of truth. Compare against the literal golden values
    that were the in-code default before the strip.
    """
    yaml_built = _configs_mod.build_birth_gate_config(cfg)
    assert yaml_built.birth_min_dist_m == 0.05
    assert yaml_built.held_birth_radius_m == 0.25


def test_process_noise_schedule(cfg):
    """Post-Phase-D7 the module-level Q_* constants and 5/50 thresholds
    are removed; the YAML is the source of truth. Compare against
    literal goldens.
    """
    sch = _configs_mod.build_process_noise_schedule(cfg)

    def _diag6(v):
        return np.diag([v] * 6)

    assert np.array_equal(sch["Q_static_stable"],   _diag6(1e-8))
    assert np.array_equal(sch["Q_static_unstable"], _diag6(1e-5))
    assert np.array_equal(sch["Q_idle"],            _diag6(1e-6))
    assert np.array_equal(sch["Q_just_released"],   _diag6(1e-3))
    # _Q_GRASPING_RELEASING = diag([0.02**2]*3 + [0.10**2]*3); IEEE
    # 754 squares 0.10 to 0.010000000000000002 (one ULP above 0.01).
    assert np.array_equal(
        sch["Q_grasping_releasing"],
        np.diag([0.02**2] * 3 + [0.10**2] * 3))
    assert np.array_equal(
        sch["Q_holding_base_frame"],
        np.diag([2.5e-4] * 3 + [1.0e-3] * 3))
    assert np.array_equal(sch["Q_held_world_frame"], _diag6(1.0e-4))

    assert sch["frames_unstable_threshold"] == 5
    assert sch["frames_stable_threshold"]   == 50


def test_fast_tier_noise_config(cfg):
    """Post-Phase-D8/D9 the four fast-tier-noise module constants are
    removed (two were dead, two moved to YAML). The two retained
    runtime-consumed values mirror their pre-strip values exactly.
    """
    fn = _configs_mod.build_fast_tier_noise_config(cfg)
    assert fn["centroid_r_cam_std_m"] == 0.02
    assert np.array_equal(fn["centroid_r_cam_3d"],
                          np.diag([0.02 ** 2] * 3))
    assert np.array_equal(fn["tiny_cov"], np.diag([1.0e-6] * 6))


@pytest.mark.skipif(_icp_pose is None, reason="open3d unavailable")
def test_pose_estimator_kwargs(cfg):
    PE = _icp_pose.PoseEstimator
    pe = _configs_mod.build_pose_estimator_kwargs(cfg)
    assert pe["voxel_size_m"]   == PE.VOXEL_SIZE
    assert pe["icp_threshold_m"]== PE.ICP_THRESHOLD
    assert pe["icp_max_iter"]   == PE.ICP_MAX_ITER
    assert pe["min_fitness"]    == PE.MIN_FITNESS
    assert pe["max_rmse"]       == PE.MAX_RMSE
    assert pe["trans_var_floor"] == PE.TRANS_VAR_FLOOR
    assert pe["rot_var_floor"]  == PE.ROT_VAR_FLOOR
    assert np.array_equal(pe["centroid_r"], PE.CENTROID_R_ICP)
    assert pe["ref_update_min_fitness"] == PE.REF_UPDATE_MIN_FITNESS
    assert pe["max_ref_points"]  == PE.MAX_REF_POINTS
    # _clean_mask defaults
    cm = pe["mask_clean"]
    cm_def = {n: _get_default(_icp_pose._clean_mask, n)
              for n in ("erosion_iter", "depth_edge_max_jump",
                         "min_depth", "max_depth", "min_points")}
    for k, v in cm_def.items():
        assert cm[k] == v, f"mask_clean.{k}: yaml={cm[k]!r}  code={v!r}"
    # _back_project defaults
    bp = pe["back_project"]
    bp_def = {n: _get_default(_icp_pose._back_project, n)
              for n in ("min_depth", "max_depth", "min_points")}
    for k, v in bp_def.items():
        assert bp[k] == v, f"back_project.{k}: yaml={bp[k]!r}  code={v!r}"


def test_voxel_observability_kwargs(cfg):
    """Post-Phase-D5 VoxelObservability has no __init__ defaults; the
    YAML is the source of truth. Compare against the literal golden
    values that were the production override before the strip
    (voxel_size_m=0.05; the legacy class default 0.02 was dead code).
    """
    vo = _configs_mod.build_voxel_observability_kwargs(cfg)
    assert vo["voxel_size_m"] == 0.05
    assert vo["workspace_aabb"] == ((-2.5, -2.5, -1.0), (2.5, 2.5, 2.0))
    assert vo["n_min_hit"]  == 2
    assert vo["n_min_pass"] == 3


def test_voxel_integrate_kwargs(cfg):
    """Post-Phase-D5 integrate_depth has no kwarg defaults; compare
    against literal goldens.
    """
    integ = _configs_mod.build_voxel_integrate_kwargs(cfg)
    assert integ["max_range_m"] == 3.0
    assert integ["subsample"]   == 4
    assert integ["min_depth_m"] == 0.05


def test_visibility_kwargs(cfg):
    """Post-Phase-D4 literal goldens (visibility_p_v has no kwarg defaults)."""
    vis = _configs_mod.build_visibility_kwargs(cfg)
    assert vis["max_samples_per_track"]   == 256
    assert vis["fallback_sphere_samples"] == 64
    assert vis["fallback_obj_radius"]     == 0.05
    assert vis["z_tol_abs"]               == 0.02
    assert vis["z_tol_rel"]               == 0.02
    assert vis["min_depth"]               == 0.1
    assert vis["max_depth"]               == 10.0


def test_det_dedup_kwargs(cfg):
    """Post-Phase-D3 literal goldens (voxelize_mask has no kwarg defaults)."""
    dd = _configs_mod.build_det_dedup_kwargs(cfg)
    assert dd["voxel_size"] == 0.02
    assert dd["min_depth"]  == 0.1
    assert dd["max_depth"]  == 5.0


@pytest.mark.skipif(_gripper_state is None, reason="gripper_state unavailable")
def test_gripper_phase_tracker_kwargs(cfg):
    """Post-Phase-D14 literal goldens (constructor has no defaults)."""
    gp = _configs_mod.build_gripper_phase_tracker_kwargs(cfg)
    assert gp["closed_width_m"]        == 0.025
    assert gp["open_width_m"]          == 0.040
    assert gp["close_delta_m"]         == 0.005
    assert gp["grasp_radius_m"]        == 0.30
    assert gp["history_size"]          == 5
    assert gp["motion_threshold_m"]    == 0.01
    assert gp["min_transition_frames"] == 5
    assert gp["min_inside_count"]      == 20


def test_grasp_owner_detector_kwargs(cfg):
    """Post-Phase-D15 literal goldens."""
    go = _configs_mod.build_grasp_owner_detector_kwargs(cfg)
    assert go["min_inside_count"]  == 20
    assert go["fallback_radius_m"] == 0.05
    assert go["perception_keys"]   == ("grasp_owner_pid", "is_grasped")


@pytest.mark.skipif(_gravity_predict is None, reason="gravity_predict unavailable")
def test_gravity_predict_kwargs(cfg):
    """Post-Phase-D16 predict_landing_pose has no kwarg defaults; the
    YAML is the source of truth. The eps_roughness module constant
    EPS_ROUGHNESS_DEFAULT remains as a documented value.
    """
    gp = _configs_mod.build_gravity_predict_kwargs(cfg)
    assert gp["gravity"]           == 9.81
    assert gp["workspace_floor_z"] == -1.0
    assert gp["max_drop_m"]        == 2.0
    assert gp["eps_roughness"]     == _gravity_predict.EPS_ROUGHNESS_DEFAULT
    assert gp["eps_roughness"]     == 5.0e-3


def test_object_dynamics_table(cfg):
    """Post-Phase-D1 the YAML is the source of truth (the module
    constants DEFAULT_DYNAMICS, LABEL_DYNAMICS_TABLE,
    SHAPE_FOOTPRINT_FACTOR were removed). Compare against literal
    goldens.
    """
    default, table, sff = _configs_mod.build_object_dynamics_table(cfg)

    # Default
    assert default.label    == "default"
    assert default.e        == 0.40
    assert default.mu       == 0.50
    assert default.shape    == "irregular"
    assert default.radius_m == 0.05
    assert default.mass_kg  == 0.1

    # Table --- 6-label production set.
    assert set(table.keys()) == {
        "apple", "milkbox", "cola", "cup", "pot", "flowerpot"}
    expected = {
        "apple":     dict(e=0.30, mu=0.55, shape="spherical",   radius_m=0.04),
        "milkbox":   dict(e=0.40, mu=0.45, shape="box",         radius_m=0.06),
        "cola":      dict(e=0.50, mu=0.40, shape="cylindrical", radius_m=0.04),
        "cup":       dict(e=0.40, mu=0.50, shape="cylindrical", radius_m=0.04),
        "pot":       dict(e=0.40, mu=0.50, shape="cylindrical", radius_m=0.07),
        "flowerpot": dict(e=0.30, mu=0.55, shape="cylindrical", radius_m=0.07),
    }
    for label, exp in expected.items():
        prop = table[label]
        for f, v in exp.items():
            assert getattr(prop, f) == v, f"table.{label}.{f}"
        assert prop.label    == label
        assert prop.mass_kg  == 0.1

    # Footprint factors.
    assert dict(sff) == {
        "spherical": 0.25, "cylindrical": 0.50,
        "box": 0.70, "irregular": 1.00,
    }


@pytest.mark.skipif(_relation_filter is None, reason="relation_filter unavailable")
def test_relation_filter_kwargs(cfg):
    """Post-Phase-D10 RelationFilter has no __init__ defaults; the
    prune_threshold (previously hardcoded 0.01) is now a required
    explicit kwarg.
    """
    rf = _configs_mod.build_relation_filter_kwargs(cfg)
    assert rf["alpha"]           == 0.3
    assert rf["threshold"]       == 0.5
    assert rf["prune_threshold"] == 0.01


def test_relation_trigger_config(cfg):
    """Post-Phase-D12 RelationTriggerConfig has no dataclass defaults;
    compare against literal goldens.
    """
    rt = _configs_mod.build_relation_trigger_config(cfg)
    assert rt.relation_every_n_frames == 90
    assert rt.relation_on_grasp       is True
    assert rt.relation_on_release     is True
    assert rt.relation_on_new_object  is True


def test_held_set_expansion_kwargs(cfg):
    """Post-Phase-D12 expand_held_with_relations(max_iters=...) is
    required.
    """
    he = _configs_mod.build_held_set_expansion_kwargs(cfg)
    assert he["max_iters"] == 8


@pytest.mark.skipif(_relation_filter is None, reason="relation_orchestrator unavailable")
def test_relation_orchestrator_kwargs(cfg):
    """Post-Phase-D11 RelationOrchestrator has no __init__ defaults.
    All values come from the YAML; compare against literal goldens.
    """
    ro = _configs_mod.build_relation_orchestrator_kwargs(cfg)
    assert ro["backend"]             == "llm"
    assert ro["llm_model"]           == "gpt-5.1"
    assert ro["llm_temperature"]     == 0.0
    assert ro["ema_alpha"]           == 0.3
    assert ro["ema_threshold"]       == 0.5
    assert ro["ema_prune_threshold"] == 0.01
    assert ro["score_threshold"]     == 0.5
    assert ro["rest_server_url"]     is None


def test_ekf_tracker_runtime(cfg):
    et = _configs_mod.build_ekf_tracker_runtime(cfg)
    # Mirrors the EkfTracker.__init__ defaults at ekf_tracker/api.py:124-141.
    assert et["robot_type"]  == "fetch"
    assert et["pose_method"] == "icp_chain"
    assert et["image_shape"] == (480, 640)
    assert et["default_owl_server"]  is None
    assert et["default_sam2_server"] is None


# ─────────────────────────────────────────────────────────────────────
# Strict-error contract: missing key raises KeyError with dotted path
# ─────────────────────────────────────────────────────────────────────

def test_strict_error_on_missing_top_section(cfg):
    """Removing an entire top-level section must raise ``KeyError`` with
    that section's name."""
    bad = {k: v for k, v in cfg.items() if k != "bernoulli"}
    with pytest.raises(KeyError, match=r"bernoulli"):
        to_bernoulli_config(bad, K=np.eye(3), image_shape=(480, 640))


def test_strict_error_on_missing_subsection(cfg):
    bad = {k: v for k, v in cfg.items() if k != "voxel_observability"}
    with pytest.raises(KeyError, match=r"voxel_observability"):
        _configs_mod.build_voxel_observability_kwargs(bad)


def test_strict_error_on_missing_nested_key(cfg):
    bad = dict(cfg)
    bad["voxel_observability"] = {k: v for k, v in cfg["voxel_observability"].items()
                                    if k != "n_min_hit"}
    with pytest.raises(KeyError, match=r"voxel_observability\.n_min_hit"):
        _configs_mod.build_voxel_observability_kwargs(bad)
