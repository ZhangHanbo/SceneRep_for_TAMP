"""Verify the YAML configs match the in-code defaults exactly.

This is the "configured params work as expected" check:

  * Loading ``ekf_tracker/configs/default.yaml`` and building a
    ``BernoulliConfig`` from it must produce a dataclass that is
    field-for-field identical to ``_default_bernoulli_cfg(K, image_shape)``
    in ``ekf_tracker/api.py`` --- the production-default constructor.

  * Loading ``configs/ekf_tracker/customization.yaml`` (which inherits via
    ``_extends:``) must overlay only the listed fields and inherit
    everything else from the canonical default; the resulting
    ``BernoulliConfig`` should also equal the production default,
    because the tuning YAML re-asserts the same values for the
    non-redundant subset.

  * ``TriggerConfig`` from the ``trigger:`` section of either YAML must
    match the in-code defaults of the ``TriggerConfig`` dataclass.

The tests are pure (no network, no servers, no external data).
"""
from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────
# Heavy-dependency isolation
#
# The full ``ekf_tracker.api`` import chain pulls in ``gtsam`` (slow-tier
# factor graph) and a handful of other optional perception dependencies
# that are not needed to verify config-loading semantics. Stub them so
# the test runs in a minimal numpy-only environment.
# ─────────────────────────────────────────────────────────────────────

class _Stub:
    def __getattr__(self, _):  # pragma: no cover
        return _Stub()
    def __call__(self, *a, **kw):  # pragma: no cover
        return _Stub()


# Stub only modules truly absent from minimal envs. scipy / open3d /
# cv2 / PIL / matplotlib are available; stubbing them breaks real
# submodule imports (e.g. `from scipy.ndimage import binary_erosion`).
sys.modules.setdefault("gtsam", _Stub())


# Import only what we need, by file location, to skip ekf_tracker/__init__.py.

def _load_module_by_path(qualname: str, path: Path):
    spec = importlib.util.spec_from_file_location(qualname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[qualname] = mod
    spec.loader.exec_module(mod)
    return mod


# ekf_tracker.config holds the dataclasses; numpy-only deps.
_config_mod = _load_module_by_path(
    "ekf_tracker.config", ROOT / "ekf_tracker" / "config.py")
BernoulliConfig = _config_mod.BernoulliConfig
TriggerConfig = _config_mod.TriggerConfig

# Establish ekf_tracker as a package alias so ekf_tracker.configs can
# resolve the `from ekf_tracker.config import BernoulliConfig` line.
import types
_pkg = types.ModuleType("ekf_tracker")
_pkg.__path__ = [str(ROOT / "ekf_tracker")]
sys.modules.setdefault("ekf_tracker", _pkg)

_configs_mod = _load_module_by_path(
    "ekf_tracker.configs", ROOT / "ekf_tracker" / "configs" / "__init__.py")
DEFAULT_PATH = _configs_mod.DEFAULT_PATH
load_config = _configs_mod.load_config
to_bernoulli_config = _configs_mod.to_bernoulli_config
to_trigger_config = _configs_mod.to_trigger_config


# ─────────────────────────────────────────────────────────────────────
# Inline construction of the production-default BernoulliConfig.
# After Phase C strips dataclass defaults, every field is required ---
# this is the canonical reference the YAML must reproduce bit-exactly.
# Fields previously-defaulted-but-dead (r_conf, enable_visibility,
# G_in_rot, self_merge_d2_trans, icp_*, r_held_min_match_frames,
# relation_*, gravity_predict, workspace_floor_z) have been removed
# from BernoulliConfig entirely.
# ─────────────────────────────────────────────────────────────────────

def _default_bernoulli_cfg(K: np.ndarray,
                            image_shape=(480, 640)) -> "BernoulliConfig":
    return BernoulliConfig(
        # Bernoulli existence model
        association_mode="hungarian",
        p_s=1.0, p_d=0.9, alpha=4.4,
        lambda_c=1.0, lambda_b=1.0,
        r_min=1e-3,
        # Probabilistic gates
        G_in=12.59, G_out=25.0,
        G_in_trans=7.815,
        G_out_trans=21.108, G_out_rot=21.108,
        gate_mode="trans", cost_d2_mode="sum",
        max_residual_m=0.30,
        # Saturation cap / floor
        P_max=np.diag([0.25**2] * 3 + [(np.pi / 4) ** 2] * 3),
        P_min_diag=np.array([0.005**2] * 3 + [0.05**2] * 3),
        # Robust / label switches
        enable_huber=True,
        init_cov_from_R=False,
        enforce_label_match=False,
        hungarian_label_penalty=6.0,
        hungarian_score_weight=2.0,
        # Subpart suppression
        dedup_voxel_size_m=0.02,
        dedup_containment_thresh=0.8,
        dedup_require_same_label=False,
        # Birth gates
        birth_border_margin_px=2,
        birth_confirm_k=3,
        birth_score_min=0.20,
        birth_fitness_min=0.5,
        birth_rmse_max=0.02,
        birth_pending_ttl_frames=30,
        birth_min_dist_m=0.05,
        # Held-track anchoring
        held_birth_radius_m=0.25,
        held_meas_radius_m=0.25,
        held_meas_innov_max_m=0.20,
        r_held_floor=0.5,
        # Self-merge
        self_merge_trans_m=0.05,
        # Scenario-specific runtime
        K=K, image_shape=image_shape, T_bc=None,
    )


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _assert_field_equal(name, a, b):
    """Equality check that handles ndarray fields."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if a is None and b is None:
            return
        assert a is not None and b is not None, (
            f"{name}: only one is None  yaml={a!r}  code={b!r}")
        assert np.array_equal(a, b), (
            f"{name}: ndarrays differ\n  yaml={a}\n  code={b}")
    else:
        assert a == b, f"{name}: yaml={a!r}  code={b!r}"


def _compare_all_fields(yaml_obj, code_obj):
    for f in dataclasses.fields(yaml_obj):
        _assert_field_equal(
            f.name,
            getattr(yaml_obj, f.name),
            getattr(code_obj, f.name),
        )


_K_DUMMY = np.array(
    [[554.3827, 0.0, 320.5],
     [0.0, 554.3827, 240.5],
     [0.0, 0.0,      1.0]],
    dtype=np.float64,
)
_IMAGE_SHAPE = (480, 640)


# ─────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────

def test_default_path_resolves():
    assert DEFAULT_PATH.exists(), f"default.yaml not found at {DEFAULT_PATH}"


def test_default_yaml_matches_production_defaults():
    """default.yaml + to_bernoulli_config should equal _default_bernoulli_cfg."""
    cfg = load_config()
    yaml_bcfg = to_bernoulli_config(cfg, K=_K_DUMMY, image_shape=_IMAGE_SHAPE)
    code_bcfg = _default_bernoulli_cfg(_K_DUMMY, _IMAGE_SHAPE)
    _compare_all_fields(yaml_bcfg, code_bcfg)


def test_tuning_yaml_inherits_defaults():
    """customization.yaml is a non-redundant subset that re-asserts defaults;
    after _extends merge the result must equal the production default."""
    tuning_path = ROOT / "configs" / "ekf_tracker" / "customization.yaml"
    assert tuning_path.exists(), f"customization.yaml not found at {tuning_path}"
    cfg = load_config(tuning_path)
    yaml_bcfg = to_bernoulli_config(cfg, K=_K_DUMMY, image_shape=_IMAGE_SHAPE)
    code_bcfg = _default_bernoulli_cfg(_K_DUMMY, _IMAGE_SHAPE)
    _compare_all_fields(yaml_bcfg, code_bcfg)


def test_tuning_yaml_overrides_take_effect():
    """Sanity: a manual override applied through the loader actually
    propagates into the final BernoulliConfig (i.e., merge isn't a no-op)."""
    cfg = load_config()
    # Pretend the user tuned r_min by editing a child YAML.
    cfg["bernoulli"]["r_min"] = 5.0e-4
    bcfg = to_bernoulli_config(cfg, K=_K_DUMMY, image_shape=_IMAGE_SHAPE)
    assert bcfg.r_min == 5.0e-4


def test_trigger_config_matches_ekftracker_implicit_default():
    """trigger: section of default.yaml mirrors what EkfTracker.__init__
    constructs when no explicit `trigger=` is supplied (api.py:151-153):
    a never-fires TriggerConfig (all event triggers off, periodic -1).

    This is the production default; the dataclass-only TriggerConfig()
    has all triggers ON, which the EkfTracker constructor explicitly
    overrides --- so the YAML mirrors the latter, not the former.
    """
    cfg = load_config()
    yaml_tcfg = to_trigger_config(cfg)
    code_tcfg = TriggerConfig(
        on_grasp=False, on_release=False,
        on_new_object=False, periodic_every_n_frames=-1,
    )
    assert yaml_tcfg.on_grasp == code_tcfg.on_grasp
    assert yaml_tcfg.on_release == code_tcfg.on_release
    assert yaml_tcfg.on_new_object == code_tcfg.on_new_object
    assert yaml_tcfg.periodic_every_n_frames == code_tcfg.periodic_every_n_frames


def test_no_dead_bernoulli_fields_in_yaml():
    """Sanity: the YAML must NOT contain any of the 20 pruned dead
    BernoulliConfig fields. Accidentally re-introducing one is silently
    a no-op at runtime, which is the misleading state we removed."""
    DEAD_BERNOULLI_FIELDS = {
        "r_conf", "enable_visibility", "G_in_rot", "self_merge_d2_trans",
        "icp_method", "icp_min_fitness", "icp_max_rmse",
        "icp_centroid_fallback_rot_var", "icp_centroid_fallback_trans_std",
        "r_held_min_match_frames",
        "relation_backend", "relation_server_url", "relation_llm_model",
        "relation_score_threshold", "relation_every_n_frames",
        "relation_on_grasp", "relation_on_release", "relation_on_new_object",
        "gravity_predict", "workspace_floor_z",
    }
    DEAD_TRIGGER_FIELDS = {"residual_threshold"}

    for path_label, path in [
        ("default", DEFAULT_PATH),
        ("tuning", ROOT / "configs" / "ekf_tracker" / "customization.yaml"),
    ]:
        # Read the *raw* YAML (no _extends merge) to check what's
        # explicitly written in each file.
        import yaml as _yaml
        with open(path, "r") as f:
            raw = _yaml.safe_load(f) or {}
        raw.pop("_extends", None)
        raw_b = set((raw.get("bernoulli") or {}).keys())
        raw_t = set((raw.get("trigger") or {}).keys())
        leaked_b = DEAD_BERNOULLI_FIELDS & raw_b
        leaked_t = DEAD_TRIGGER_FIELDS & raw_t
        assert not leaked_b, (
            f"{path_label}.yaml leaks dead BernoulliConfig fields: {sorted(leaked_b)}")
        assert not leaked_t, (
            f"{path_label}.yaml leaks dead TriggerConfig fields: {sorted(leaked_t)}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
