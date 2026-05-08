"""Shared dataclasses for the EKF tracker.

`BernoulliConfig` configures the Bernoulli-EKF fast tier
(see ``docs/ekf_tracker/latex/bernoulli_ekf.tex``).
`TriggerConfig` configures when the slow-tier `PoseGraphOptimizer` fires.

Two-hierarchy contract: every field is required (no dataclass defaults).
The single source of defaults is ``ekf_tracker/configs/default.yaml``;
construct via ``ekf_tracker.configs.to_bernoulli_config(load_config(),
K=..., image_shape=..., T_bc=...)`` (or its ``to_trigger_config``
counterpart). Bare ``BernoulliConfig()`` with no kwargs is a programmer
error and raises ``TypeError``.

Dead fields removed (zero runtime readers, dropped during the YAML
audit): ``r_conf``, ``enable_visibility``, ``G_in_rot``,
``self_merge_d2_trans``, ``icp_method``, ``icp_min_fitness``,
``icp_max_rmse``, ``icp_centroid_fallback_rot_var``,
``icp_centroid_fallback_trans_std``, ``r_held_min_match_frames``,
``relation_backend``, ``relation_server_url``, ``relation_llm_model``,
``relation_score_threshold``, ``relation_every_n_frames``,
``relation_on_grasp``, ``relation_on_release``, ``relation_on_new_object``,
``gravity_predict``, ``workspace_floor_z``. The live wiring path for
each lives elsewhere; see the ``default.yaml`` header.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Trigger policy (slow-tier scheduling)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class TriggerConfig:
    """Configuration for when the slow tier fires.

    Every field is required --- callers must construct via
    :func:`ekf_tracker.configs.to_trigger_config` (which reads the
    canonical ``default.yaml``) or supply every value explicitly.
    """
    on_grasp: bool
    on_release: bool
    on_new_object: bool
    periodic_every_n_frames: int


# ─────────────────────────────────────────────────────────────────────
# Bernoulli-EKF mode config (docs/ekf_tracker/latex/bernoulli_ekf.tex)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class BernoulliConfig:
    """Opts the fast tier into the Bernoulli-EKF behaviour.

    Every field is required --- callers must construct via
    :func:`ekf_tracker.configs.to_bernoulli_config` (which reads the
    canonical ``default.yaml``) or supply every value explicitly. The
    fields whose dataclass type is ``Optional[T]`` may be set to
    ``None``; all others must be a non-None value.

    See ``docs/ekf_tracker/latex/bernoulli_ekf.tex`` for the full
    derivation.
    """

    # ── Bernoulli existence model ────────────────────────────────
    association_mode: str
    p_s: float
    p_d: float
    alpha: float
    lambda_c: float
    lambda_b: float
    r_min: float

    # ── Probabilistic gates ──────────────────────────────────────
    G_in: float
    G_out: float
    G_in_trans: float
    G_out_trans: float
    G_out_rot: float
    gate_mode: str             # 'full' | 'trans' | 'trans_and_rot'
    cost_d2_mode: str          # 'full' | 'trans' | 'sum'
    max_residual_m: Optional[float]

    # ── Saturation cap / floor on P_bo (None disables) ───────────
    P_max: Optional[np.ndarray]
    P_min_diag: Optional[np.ndarray]

    # ── Robust / label switches ──────────────────────────────────
    enable_huber: bool
    init_cov_from_R: bool
    enforce_label_match: bool

    # ── Soft-mode cost augmentation ──────────────────────────────
    hungarian_label_penalty: float
    hungarian_score_weight: float

    # ── Subpart suppression ──────────────────────────────────────
    dedup_voxel_size_m: float
    dedup_containment_thresh: float
    dedup_require_same_label: bool

    # ── Birth gates ──────────────────────────────────────────────
    birth_border_margin_px: int
    birth_confirm_k: int
    birth_score_min: float
    birth_fitness_min: float
    birth_rmse_max: float
    birth_pending_ttl_frames: int
    birth_min_dist_m: float

    # ── Held-track anchoring ─────────────────────────────────────
    held_birth_radius_m: float
    held_meas_radius_m: float
    held_meas_innov_max_m: float
    r_held_floor: float

    # ── Self-merge ───────────────────────────────────────────────
    self_merge_trans_m: float

    # ── Scenario-specific runtime values (caller supplies) ───────
    K: Optional[np.ndarray]
    image_shape: Optional[Tuple[int, int]]
    T_bc: Optional[np.ndarray]
