"""Bit-exact parity test for apple_drop.

Asserts that the post-refactor (config-driven) EkfTracker.step()
reproduces the cached state dumps under
``tests/visualization_pipeline/apple_drop/ekf_state/`` with **zero**
divergence in pose, covariance, and Bernoulli r.

Auto-skips when heavy runtime deps are missing (PIL/scipy/cv2/open3d/
gtsam). The CI environment with those installed runs the full 416-frame
parity loop and asserts ``pose_max_err_world == 0.0``.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Heavy runtime deps required by EkfTracker.step() and the parity loop.
pytest.importorskip("PIL")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("open3d")
pytest.importorskip("gtsam")


def test_apple_drop_zero_divergence() -> None:
    """``EkfTracker.step()`` must reproduce the cached state dumps
    bit-exactly. Any non-zero pose / r delta indicates that the
    post-refactor config-driven path no longer mirrors the baseline.
    """
    from tests._parity_lib import compute_parity_stats

    stats = compute_parity_stats("apple_drop")

    assert stats["n_compared"] > 0, "no frames were compared (missing fixtures)"
    assert stats["n_oid_mismatch"] == 0, (
        f"{stats['n_oid_mismatch']} frames had oid-set mismatches: "
        f"{stats['mismatched_frames'][:5]}")
    assert stats["pose_max_err_world"] == 0.0, (
        f"pose (world) max |Δ| = {stats['pose_max_err_world']:.3e}")
    assert stats["r_max_err"] == 0.0, (
        f"r max |Δ| = {stats['r_max_err']:.3e}")
    # Covariance is allowed to drift below 1 ULP because of accumulated
    # FP rounding in the Joseph form, but in practice the apple_drop
    # baseline is also bit-exact.
    assert stats["cov_max_err_world"] == 0.0, (
        f"cov (world) max |Δ| = {stats['cov_max_err_world']:.3e}")
