"""Bit-exact parity test for apple_in_the_tray.

Mirror of ``test_parity_apple_drop.py`` --- same zero-divergence
assertion, on the 700-frame apple_in_the_tray trajectory. Auto-skips
when heavy runtime deps are missing.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("PIL")
pytest.importorskip("scipy")
pytest.importorskip("cv2")
pytest.importorskip("open3d")
pytest.importorskip("gtsam")


def test_apple_in_the_tray_zero_divergence() -> None:
    from tests._parity_lib import compute_parity_stats

    stats = compute_parity_stats("apple_in_the_tray")

    assert stats["n_compared"] > 0, "no frames were compared (missing fixtures)"
    assert stats["n_oid_mismatch"] == 0, (
        f"{stats['n_oid_mismatch']} frames had oid-set mismatches: "
        f"{stats['mismatched_frames'][:5]}")
    assert stats["pose_max_err_world"] == 0.0, (
        f"pose (world) max |Δ| = {stats['pose_max_err_world']:.3e}")
    assert stats["r_max_err"] == 0.0, (
        f"r max |Δ| = {stats['r_max_err']:.3e}")
    assert stats["cov_max_err_world"] == 0.0, (
        f"cov (world) max |Δ| = {stats['cov_max_err_world']:.3e}")
