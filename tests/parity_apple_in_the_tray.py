"""Parity test: EkfTracker public API vs visualize_ekf_tracking.main()
on apple_in_the_tray.

Mirror of ``tests/parity_apple_drop.py`` --- same loop pattern, same JSON
state-dump comparison, just for the apple_in_the_tray trajectory (700
frames). For each frame, drives ``EkfTracker.step()`` and compares its
per-track world-frame mean / covariance / Bernoulli r against the JSON
state dumps stored under
``tests/visualization_pipeline/apple_in_the_tray/ekf_state/``.

Both runs use the SAME LLM relation cache so the relation graph (which
gates self-merge and held-set expansion) is identical.

Pass criterion: pose_max_err_world == 0.0 and r_max_err == 0.0.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests._parity_lib import compute_parity_stats, print_parity_report  # noqa: E402


def main() -> None:
    stats = compute_parity_stats("apple_in_the_tray")
    print_parity_report(stats)


if __name__ == "__main__":
    main()
