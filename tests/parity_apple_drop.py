"""Parity test: EkfTracker public API vs visualize_ekf_tracking.main().

For each frame of `apple_drop`, drives the new EkfTracker.step() and
compares its per-track world-frame mean / covariance / Bernoulli r
against the JSON state dumps already produced by main()
(tests/visualization_pipeline/apple_drop/ekf_state/).

Both runs use the SAME LLM relation cache so the relation graph
(which gates self-merge and held-set expansion) is identical.

Pass criterion: pose_max_err_world == 0.0 and r_max_err == 0.0.
The shared loop, parity comparison, and report formatting live in
``tests/_parity_lib.py``; this script is a trajectory-specific
CLI wrapper.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests._parity_lib import compute_parity_stats, print_parity_report  # noqa: E402


def main() -> None:
    stats = compute_parity_stats("apple_drop")
    # apple_drop's release transition is around frame 274; the
    # pre-refactor script printed a per-oid worst-offender breakdown
    # for frames >= 274. Preserve that diagnostic.
    print_parity_report(stats, post_frame_breakdown_at=274)


if __name__ == "__main__":
    main()
