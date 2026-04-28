"""Standalone check for ``TwoTierOrchestrator._should_recompute_relations``.

The full orchestrator import drags in ``gtsam``; this test extracts the
method and calls it bound to a stub object, so it runs in any env with
just Python + the dataclass module.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Stub out gtsam so ``pose_update.orchestrator`` can be imported.
sys.modules.setdefault("gtsam", types.ModuleType("gtsam"))
sys.modules["gtsam"].noiseModel = types.SimpleNamespace(
    Base=type("Base", (), {}),
    Gaussian=types.SimpleNamespace(Covariance=lambda *a, **k: None),
    Diagonal=types.SimpleNamespace(Sigmas=lambda *a, **k: None),
)
sys.modules["gtsam"].Pose3 = lambda *a, **k: None
sys.modules["gtsam"].NonlinearFactorGraph = lambda *a, **k: None
sys.modules["gtsam"].Values = lambda *a, **k: None
sys.modules["gtsam"].LevenbergMarquardtParams = lambda *a, **k: None
sys.modules["gtsam"].LevenbergMarquardtOptimizer = lambda *a, **k: None
sys.modules["gtsam"].Marginals = lambda *a, **k: None
sys.modules["gtsam"].BetweenFactorPose3 = lambda *a, **k: None

from pose_update.orchestrator import (  # noqa: E402
    TwoTierOrchestrator, BernoulliConfig,
)


class Stub:
    """Minimal shape expected by ``_should_recompute_relations``."""
    def __init__(self, **kw):
        self.bernoulli: BernoulliConfig = kw["bernoulli"]
        self.relation_client = kw.get("client", object())   # not None = wired
        self._last_relation_frame = kw.get("last_rel", 0)
        self.last_state = {"phase": kw.get("last_phase", "idle")}
        self.existence = kw.get("existence", {})
        self._known_oids_before_step = kw.get("known", set())
        self.frame_count = kw.get("frame", 0)
        self.state = types.SimpleNamespace(collapsed_objects=lambda: {})


should = TwoTierOrchestrator._should_recompute_relations

CFG = BernoulliConfig(relation_backend="llm", relation_every_n_frames=90)


def check(label, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return bool(cond)


def main() -> int:
    ok = True

    # 1. No client wired → always True (legacy path)
    s = Stub(bernoulli=CFG, client=None)
    ok &= check("legacy (no client) → always fires",
                should(s, {"phase": "idle"}))

    # 2. First call (sentinel _last_relation_frame < 0) → fire
    s = Stub(bernoulli=CFG, last_rel=-10**9, frame=0)
    ok &= check("first call fires",
                should(s, {"phase": "idle"}))

    # 3. Periodic tick
    s = Stub(bernoulli=CFG, last_rel=0, frame=89)
    ok &= check("89 < 90 frames → skip",
                not should(s, {"phase": "idle"}))
    s = Stub(bernoulli=CFG, last_rel=0, frame=90)
    ok &= check("90 frames since last fire → fires",
                should(s, {"phase": "idle"}))

    # 4. periodic disabled
    cfg_no_tick = BernoulliConfig(relation_backend="llm",
                                   relation_every_n_frames=0)
    s = Stub(bernoulli=cfg_no_tick, last_rel=0, frame=10_000)
    ok &= check("periodic disabled + no event → skip",
                not should(s, {"phase": "idle"}))

    # 5. Grasp transition
    s = Stub(bernoulli=CFG, last_rel=0, frame=5, last_phase="idle")
    ok &= check("idle→grasping fires",
                should(s, {"phase": "grasping"}))
    s = Stub(bernoulli=CFG, last_rel=0, frame=5, last_phase="grasping")
    ok &= check("grasping→grasping (no edge) → skip",
                not should(s, {"phase": "grasping"}))

    # 6. Release transition (releasing → idle)
    s = Stub(bernoulli=CFG, last_rel=0, frame=5, last_phase="releasing")
    ok &= check("releasing→idle fires",
                should(s, {"phase": "idle"}))

    # 7. New track born
    s = Stub(bernoulli=CFG, last_rel=0, frame=5,
             existence={1: 0.9, 2: 0.9, 3: 0.9},       # 3 newly confirmed
             known={1: 0.9, 2: 0.9}.keys())
    ok &= check("new confirmed oid fires",
                should(s, {"phase": "idle"}))

    # 8. Confirmed set unchanged
    s = Stub(bernoulli=CFG, last_rel=0, frame=5,
             existence={1: 0.9, 2: 0.9},
             known={1, 2})
    ok &= check("same confirmed set + no event + not at periodic → skip",
                not should(s, {"phase": "idle"}))

    # 9. Low-existence track doesn't count as "new"
    s = Stub(bernoulli=CFG, last_rel=0, frame=5,
             existence={1: 0.9, 2: 0.9, 3: 0.2},  # 3 is tentative
             known={1, 2})
    ok &= check("tentative track (r<r_conf) is not a 'new object' event",
                not should(s, {"phase": "idle"}))

    print("\nALL PASS" if ok else "\nSOMETHING BROKE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
