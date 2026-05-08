"""pytest conftest: make the repo root importable so tests can do
``from utils import …`` regardless of where they live.

Also pre-stubs the optional slow-tier dependency ``gtsam`` so tests
that touch ``ekf_tracker.configs`` (which goes through the package
``__init__.py`` and transitively pulls ``factor_graph`` → ``gtsam``)
collect cleanly in environments without gtsam installed. The stub is
inert at runtime --- tests that actually invoke the slow tier still
need a real gtsam install.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _GtsamStub:
    def __getattr__(self, _):
        return _GtsamStub()
    def __call__(self, *a, **kw):
        return _GtsamStub()


sys.modules.setdefault("gtsam", _GtsamStub())
