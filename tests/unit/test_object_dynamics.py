"""Unit tests for ``utils/object_dynamics.py``.

Post-Phase-D1 the module-level ``DEFAULT_DYNAMICS``,
``LABEL_DYNAMICS_TABLE`` and ``SHAPE_FOOTPRINT_FACTOR`` constants live
in ``ekf_tracker/configs/default.yaml``. This test loads them via the
canonical builder and threads them into the lookup helpers (which now
take their table / footprint dict as required kwargs).
"""
from __future__ import annotations

import pytest

from utils.object_dynamics import (
    ObjectDynamicsProperty,
    lookup_dynamics,
    shape_footprint_factor,
)
from ekf_tracker.configs import load_config, build_object_dynamics_table

_CFG = load_config()
DEFAULT_DYNAMICS, LABEL_DYNAMICS_TABLE, SHAPE_FOOTPRINT_FACTOR = (
    build_object_dynamics_table(_CFG))


class TestObjectDynamicsProperty:
    def test_construction_valid(self):
        p = ObjectDynamicsProperty(
            label="x", e=0.5, mu=0.5, shape="spherical", radius_m=0.05)
        assert p.label == "x"
        assert p.shape == "spherical"

    def test_invalid_restitution_raises(self):
        with pytest.raises(ValueError):
            ObjectDynamicsProperty("x", e=1.5, mu=0.5, shape="spherical", radius_m=0.05)
        with pytest.raises(ValueError):
            ObjectDynamicsProperty("x", e=-0.1, mu=0.5, shape="spherical", radius_m=0.05)

    def test_invalid_friction_raises(self):
        with pytest.raises(ValueError):
            ObjectDynamicsProperty("x", e=0.5, mu=-0.1, shape="spherical", radius_m=0.05)
        with pytest.raises(ValueError):
            ObjectDynamicsProperty("x", e=0.5, mu=3.0, shape="spherical", radius_m=0.05)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            ObjectDynamicsProperty("x", e=0.5, mu=0.5, shape="banana", radius_m=0.05)

    def test_invalid_radius_raises(self):
        with pytest.raises(ValueError):
            ObjectDynamicsProperty("x", e=0.5, mu=0.5, shape="spherical", radius_m=0.0)
        with pytest.raises(ValueError):
            ObjectDynamicsProperty("x", e=0.5, mu=0.5, shape="spherical", radius_m=-0.1)


class TestLookupDynamics:
    def test_known_label(self):
        p = lookup_dynamics(
            "apple", default=DEFAULT_DYNAMICS, table=LABEL_DYNAMICS_TABLE)
        assert p.label == "apple"
        assert p.shape == "spherical"

    def test_unknown_label_returns_default(self):
        assert lookup_dynamics(
            "zzzz", default=DEFAULT_DYNAMICS, table=LABEL_DYNAMICS_TABLE
        ) is DEFAULT_DYNAMICS

    def test_none_label_returns_default(self):
        assert lookup_dynamics(
            None, default=DEFAULT_DYNAMICS, table=LABEL_DYNAMICS_TABLE
        ) is DEFAULT_DYNAMICS

    def test_override_takes_priority(self):
        custom = ObjectDynamicsProperty(
            label="custom", e=0.9, mu=0.1, shape="box", radius_m=0.1)
        # Override wins even when the label has a table entry.
        assert lookup_dynamics(
            "apple", default=DEFAULT_DYNAMICS, table=LABEL_DYNAMICS_TABLE,
            override=custom) is custom
        assert lookup_dynamics(
            None, default=DEFAULT_DYNAMICS, table=LABEL_DYNAMICS_TABLE,
            override=custom) is custom


class TestTableContents:
    def test_known_labels_are_valid(self):
        # The six labels declared in det_pipeline OBJECTS list must each
        # have a dynamics property.
        for lbl in ("apple", "milkbox", "cola", "cup", "pot", "flowerpot"):
            assert lbl in LABEL_DYNAMICS_TABLE
            p = LABEL_DYNAMICS_TABLE[lbl]
            assert p.label == lbl
            # Sanity-check the values are in the documented ranges.
            assert 0.0 <= p.e <= 1.0
            assert 0.0 <= p.mu <= 2.0
            assert p.radius_m > 0.0

    def test_apple_is_softer_than_cola(self):
        # Documented ordering: ripe fruit < rigid plastic.
        assert LABEL_DYNAMICS_TABLE["apple"].e < LABEL_DYNAMICS_TABLE["cola"].e


class TestShapeFootprintFactor:
    def test_known_shapes(self):
        assert shape_footprint_factor("spherical",   factors=SHAPE_FOOTPRINT_FACTOR) == 0.25
        assert shape_footprint_factor("cylindrical", factors=SHAPE_FOOTPRINT_FACTOR) == 0.50
        assert shape_footprint_factor("box",         factors=SHAPE_FOOTPRINT_FACTOR) == 0.70
        assert shape_footprint_factor("irregular",   factors=SHAPE_FOOTPRINT_FACTOR) == 1.00

    def test_unknown_shape_falls_back_to_irregular(self):
        assert shape_footprint_factor(
            "banana", factors=SHAPE_FOOTPRINT_FACTOR) == 1.00
