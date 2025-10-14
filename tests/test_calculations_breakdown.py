import numpy as np
import pytest

from molecsim.system.data import (
    SystemForces,
    make_potcall,
)
from molecsim.system.calculations import potential_breakdown, full_potential
from tests.fixtures import simsys, lin_sum_x


def test_breakdown_matches_sum_and_preserves_order(simsys, lin_sum_x):
    pc1 = make_potcall(lin_sum_x, 0.0, name="E1")
    pc2 = make_potcall(lin_sum_x, 2.0, name="E2")
    forces = SystemForces([pc1, pc2])

    parts = potential_breakdown(simsys, forces)
    names, vals = zip(*parts)
    assert names == ("E1", "E2")
    assert np.isclose(sum(vals), full_potential(simsys, forces))


def test_breakdown_include_disabled_flag(simsys, lin_sum_x):
    pc1 = make_potcall(lin_sum_x, 0.0, name="on", enabled=True)
    pc2 = make_potcall(lin_sum_x, 5.0, name="off", enabled=False)
    forces = SystemForces([pc1, pc2])

    # Default: disabled skipped
    parts_def = potential_breakdown(simsys, forces)
    assert tuple(n for n, _ in parts_def) == ("on",)

    # Explicitly include disabled
    parts_all = potential_breakdown(simsys, forces, include_disabled=True)
    assert set(n for n, _ in parts_all) == {"on", "off"}
