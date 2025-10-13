import numpy as np
import pytest

from molecsim.system.data import (
    SystemForces,
    make_potcall,
)
from molecsim.system.calculations import full_potential
from tests.fixtures import lin_sum_x, simsys


def test_full_potential_skips_disabled_by_default(simsys, lin_sum_x):
    pc_on = make_potcall(lin_sum_x, 0.0, name="on", enabled=True)
    pc_off = make_potcall(lin_sum_x, 10.0, name="off", enabled=False)
    forces = SystemForces([pc_on, pc_off])
    assert np.isclose(full_potential(simsys, forces), 1.0)


def test_full_potential_can_include_disabled(simsys, lin_sum_x):
    pc_on = make_potcall(lin_sum_x, 0.0, name="on", enabled=True)
    pc_off = make_potcall(lin_sum_x, 3.0, name="off", enabled=False)
    forces = SystemForces([pc_on, pc_off])
    val = full_potential(simsys, forces, include_disabled=True)
    assert np.isclose(val, 5.0)
