import numpy as np
import pytest

from molecsim.system.data import (
    SystemForces,
    make_potcall,
)
from molecsim.system.calculations import full_potential


def test_full_potential_skips_disabled_by_default(simsys, lin_sum_x):
    """Test that full_potential skips disabled potential calls by default."""
    pot_call_on = make_potcall(lin_sum_x, 0.0, name="on", enabled=True)
    pot_call_off = make_potcall(lin_sum_x, 10.0, name="off", enabled=False)
    forces = SystemForces([pot_call_on, pot_call_off])
    assert np.isclose(full_potential(simsys, forces), 1.0)


def test_full_potential_can_include_disabled(simsys, lin_sum_x):
    """Test that full_potential can include disabled potential calls."""
    pot_call_on = make_potcall(lin_sum_x, 0.0, name="on", enabled=True)
    pot_call_off = make_potcall(lin_sum_x, 3.0, name="off", enabled=False)
    forces = SystemForces([pot_call_on, pot_call_off])
    val = full_potential(simsys, forces, include_disabled=True)
    assert np.isclose(val, 5.0)
