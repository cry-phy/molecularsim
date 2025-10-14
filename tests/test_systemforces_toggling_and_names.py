import numpy as np
import pytest

from molecsim.system.data import (
    SystemForces,
    make_potcall,
)
from molecsim.system.calculations import full_potential
from tests.fixtures import lin_sum_x, simsys


def test_toggling_and_names_helpers_exist(simsys, lin_sum_x):
    """Test that toggling and naming helpers exist and function correctly."""
    pot_call_1 = make_potcall(lin_sum_x, 0.0, name="E1")
    pot_call_2 = make_potcall(lin_sum_x, 1.0, name="E2")
    pot_call_3 = make_potcall(lin_sum_x, 2.0, name="E3")
    forces = SystemForces([pot_call_1, pot_call_2, pot_call_3])

    # .names() helper for simple introspection
    assert forces.names() == ("E1", "E2", "E3")

    # Disable E2 and E3; only E1 contributes
    toggled = forces.with_disabled("E2", "E3")
    assert toggled.names() == ("E1", "E2", "E3")  # names unchanged
    v1 = full_potential(simsys, toggled)
    assert np.isclose(v1, 1.0)  # sum_x=1.0 + bias 0.0

    # Re-enable E3 only
    re_enabled = toggled.with_enabled("E3")
    v2 = full_potential(simsys, re_enabled)
    # contributions: E1 -> 1.0, E3 -> 1.0+2.0 = 3.0; total 4.0
    assert np.isclose(v2, 4.0)
