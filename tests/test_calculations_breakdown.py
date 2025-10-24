import numpy as np
import pytest

from molecsim.system.data import (
    SystemForces,
    make_potcall,
)
from molecsim.system.calculations import potential_breakdown, full_potential


def test_breakdown_matches_sum_and_preserves_order(simsys, lin_sum_x):
    """Test that potential_breakdown matches the sum of full_potential and preserves order."""
    pot_call_1 = make_potcall(lin_sum_x, 0.0, name="E1")
    pot_call_2 = make_potcall(lin_sum_x, 2.0, name="E2")
    forces = SystemForces([pot_call_1, pot_call_2])

    parts = potential_breakdown(simsys, forces)
    names, vals = zip(*parts)
    assert names == ("E1", "E2")
    assert np.isclose(sum(vals), full_potential(simsys, forces))


def test_breakdown_include_disabled_flag(simsys, lin_sum_x):
    """Test the include_disabled flag in potential_breakdown."""
    pot_call_1 = make_potcall(lin_sum_x, 0.0, name="on", enabled=True)
    pot_call_2 = make_potcall(lin_sum_x, 5.0, name="off", enabled=False)
    forces = SystemForces([pot_call_1, pot_call_2])

    # Default: disabled skipped
    parts_def = potential_breakdown(simsys, forces)
    assert tuple(n for n, _ in parts_def) == ("on",)

    # Explicitly include disabled
    parts_all = potential_breakdown(simsys, forces, include_disabled=True)
    assert set(n for n, _ in parts_all) == {"on", "off"}
