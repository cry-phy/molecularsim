import pytest

from molecsim.system.data import (
    SystemData,
    make_potcall,
    validate_contributor,
    SystemForces,
)
from molecsim.system.calculations import full_potential
from tests.fixtures import simsys, lin_sum_x


def test_validate_contributor_requires_defaults_or_runtime_guard(simsys):
    # 'bad' requires an extra arg 'y' without a default
    def bad(d, x, y):
        return 0.0

    # Prefer compile-time validation if available:
    try:
        validate_contributor(bad, args=(1,), kwargs={})
        raised = False
    except TypeError:
        raised = True

    if not raised:
        # If validator is permissive, ensure evaluation fails when args are missing.
        pc = make_potcall(bad, 1, name="bad")  # provide x only; y missing
        forces = SystemForces([pc])
        with pytest.raises(TypeError):
            full_potential(simsys, forces)


def test_make_potcall_freezes_kwargs_and_label():
    def lin(d: SystemData, bias: float = 0.0) -> float:
        return bias

    pc = make_potcall(lin, bias=2.5, name="myterm")
    # kwargs mapping proxy is immutable
    with pytest.raises(TypeError):
        pc.kwargs["bias"] = 0.0  # type: ignore[index]
    # label helper (falls back to function name if name None)
    assert pc.label() == "myterm"
    pc2 = make_potcall(lin, 1.0)  # no name
    assert pc2.label() in {"lin", "<unnamed>"}
