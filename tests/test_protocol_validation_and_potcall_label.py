import pytest

from molecsim.data.contributors import make_potcall, SystemPotentials, validate_contributor
from molecsim.physics.calculations import full_potential
from molecsim.data.data import SystemData


def test_validate_contributor_requires_defaults_or_runtime_guard(simsys):
    """Test that validate_contributor requires defaults or runtime guard."""

    # 'bad' requires an extra arg 'z' without a default
    def bad(data, x, y, z):
        return 0.0

    # Prefer compile-time validation if available:
    try:
        validate_contributor(bad, args=(1,), kwargs={})
        raised = False
    except TypeError:
        raised = True

    if not raised:
        # If validator is permissive, ensure evaluation fails when args are missing.
        pot_call = make_potcall(bad, 1, name="bad")  # provide x, only; y,z missing
        forces = SystemPotentials([pot_call])
        with pytest.raises(TypeError):
            full_potential(simsys, forces)


def test_make_potcall_freezes_kwargs_and_label():
    """Test that make_potcall freezes kwargs and label."""

    def lin(data: SystemData, bias: float = 0.0) -> float:
        return bias

    pot_call = make_potcall(lin, bias=2.5, name="myterm")
    # kwargs mapping proxy is immutable
    with pytest.raises(TypeError):
        pot_call.kwargs["bias"] = 0.0  # type: ignore[index]
    # label helper (falls back to function name if name None)
    assert pot_call.label() == "myterm"
    pot_call_2 = make_potcall(lin, 1.0)  # no name
    assert pot_call_2.label() in {"lin", "<unnamed>"}
