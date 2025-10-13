import pytest
import numpy as np
from molecsim.system.molecule import Molecule
from molecsim.utils.rng import RNG
from molecsim.system.data import SystemData, SystemProperties


@pytest.fixture
def test_molecule() -> Molecule:
    """Return a small Molecule instance for testing."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    atoms = ["H", "O", "H"]
    charges = np.array([1, 2, 1])
    data = SystemData.as_owner(atoms, positions, charges)
    mol = Molecule(data)
    return mol


@pytest.fixture
def test_rng() -> RNG:
    """Generate test rng instance with fixed seed."""
    return RNG(0)


@pytest.fixture
def second_proto(test_molecule: Molecule) -> Molecule:
    """Independent copy shifted in x; ensures two distinct molecule types."""
    data2 = SystemData.as_owner(
        test_molecule.data._atoms,
        test_molecule.data.positions + np.array([1.0, 0.0, 0.0]),
        test_molecule.data._charges,
    )
    return Molecule(data2)


@pytest.fixture
def props(test_molecule: Molecule, second_proto: Molecule) -> SystemProperties:
    """Default SystemProperties fixture — periodic=True."""
    box = np.array([10.0, 8.0, 6.0], dtype=float)
    return SystemProperties(
        molecules={test_molecule: 2, second_proto: 1},
        box=box,
        temperature=300.0,
        periodic=True,
    )


@pytest.fixture
def props_nonperiodic(
    test_molecule: Molecule, second_proto: Molecule
) -> SystemProperties:
    """Non-periodic version for clamping tests."""
    box = np.array([10.0, 8.0, 6.0], dtype=float)
    return SystemProperties(
        molecules={test_molecule: 2, second_proto: 1},
        box=box,
        temperature=300.0,
        periodic=False,
    )


@pytest.fixture
def flex_dummy_grid():
    """Grid function with flexible signature returning zero-origin nodes."""

    def _grid(n, box, *args, **kwargs):
        return np.zeros((n, 3), dtype=float)

    return _grid
