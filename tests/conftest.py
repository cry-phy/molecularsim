import pytest
import numpy as np
from molecsim.data.molecule import Molecule
from molecsim.data.properties import SystemProperties
from molecsim.utils.rng import RNG
from molecsim.data.data import FLOAT_DTYPE, SystemData
from molecsim.data.system import SimSystem


@pytest.fixture
def test_molecule() -> Molecule:
    """Return a small Molecule instance for testing."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=FLOAT_DTYPE,
    )
    atoms = ["H", "O", "H"]
    charges = np.array([1.0, 2.0, 1.0], dtype=FLOAT_DTYPE)
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
        test_molecule.data.atoms_as_str(),
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


@pytest.fixture
def lin_sum_x():
    def energy(data1: SystemData, data2: SystemData, bias: float = 0.0) -> float:
        return float(np.sum(data1.positions[:, 0]) + bias)

    return energy


@pytest.fixture
def simsys():
    """Simples system for interaction."""
    atoms = np.array(["A", "B"])
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    charges = np.zeros(2)
    data = SystemData.as_owner(atoms, pos, charges)
    props = SystemProperties(
        molecules={}, box=np.array([5.0, 5.0, 5.0]), temperature=300.0
    )
    return SimSystem(contigous_data=data, molecules=(), properties=props)


@pytest.fixture
def mock_props(test_molecule: Molecule, second_proto: Molecule) -> SystemProperties:
    """Mock SystemProperties with LJ_A/B and coulomb_coeff set to 1."""

    class MockSystemProperties(SystemProperties):
        @property
        def lj_params(self):
            return 1.0, 1.0

        @property
        def coulomb_coeff(self):
            return 1.0

    box = np.array([10.0, 8.0, 6.0], dtype=float)
    return MockSystemProperties(
        molecules={test_molecule: 2, second_proto: 1},
        box=box,
        temperature=300.0,
        periodic=False,
    )
