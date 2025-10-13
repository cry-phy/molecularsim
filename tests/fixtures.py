import pytest
import numpy as np
from molecsim.system.molecule import Molecule
from molecsim.system.data import SystemData, SystemProperties
from molecsim.system.system import SimSystem
from molecsim.utils.rng import RNG


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
def lin_sum_x():
    def energy(data: SystemData, bias: float = 0.0) -> float:
        return float(np.sum(data.positions[:, 0]) + bias)

    return energy
