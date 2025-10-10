import pytest
import numpy as np
from molecsim.system.molecule import Molecule
from molecsim.system.data import SystemData
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
