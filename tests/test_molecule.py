import pytest
import numpy as np
from molecsim.system import Molecule
from molecsim.system.data import SystemData  # or your class


@pytest.fixture
def test_molecule():
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
    SystemData.as_owner(atoms, positions, charges)
    mol = Molecule()
    return mol
