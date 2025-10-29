# conftest.py (or your tests module)
import pytest
import numpy as np

from molecsim.data.molecule import Molecule
from molecsim.data.properties import SystemProperties
from molecsim.data.rng import RNG
from molecsim.data.data import FLOAT_DTYPE, SystemData
from molecsim.data.system import SimSystem


class _MockRegistry:
    """Tiny registry that supports the lookups your code may perform."""

    def __init__(self, types=("H", "O", "A", "B", "C", "D")) -> None:
        self._types = tuple(types)
        self._index = {t: i for i, t in enumerate(self._types)}

    @property
    def types(self):
        return self._types

    def index_of(self, name: str) -> int:
        return self._index[name]

    # Optional nicety if someone does registry[idx]
    def __getitem__(self, idx: int) -> str:
        return self._types[idx]

    def names(self, ids):
        return np.array([self._types[i] for i in ids])

    def type_index(self, name: str) -> int:
        """Type name -> registry index."""
        return self._index[name]


class _MockQTable:
    """Charges table stub; tests don't recompute, so return 0.0 for all."""

    def charge_of_type(self, name: str) -> float:
        return 0.0


class MockModel:
    """Minimal stand-in for ModelConstants used by potentials and props."""

    def __init__(self) -> None:
        self._registry = _MockRegistry()
        self._qtable = _MockQTable()
        self._k_b = 0.0019872041  # kcal/mol/K (not used here)
        self._coulomb = 1.0  # unitless in your tests

    @property
    def registry(self):
        return self._registry

    @property
    def qtable(self):
        return self._qtable

    @property
    def k_b(self) -> float:
        return self._k_b

    @property
    def coulomb_coeff(self) -> float:
        return self._coulomb

    def lj_for(self, ti: int, tj: int) -> tuple[float, float]:
        # A_ij, B_ij
        return 1.0, 1.0

    def type_index(self, name: str) -> int:
        return self._registry.type_index(name)

    def charge_of(self, name: str) -> float:
        return self._qtable.charge_of_type(name)

    def names(self, ids):
        return self._registry.names(ids)


@pytest.fixture
def mock_model() -> MockModel:
    """Provide a unit-coefficients model for tests."""
    return MockModel()


@pytest.fixture
def test_molecule(mock_model) -> Molecule:
    """Return a small Molecule instance for testing."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=FLOAT_DTYPE,
    )
    atoms = ["H", "O", "H"]
    charges = np.array([1.0, 2.0, 1.0], dtype=FLOAT_DTYPE)
    data = SystemData.as_owner(
        np.array([mock_model.registry.index_of(a) for a in atoms]), positions, charges
    )
    return Molecule(data)


@pytest.fixture
def test_rng() -> RNG:
    """Generate test rng instance with fixed seed."""
    return RNG(0)


@pytest.fixture
def second_proto(test_molecule: Molecule) -> Molecule:
    """Independent copy shifted in x; ensures two distinct molecule types."""
    data2 = SystemData.as_owner(
        test_molecule.data.atoms,
        test_molecule.data.positions + np.array([1.0, 0.0, 0.0]),
        test_molecule.data._charges,
    )
    return Molecule(data2)


@pytest.fixture
def props(test_molecule: Molecule, second_proto: Molecule) -> SystemProperties:
    """Default SystemProperties fixture periodic=True."""
    # NOTE: SystemProperties is now state-only (box, T, periodic, constants)
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
        periodic=True,
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
def simsys(test_molecule: Molecule, second_proto: Molecule, mock_model: MockModel):
    """Simple system for interaction."""
    atoms = np.array(["A", "B"])
    pos = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    charges = np.zeros(2)
    data = SystemData.as_owner(
        np.array([mock_model.registry.index_of(a) for a in atoms]), pos, charges
    )

    # New SystemProperties signature (no molecules dict here)
    props = SystemProperties(
        molecules={test_molecule: 2, second_proto: 1},
        box=np.array([5.0, 5.0, 5.0], dtype=float),
        temperature=300.0,
        periodic=True,
    )

    # Your SimSystem constructor may accept model now; if so, pass it:
    # return SimSystem(contigous_data=data, molecules=(), properties=props, model=mock_model)
    return SimSystem(
        contigous_data=data, molecules=(), properties=props, model=mock_model
    )


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
