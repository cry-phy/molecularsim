import numpy as np
import pytest


from molecsim.system.data import SystemData, SystemProperties
from molecsim.system.molecule import Molecule
from molecsim.system.system import SimSystem
from molecsim.utils.geometry import BOUND_EPS
from molecsim.utils.place import randomized_system
from molecsim.utils.rng import RNG
from tests.fixtures import props, second_proto, test_molecule, test_rng


def pairwise_dists_mic(
    positions: np.ndarray, box: np.ndarray | None, periodic: bool
) -> np.ndarray:
    """Pairwise distances with minimum-image convention if periodic, else Euclidean."""
    diff = positions[:, None, :] - positions[None, :, :]
    if periodic:
        # shift by nearest periodic image
        box = np.asarray(box, dtype=float)
        diff -= np.round(diff / box) * box
    return np.linalg.norm(diff, axis=-1)


def _proto() -> Molecule:
    """Return a small asymmetric molecule so some atoms can cross box boundaries."""
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.2, 0.8, 0.0],
        ],
        dtype=float,
    )
    atoms = ["A", "B", "C"]
    charges = np.zeros(3)
    data = SystemData.as_owner(atoms, positions, charges)
    return Molecule(data)


def _flex_dummy_grid(n, box, *args, **kwargs):
    return np.zeros((n, 3), dtype=float)


def pairwise_dists(positions: np.ndarray):
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def test_randomized_system_counts_and_views(props, test_rng):
    """Test that the randomized system has the correct counts and memory views."""
    sys = randomized_system(props, test_rng, grid_function=_flex_dummy_grid)
    assert isinstance(sys, SimSystem)
    total_mols = sum(props.molecules.values())
    total_atoms = sum(m.size() * n for m, n in props.molecules.items())
    assert len(sys.molecules) == total_mols
    assert len(sys.data) == total_atoms
    assert np.shares_memory(sys.molecules[0].data.positions, sys.data.positions)


def test_nonperiodic_is_rigidly_clamped_inside_box():
    """Test that non-periodic systems are rigidly clamped inside the box."""
    mol = _proto()
    box = np.array([2.0, 1.5, 1.0], dtype=float)
    props = SystemProperties(
        molecules={mol: 1}, box=box, temperature=300.0, periodic=False
    )
    rng = RNG(123)
    sys = randomized_system(props, rng)
    pos = sys.data.positions
    assert np.all(pos >= -BOUND_EPS)
    assert np.all(pos <= box + BOUND_EPS)
    from numpy.linalg import norm

    def dmat(positions):
        return norm(positions[:, None, :] - positions[None, :, :], axis=-1)

    np.testing.assert_allclose(
        dmat(pos), dmat(mol.data.positions), rtol=1e-12, atol=1e-12
    )


def test_periodic_wraps_instead_of_error_or_clamp():
    """Test that periodic systems wrap molecules instead of erroring or clamping."""
    mol = _proto()
    box = np.array([2.0, 1.5, 1.0], dtype=float)
    props = SystemProperties(
        molecules={mol: 1}, box=box, temperature=300.0, periodic=True
    )
    rng = RNG(123)
    sys = randomized_system(props, rng)
    pos = sys.data.positions
    assert np.all(pos >= -1e-9)
    assert np.all(pos <= box + 1e-9)


def test_nonperiodic_raises_if_molecule_larger_than_box():
    """Test that non-periodic systems raise an error if a molecule is larger than the box."""
    # Make the prototype wider than the box along x
    mol = _proto()
    too_small_box = np.array([0.5, 1.5, 1.0], dtype=float)
    props = SystemProperties(
        molecules={mol: 1}, box=too_small_box, temperature=300.0, periodic=False
    )
    rng = RNG(123)
    with pytest.raises(ValueError):
        randomized_system(props, rng)


def test_randomized_system_rigid_body_invariance(props, test_rng):
    """Test that the randomized system preserves rigid body invariance."""
    sys = randomized_system(props, test_rng, grid_function=_flex_dummy_grid)

    periodic = getattr(props, "periodic", True)
    box = props.box if periodic else None

    ref_dists = [
        pairwise_dists_mic(m.data.positions, box, periodic)
        for m in props.molecules.keys()
    ]

    for mol in sys.molecules:
        mol_d = pairwise_dists_mic(mol.data.positions, box, periodic)
        assert any(
            np.allclose(mol_d, rd, rtol=1e-10, atol=1e-12) for rd in ref_dists
        ), "Placed molecule does not match any prototype's internal distances under MIC"


def test_randomized_system_custom_grid_is_used(props, test_rng):
    """Test that a custom grid function is used by the randomized system."""
    calls = {}

    def recorder(n, box, *args, **kwargs):
        calls["n"] = n
        calls["box"] = np.array(box, copy=True)
        xs = np.arange(n, dtype=float)
        return np.stack([xs, np.zeros(n), np.zeros(n)], axis=1)

    sys = randomized_system(props, test_rng, grid_function=recorder)

    # grid function received expected inputs
    assert calls.get("n") == sum(props.molecules.values())
    np.testing.assert_allclose(calls.get("box"), props.box)

    # Compare molecule centers on x-axis: should be non-decreasing by construction
    centers_x = [float(np.mean(mol.data.positions[:, 0])) for mol in sys.molecules]
    assert all(centers_x[i] <= centers_x[i + 1] for i in range(len(centers_x) - 1))
