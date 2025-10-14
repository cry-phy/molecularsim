import numpy as np
import pytest

from tests.fixtures import test_molecule, test_rng
from molecsim.utils.rng import RNG
from molecsim.system.molecule import Molecule
from molecsim.system.mutate import mutate_translate, mutate_rotate, MUTATION_FUNCTIONS


def is_proper_rotation(rotation_matrix: np.ndarray, rtol=1e-10, atol=1e-12) -> bool:
    """Check if a matrix is a proper rotation matrix."""
    identity_matrix = np.eye(3, dtype=rotation_matrix.dtype)
    return np.allclose(rotation_matrix.T @ rotation_matrix, identity_matrix, rtol=rtol, atol=atol) and np.allclose(
        np.linalg.det(rotation_matrix), 1.0, rtol=rtol, atol=1e-10
    )


def test_mutate_translate_shifts_positions(test_molecule: Molecule, test_rng: RNG):
    """Test that mutate_translate shifts positions correctly."""
    mol0 = test_molecule
    orig = mol0.data.positions.copy()
    c0 = mol0.center.copy()

    magnitude = 0.5
    mol1 = mutate_translate(mol0, magnitude, test_rng)

    # Original unchanged
    np.testing.assert_allclose(mol0.data.positions, orig)

    # Infer shift from centers
    shift = mol1.center - c0

    # Every atom must have moved by the same shift
    np.testing.assert_allclose(mol1.data.positions, orig + shift, rtol=1e-12, atol=1e-12)

    # Deepcopy semantics
    assert mol1 is not mol0
    assert mol1.data is not mol0.data
    assert not np.shares_memory(mol1.data.positions, mol0.data.positions)

    # Optional: if RNG.random_direction() returns unit vectors, |s| ≈ magnitude
    # (tolerate tiny FP drift)
    assert np.isclose(np.linalg.norm(shift), magnitude, rtol=1e-12, atol=1e-12)


def test_mutate_translate_zero_magnitude_is_noop(
    test_molecule: Molecule, test_rng: RNG
):
    """Test that mutate_translate with zero magnitude is a no-op."""
    mol0 = test_molecule
    orig = mol0.data.positions.copy()
    mol1 = mutate_translate(mol0, 0.0, test_rng)
    np.testing.assert_allclose(mol1.data.positions, orig, rtol=1e-12, atol=1e-12)
    assert mol1 is not mol0
    assert mol1.data is not mol0.data


def test_mutate_rotate_preserves_distances_only(test_molecule: Molecule, test_rng: RNG):
    """Test that mutate_rotate preserves distances only."""
    mol0 = test_molecule
    positions_0 = mol0.data.positions.copy()

    angle = np.pi / 3
    mol1 = mutate_rotate(mol0, angle, test_rng)
    positions_1 = mol1.data.positions.copy()

    # Original molecule unchanged
    np.testing.assert_allclose(mol0.data.positions, positions_0)

    # --- Core invariant: all pairwise distances preserved ---
    n = positions_0.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d0 = np.linalg.norm(positions_0[i] - positions_0[j])
            d1 = np.linalg.norm(positions_1[i] - positions_1[j])
            np.testing.assert_allclose(d1, d0, rtol=1e-12, atol=1e-12)

    # Ensure the rotation actually changed orientation (not identity)
    # unless angle == 0
    if not np.isclose(angle, 0.0):
        diff = np.linalg.norm(positions_0 - positions_1)
        assert diff > 0.0


def test_mutate_rotate_zero_angle_is_rigid(test_molecule, test_rng):
    """Test that mutate_rotate with zero angle is rigid."""
    mol0 = test_molecule
    positions_0 = mol0.data.positions.copy()
    center = mol0.center.copy()

    mol1 = mutate_rotate(mol0, 0.0, test_rng)
    positions_1 = mol1.data.positions

    # Original unchanged
    np.testing.assert_allclose(mol0.data.positions, positions_0)

    # It must be a rigid motion about the center (rotation), not necessarily identity:
    np.testing.assert_allclose(mol1.center, center, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(positions_1 - center, axis=1),
        np.linalg.norm(positions_0 - center, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_mutation_functions_registry_contains_both():
    """Test that the mutation functions registry contains both translate and rotate."""
    names = {f.__name__ for f in MUTATION_FUNCTIONS}
    assert {"mutate_translate", "mutate_rotate"}.issubset(names)
    for f in MUTATION_FUNCTIONS:
        assert callable(f)
        assert f.__code__.co_argcount == 3
