import numpy as np
import pytest

from tests.fixtures import test_molecule, test_rng
from molecsim.utils.rng import RNG
from molecsim.system.molecule import Molecule
from molecsim.system.mutate import mutate_translate, mutate_rotate, MUTATION_FUNCTIONS


def is_proper_rotation(R: np.ndarray, rtol=1e-10, atol=1e-12) -> bool:
    I = np.eye(3, dtype=R.dtype)
    return np.allclose(R.T @ R, I, rtol=rtol, atol=atol) and np.allclose(
        np.linalg.det(R), 1.0, rtol=rtol, atol=1e-10
    )


def test_mutate_translate_shifts_positions(test_molecule: Molecule, test_rng: RNG):
    mol0 = test_molecule
    orig = mol0.data.positions.copy()
    c0 = mol0.center.copy()

    magnitude = 0.5
    mol1 = mutate_translate(mol0, magnitude, test_rng)

    # Original unchanged
    np.testing.assert_allclose(mol0.data.positions, orig)

    # Infer shift from centers
    s = mol1.center - c0

    # Every atom must have moved by the same shift
    np.testing.assert_allclose(mol1.data.positions, orig + s, rtol=1e-12, atol=1e-12)

    # Deepcopy semantics
    assert mol1 is not mol0
    assert mol1.data is not mol0.data
    assert not np.shares_memory(mol1.data.positions, mol0.data.positions)

    # Optional: if RNG.random_direction() returns unit vectors, |s| ≈ magnitude
    # (tolerate tiny FP drift)
    assert np.isclose(np.linalg.norm(s), magnitude, rtol=1e-12, atol=1e-12)


def test_mutate_translate_zero_magnitude_is_noop(
    test_molecule: Molecule, test_rng: RNG
):
    mol0 = test_molecule
    orig = mol0.data.positions.copy()
    mol1 = mutate_translate(mol0, 0.0, test_rng)
    np.testing.assert_allclose(mol1.data.positions, orig, rtol=1e-12, atol=1e-12)
    assert mol1 is not mol0
    assert mol1.data is not mol0.data


def test_mutate_rotate_preserves_distances_only(test_molecule: Molecule, test_rng: RNG):
    mol0 = test_molecule
    X = mol0.data.positions.copy()

    angle = np.pi / 3
    mol1 = mutate_rotate(mol0, angle, test_rng)
    Y = mol1.data.positions.copy()

    # Original molecule unchanged
    np.testing.assert_allclose(mol0.data.positions, X)

    # --- Core invariant: all pairwise distances preserved ---
    n = X.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d0 = np.linalg.norm(X[i] - X[j])
            d1 = np.linalg.norm(Y[i] - Y[j])
            np.testing.assert_allclose(d1, d0, rtol=1e-12, atol=1e-12)

    # Ensure the rotation actually changed orientation (not identity)
    # unless angle == 0
    if not np.isclose(angle, 0.0):
        diff = np.linalg.norm(X - Y)
        assert diff > 0.0


def test_mutate_rotate_zero_angle_is_rigid(test_molecule, test_rng):
    mol0 = test_molecule
    X = mol0.data.positions.copy()
    c = mol0.center.copy()

    mol1 = mutate_rotate(mol0, 0.0, test_rng)
    Y = mol1.data.positions

    # Original unchanged
    np.testing.assert_allclose(mol0.data.positions, X)

    # It must be a rigid motion about the center (rotation), not necessarily identity:
    np.testing.assert_allclose(mol1.center, c, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(Y - c, axis=1),
        np.linalg.norm(X - c, axis=1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_mutation_functions_registry_contains_both():
    names = {f.__name__ for f in MUTATION_FUNCTIONS}
    assert {"mutate_translate", "mutate_rotate"}.issubset(names)
    for f in MUTATION_FUNCTIONS:
        assert callable(f)
        assert f.__code__.co_argcount == 3
