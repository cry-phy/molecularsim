import numpy as np
import pytest

from molecsim.system.data import FLOAT_DTYPE
from molecsim.utils.geometry import (
    rot_matrix,
    rot_matrix_quaternions,
)


RTOL = 1e-12
ATOL = 1e-12


def is_rotation_matrix(R: np.ndarray, rtol=RTOL, atol=ATOL):
    I = np.eye(3, dtype=R.dtype)
    return np.allclose(R.T @ R, I, rtol=rtol, atol=atol) and np.allclose(
        np.linalg.det(R), 1.0, rtol=rtol, atol=1e-10
    )


# ---------- rot_matrix (bend/dihedral) ----------


def test_rot_matrix_identity():
    R = rot_matrix(0.0, 0.0)
    assert R.shape == (3, 3)
    assert R.dtype == FLOAT_DTYPE
    assert np.allclose(R, np.eye(3, dtype=FLOAT_DTYPE), rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize(
    "bend, dihedral",
    [
        (0.0, 0.0),
        (np.pi / 6, 0.0),
        (0.0, np.pi / 3),
        (np.pi / 4, np.pi / 7),
        (np.pi / 2, np.pi / 2),
        (np.pi, np.pi / 5),
    ],
)
def test_rot_matrix_is_proper_rotation(bend, dihedral):
    R = rot_matrix(bend, dihedral)
    assert is_rotation_matrix(R)


def test_rot_matrix_known_case():
    bend = np.pi / 3
    R = rot_matrix(bend, 0.0)
    Ry = np.array(
        [
            [np.cos(bend), 0.0, np.sin(bend)],
            [0.0, 1.0, 0.0],
            [-np.sin(bend), 0.0, np.cos(bend)],
        ],
        dtype=FLOAT_DTYPE,
    )
    assert np.allclose(R, Ry, rtol=RTOL, atol=ATOL)


# ---------- rot_matriquaternions (u1,u2,u3) ----------


@pytest.mark.parametrize("u1", [0.0, 0.2, 0.5, 0.9])
@pytest.mark.parametrize("u2", [0.0, 0.37])
@pytest.mark.parametrize("u3", [0.0, 0.73])
def test_rot_matrix_quaternions_is_proper_rotation(u1, u2, u3):
    matrix = rot_matrix_quaternions(u1, u2, u3)
    assert matrix.shape == (3, 3)
    assert np.issubdtype(matrix.dtype, np.floating)
    assert is_rotation_matrix(matrix, rtol=1e-10, atol=1e-12)


def test_quaternion_rotation_preserves_norms():
    rng = np.random.default_rng(1234)
    for _ in range(10):
        u1, u2, u3 = rng.random(3)
        matrix = rot_matrix_quaternions(u1, u2, u3)
        v = rng.normal(size=3)
        vmatrix = matrix @ v
        assert np.allclose(
            np.linalg.norm(vmatrix), np.linalg.norm(v), rtol=1e-12, atol=1e-12
        )


def test_quaternion_edge_case_u1_zero_is_pi_rotation():
    u1, u2, u3 = 0.0, 0.25, 0.5
    matrix = rot_matrix_quaternions(u1, u2, u3)
    tr = np.trace(matrix)
    assert np.allclose(tr, -1.0, rtol=1e-12, atol=1e-12)
    assert is_rotation_matrix(matrix)
