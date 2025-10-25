from __future__ import annotations
import typing
import numpy as np
import numpy.typing as npt
from molecsim.system.data import (
    FloatPrecision,
    FLOAT_DTYPE,
    StructureArray,
    SystemData,
    SystemProperties,
)
from molecsim.system.molecule import Molecule

if typing.TYPE_CHECKING:
    from molecsim.utils.rng import RNG

BOUND_EPS = 1e-8


def rot_matrix(bend_angle: float, dihedral_angle: float) -> npt.NDArray[FloatPrecision]:
    """Generate a 3D rotation matrix from bend and dihedral angles (radians).

    Now deprecated since it doesnt appreciate homogenous sampling.
    """
    cos_bend = np.cos(bend_angle)
    sin_bend = np.sin(bend_angle)
    cos_dihedral = np.cos(dihedral_angle)
    sin_dihedral = np.sin(dihedral_angle)

    return np.array(
        [
            [cos_dihedral * cos_bend, -sin_dihedral, cos_dihedral * sin_bend],
            [sin_dihedral * cos_bend, cos_dihedral, sin_dihedral * sin_bend],
            [-sin_bend, 0.0, cos_bend],
        ],
        dtype=FLOAT_DTYPE,
    )


def rot_matrix_uniform(
    u1: float, u2: float, u3: float, theta_max: float
) -> npt.NDArray[FloatPrecision]:
    """Generate rotation matrix uniform up to angle.

    Truly black magic but somehow works!!.
    """
    phi = 2 * np.pi * u1
    z = 2 * u2 - 1
    r = np.sqrt(1 - z**2)
    kx = r * np.cos(phi)
    ky = r * np.sin(phi)
    kz = z

    theta_max = float(np.clip(theta_max, 0.0, np.pi))
    if theta_max == 0.0:
        return np.eye(3, dtype=FLOAT_DTYPE)

    c1 = theta_max - np.sin(theta_max)
    u = np.clip(u3, 0.0, 1.0)

    c = ((6.0 * c1) ** (1.0 / 3.0)) / theta_max - 1.0
    theta = ((6.0 * c1 * u) ** (1.0 / 3.0)) / (1.0 + c * (u ** (2.0 / 3.0)))

    c2 = u * c1
    for _ in range(2):
        denom = 1.0 - np.cos(theta)
        if denom < 1e-12:
            theta += 1e-6
            denom = 1.0 - np.cos(theta)
        theta -= (theta - np.sin(theta) - c2) / denom
    theta = float(np.clip(theta, 0.0, theta_max))

    cth = np.cos(theta)
    sth = np.sin(theta)
    t = 1.0 - cth

    return np.array(
        [
            [cth + kx * kx * t, kx * ky * t - kz * sth, kx * kz * t + ky * sth],
            [ky * kx * t + kz * sth, cth + ky * ky * t, ky * kz * t - kx * sth],
            [kz * kx * t - ky * sth, kz * ky * t + kx * sth, cth + kz * kz * t],
        ],
        dtype=FLOAT_DTYPE,
    )


def rot_matrix_quaternions(
    u1: float, u2: float, u3: float
) -> npt.NDArray[FloatPrecision]:
    """Rotation matrix using quaternions.

    Now deprecated since it doesnt allow for easy limits to rotation.
    """
    # No real clue whats going on here but seems to work (ChatGPT)
    # Using method from "Graphics Gems" (Ken Shoemake)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )
    # Convert quaternion to rotation matrix
    w, x, y, z = q[3], q[0], q[1], q[2]
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return matrix


def contain_to_box(
    data: SystemData,
    props: SystemProperties,
) -> None:
    """Contain molecule into box based on periodicidy."""
    if props.periodic:
        new_positions = wrap_periodic(
            data.positions, np.zeros_like(props.box), props.box
        )
    else:
        new_positions = hold_non_periodic(data.positions, props.box)

    np.copyto(data.positions, new_positions)


def wrap_periodic(x, a, b):
    """Wrap N-D positions x into the periodic interval [a, b)."""
    x = np.asarray(x)
    a = np.asarray(a)
    b = np.asarray(b)
    return a + np.mod(x - a, b - a)


def hold_non_periodic(
    positions: npt.NDArray[np.floating],
    box: npt.NDArray[np.floating],
    *,
    eps: float = 1e-9,
) -> StructureArray:
    """Rigidly translate the molecule so all atoms lie in [0, box] (±eps).

    Preserves pairwise distances.
    """
    positions = np.asarray(positions, dtype=float)
    box = np.asarray(box, dtype=float)

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    span = maxs - mins

    if np.any(span > box + eps):
        raise ValueError(
            "Molecule extent exceeds non-periodic box along at least one axis."
        )
    lo = -mins + eps
    hi = box - maxs - eps
    if np.any(lo > hi + 1e-15):
        raise ValueError("No feasible rigid shift to place molecule inside box.")
    shift = np.clip(0.0, lo, hi)
    new_pos = positions + shift
    return new_pos


def fuzzyfy(
    positions: StructureArray,
    box: StructureArray,
    rng: RNG,
    strength: float = 0.2,
) -> StructureArray:
    """Fuzzy move the positions based on density."""
    position_volume = np.prod(box) / len(positions)
    position_radius = position_volume ** (1 / 3)
    fuzz = rng.random_like(positions, limit=float(strength * position_radius))
    proposal = fuzz + positions
    return wrap_periodic(proposal, 0, box)


def unwrap_to_reference(
    pos: npt.NDArray[np.floating], box: npt.NDArray[np.floating], iref: int = 0
) -> npt.NDArray[np.floating]:
    """Place all atoms in the nearest image relative to atom iref."""
    pos = np.asarray(pos, float)
    box = np.asarray(box, float)
    ref = pos[iref]
    diff = pos - ref  # pairwise to reference
    diff -= box * np.round(diff / box)  # minimum-image to reference
    return ref + diff


def rotate_rigid_pbc(
    pos: npt.NDArray[np.floating],
    box: npt.NDArray[np.floating],
    R: npt.NDArray[np.floating],
    iref: int = 0,
) -> npt.NDArray[np.floating]:
    """Rigidly rotate a molecule across PBC and rewrap with a single lattice shift."""
    u = unwrap_to_reference(pos, box, iref=iref)

    c = u.mean(axis=0)
    ru = (u - c) @ R.T + c

    shift = -box * np.floor(ru[iref] / box)
    rw = ru + shift

    return rw


def dmat_min_image(
    pos: npt.NDArray[np.floating], box: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    diff = pos[:, None, :] - pos[None, :, :]
    diff -= box * np.round(diff / box)
    return np.linalg.norm(diff, axis=-1)
