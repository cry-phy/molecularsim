from __future__ import annotations
import math
from molecsim.data.data import FloatPrecision, SystemData, StructureArray, FLOAT_DTYPE
from molecsim.data.properties import SystemProperties
import numpy.typing as npt
import numpy as np
import typing
from numba import njit, float64

if typing.TYPE_CHECKING:
    from molecsim.data.rng import RNG

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


@njit(float64[:, :](float64, float64, float64, float64), fastmath=True)
def rot_matrix_uniform(
    u1: float,
    u2: float,
    u3: float,
    theta_max: float,
) -> npt.NDArray[FloatPrecision]:
    """Generate rotation matrix uniform up to angle.

    Truly black magic but somehow works!!.
    """
    # scalar-safe clip
    if theta_max < 0.0:
        theta_max = 0.0
    elif theta_max > math.pi:
        theta_max = math.pi

    phi = 2.0 * math.pi * u1
    z = 2.0 * u2 - 1.0
    r = math.sqrt(max(0.0, 1.0 - z * z))
    kx = r * math.cos(phi)
    ky = r * math.sin(phi)
    kz = z

    if theta_max == 0.0:
        matrix = np.zeros((3, 3), dtype=np.float64)
        matrix[0, 0] = 1.0
        matrix[1, 1] = 1.0
        matrix[2, 2] = 1.0
        return matrix

    c1 = theta_max - math.sin(theta_max)
    u = u3
    if u < 0.0:
        u = 0.0
    elif u > 1.0:
        u = 1.0

    c = ((6.0 * c1) ** (1.0 / 3.0)) / theta_max - 1.0
    theta = ((6.0 * c1 * u) ** (1.0 / 3.0)) / (1.0 + c * (u ** (2.0 / 3.0)))

    c2 = u * c1
    for _ in range(2):
        denom = 1.0 - math.cos(theta)
        if denom < 1e-12:
            theta += 1e-6
            denom = 1.0 - math.cos(theta)
        theta -= (theta - math.sin(theta) - c2) / denom

    if theta < 0.0:
        theta = 0.0
    elif theta > theta_max:
        theta = theta_max

    cth = math.cos(theta)
    sth = math.sin(theta)
    t = 1.0 - cth

    matrix = np.empty((3, 3), dtype=np.float64)
    matrix[0, 0] = cth + kx * kx * t
    matrix[0, 1] = kx * ky * t - kz * sth
    matrix[0, 2] = kx * kz * t + ky * sth
    matrix[1, 0] = ky * kx * t + kz * sth
    matrix[1, 1] = cth + ky * ky * t
    matrix[1, 2] = ky * kz * t - kx * sth
    matrix[2, 0] = kz * kx * t - ky * sth
    matrix[2, 1] = kz * ky * t + kx * sth
    matrix[2, 2] = cth + kz * kz * t
    return matrix


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


@njit
def unwrap_to_reference(
    pos: npt.NDArray[np.floating], box: npt.NDArray[np.floating], iref: int = 0
) -> npt.NDArray[np.floating]:
    """Place all atoms in the nearest image relative to atom iref."""
    ref = pos[iref]
    diff = pos - ref  # pairwise to reference
    diff -= box * np.round(diff / box)  # minimum-image to reference
    return ref + diff


@njit
def rotate_rigid_pbc(
    pos: npt.NDArray[np.floating],
    box: npt.NDArray[np.floating],
    r_matrix: npt.NDArray[np.floating],
    iref: int = 0,
) -> npt.NDArray[np.floating]:
    """Rigidly rotate a molecule across PBC and rewrap with a single lattice shift."""
    unwrapped = unwrap_to_reference(pos, box, iref=iref)

    center = np.sum(unwrapped, axis=0) / unwrapped.shape[0]
    ru = (unwrapped - center) @ r_matrix.T + center

    shift = -box * np.floor(ru[iref] / box)
    rw = ru + shift

    return rw


def dmat_min_image(
    pos: npt.NDArray[np.floating], box: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    diff = pos[:, None, :] - pos[None, :, :]
    diff -= box * np.round(diff / box)
    return np.linalg.norm(diff, axis=-1)


def extra_interaction_mask(
    large_data: SystemData,
    small_data: SystemData,
) -> npt.NDArray[np.bool_]:
    """Generate mask for all pairs between small and large data.

    Only includese interactions between small_data and large_data, not within small_data.
    """
    mask = np.ones(shape=(len(large_data), len(small_data)), dtype=bool)
    sub_start, sub_end = small_data.data_interval()
    mask[sub_start:sub_end, :] = False
    return mask


def intra_interaction_mask(
    small_data: SystemData,
) -> npt.NDArray[np.bool_]:
    """Generate mask for all pairs within small_data only."""
    subset_mask = np.tril(np.ones((len(small_data), len(small_data)), dtype=bool), k=-1)
    return subset_mask


def distance_squares(
    large_data: SystemData,
    small_data: SystemData,
    box: npt.NDArray[FloatPrecision],
    periodic: bool,
) -> npt.NDArray[FloatPrecision]:
    """Calculate MxN distance pair matrix with or without MIC based on periodicity."""
    dist_vecs = large_data.positions[:, None, :] - small_data.positions[None, :, :]
    if periodic:  # needs MIC
        dist_vecs -= box * np.round(dist_vecs / box)
    return np.einsum("mni->mn", dist_vecs**2)
