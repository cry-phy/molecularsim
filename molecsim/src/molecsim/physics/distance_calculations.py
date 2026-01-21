"""Distance calculation backends.

This file centralizes the different distance-matrix implementations so the
potential modules can stay small and focused.

Naming convention:

  - distance_squares: classic NumPy vectorized (round + einsum)
  - distance_squares_njit: same algorithm but compiled with Numba
  - distance_squares_parallel: explicit parallel kernel (Numba, no einsum)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange

from molecsim.data.data import FloatPrecision, SystemData, StructureArray, FLOAT_DTYPE


@njit(fastmath=True, parallel=True)
def _distance_squares_pos_pos(pos_i, pos_j, box, periodic):
    M = pos_i.shape[0]
    N = pos_j.shape[0]
    out = np.empty((M, N), dtype=FLOAT_DTYPE)

    if periodic:
        bx = box[0]
        by = box[1]
        bz = box[2]
        for i in prange(M):
            xi = pos_i[i, 0]
            yi = pos_i[i, 1]
            zi = pos_i[i, 2]
            for j in range(N):
                dx = xi - pos_j[j, 0]
                dy = yi - pos_j[j, 1]
                dz = zi - pos_j[j, 2]
                dx -= bx * np.rint(dx / bx)
                dy -= by * np.rint(dy / by)
                dz -= bz * np.rint(dz / bz)
                out[i, j] = dx * dx + dy * dy + dz * dz
    else:
        for i in prange(M):
            xi = pos_i[i, 0]
            yi = pos_i[i, 1]
            zi = pos_i[i, 2]
            for j in range(N):
                dx = xi - pos_j[j, 0]
                dy = yi - pos_j[j, 1]
                dz = zi - pos_j[j, 2]
                out[i, j] = dx * dx + dy * dy + dz * dz

    return out


def distance_squares_parallel(
    large_data: SystemData,
    small_data: SystemData,
    box: npt.NDArray[FloatPrecision],
    periodic: bool,
) -> npt.NDArray[FloatPrecision]:
    """Compute MxN distance^2 matrix using an explicit parallel kernel."""
    pos_i = np.asarray(large_data.positions, dtype=FLOAT_DTYPE)
    pos_j = np.asarray(small_data.positions, dtype=FLOAT_DTYPE)
    box = np.asarray(box, dtype=FLOAT_DTYPE)
    return _distance_squares_pos_pos(pos_i, pos_j, box, periodic)


@njit(fastmath=True)
def _distance_squares_njit(
    large_positions: StructureArray,
    small_positions: StructureArray,
    box: npt.NDArray[FloatPrecision],
    periodic: bool,
) -> npt.NDArray[FloatPrecision]:
    """Compute MxN distance^2 in nopython mode.

    Numba does not support `np.einsum` here, so we use an explicit kernel.
    This also avoids allocating the intermediate (M,N,3) `dist_vecs` array.
    """
    M = large_positions.shape[0]
    N = small_positions.shape[0]
    out = np.empty((M, N), dtype=FLOAT_DTYPE)

    if periodic:
        bx = box[0]
        by = box[1]
        bz = box[2]
        for i in range(M):
            xi = large_positions[i, 0]
            yi = large_positions[i, 1]
            zi = large_positions[i, 2]
            for j in range(N):
                dx = xi - small_positions[j, 0]
                dy = yi - small_positions[j, 1]
                dz = zi - small_positions[j, 2]
                # MIC: use rint (supported by numba) instead of round
                dx -= bx * np.rint(dx / bx)
                dy -= by * np.rint(dy / by)
                dz -= bz * np.rint(dz / bz)
                out[i, j] = dx * dx + dy * dy + dz * dz
    else:
        for i in range(M):
            xi = large_positions[i, 0]
            yi = large_positions[i, 1]
            zi = large_positions[i, 2]
            for j in range(N):
                dx = xi - small_positions[j, 0]
                dy = yi - small_positions[j, 1]
                dz = zi - small_positions[j, 2]
                out[i, j] = dx * dx + dy * dy + dz * dz

    return out


def distance_squares_njit(
    large_data: SystemData,
    small_data: SystemData,
    box: npt.NDArray[FloatPrecision],
    periodic: bool,
) -> npt.NDArray[FloatPrecision]:
    """Classic round+einsum distance^2 but compiled with Numba."""
    return _distance_squares_njit(
        large_data.positions,
        small_data.positions,
        np.asarray(box, dtype=FLOAT_DTYPE),
        periodic,
    )


def distance_squares(
    large_data: SystemData,
    small_data: SystemData,
    box: npt.NDArray[FloatPrecision],
    periodic: bool,
) -> npt.NDArray[FloatPrecision]:
    """Classic NumPy vectorized distance^2 (round + einsum)."""
    dist_vecs = large_data.positions[:, None, :] - small_data.positions[None, :, :]
    if periodic:
        dist_vecs -= box * np.round(dist_vecs / box)
    return np.einsum("mni->mn", dist_vecs**2)
