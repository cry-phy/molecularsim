"""Streaming (distance-on-the-fly) potentials.

These versions avoid building an MxN distance matrix by computing distances
inside the Numba loops.
"""

from __future__ import annotations

from numba import njit, prange
import numpy as np

from molecsim.data.contributors import Contribution
from molecsim.data.data import SystemData, FLOAT_DTYPE
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import erfc


@njit(fastmath=True, parallel=True)
def _lj_extra_stream(
    data_pos,
    data_types,
    mol_pos,
    mol_types,
    box,
    periodic,
    table,
):
    M = data_pos.shape[0]
    N = mol_pos.shape[0]
    acc = 0.0

    for i in prange(M):
        xi = data_pos[i, 0]
        yi = data_pos[i, 1]
        zi = data_pos[i, 2]
        ti = data_types[i]
        local = 0.0

        for j in range(N):
            dx = xi - mol_pos[j, 0]
            dy = yi - mol_pos[j, 1]
            dz = zi - mol_pos[j, 2]
            if periodic:
                dx -= box[0] * np.rint(dx / box[0])
                dy -= box[1] * np.rint(dy / box[1])
                dz -= box[2] * np.rint(dz / box[2])

            r2 = dx * dx + dy * dy + dz * dz
            tj = mol_types[j]
            A = table[ti, tj, 0]
            B = table[ti, tj, 1]

            r6 = r2 * r2 * r2
            r12 = r6 * r6
            local += (A / r12) - (B / r6)

        acc += local

    return acc


@njit(fastmath=True)
def _lj_intra_upper_stream(
    mol_pos,
    mol_types,
    box,
    periodic,
    table,
):
    N = mol_pos.shape[0]
    acc = 0.0

    for i in range(N):
        xi = mol_pos[i, 0]
        yi = mol_pos[i, 1]
        zi = mol_pos[i, 2]
        ti = mol_types[i]
        for j in range(i + 1, N):
            dx = xi - mol_pos[j, 0]
            dy = yi - mol_pos[j, 1]
            dz = zi - mol_pos[j, 2]
            if periodic:
                dx -= box[0] * np.rint(dx / box[0])
                dy -= box[1] * np.rint(dy / box[1])
                dz -= box[2] * np.rint(dz / box[2])

            r2 = dx * dx + dy * dy + dz * dz
            tj = mol_types[j]
            A = table[ti, tj, 0]
            B = table[ti, tj, 1]

            r6 = r2 * r2 * r2
            r12 = r6 * r6
            acc += (A / r12) - (B / r6)

    return acc


def lennard_jones_stream(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """Dense LJ, streaming distances (no r2 matrix)."""

    data_pos = np.asarray(data.positions, dtype=np.float64)
    mol_pos = np.asarray(molecule_data.positions, dtype=np.float64)
    data_types = np.asarray(data.atoms, dtype=np.int64)
    mol_types = np.asarray(molecule_data.atoms, dtype=np.int64)
    box = np.asarray(props.box, dtype=np.float64)
    table = np.asarray(model.ljtable.table, dtype=np.float64)
    periodic = props.periodic

    e_extra = _lj_extra_stream(
        data_pos,
        data_types,
        mol_pos,
        mol_types,
        box,
        periodic,
        table,
    )
    e_intra = _lj_intra_upper_stream(
        mol_pos,
        mol_types,
        box,
        periodic,
        table,
    )

    return float(np.array(e_extra + e_intra, dtype=FLOAT_DTYPE))


# Backwards compatible alias (old name)
lennard_jones_dense_stream = lennard_jones_stream


# ----------------------------
# Coulomb stream (matches dense structure; no masks)
# ----------------------------


@njit(fastmath=True, parallel=True)
def _coulomb_extra_stream(
    data_pos,
    data_q,
    mol_pos,
    mol_q,
    box,
    periodic,
    coulomb_coeff: float,
    cutoff: float,
):
    M = data_pos.shape[0]
    N = mol_pos.shape[0]
    acc = 0.0

    for i in prange(M):
        xi = data_pos[i, 0]
        yi = data_pos[i, 1]
        zi = data_pos[i, 2]
        qi = data_q[i]
        local = 0.0

        for j in range(N):
            dx = xi - mol_pos[j, 0]
            dy = yi - mol_pos[j, 1]
            dz = zi - mol_pos[j, 2]
            if periodic:
                dx -= box[0] * np.rint(dx / box[0])
                dy -= box[1] * np.rint(dy / box[1])
                dz -= box[2] * np.rint(dz / box[2])

            r2 = dx * dx + dy * dy + dz * dz
            r = np.sqrt(r2)
            cut = erfc(r - cutoff)
            local += (qi * mol_q[j]) * (cut / r)

        acc += local

    return coulomb_coeff * acc


@njit(fastmath=True)
def _coulomb_intra_upper_stream(
    mol_pos,
    mol_q,
    box,
    periodic,
    coulomb_coeff: float,
    cutoff: float,
):
    N = mol_pos.shape[0]
    acc = 0.0
    for i in range(N):
        xi = mol_pos[i, 0]
        yi = mol_pos[i, 1]
        zi = mol_pos[i, 2]
        qi = mol_q[i]
        for j in range(i + 1, N):
            dx = xi - mol_pos[j, 0]
            dy = yi - mol_pos[j, 1]
            dz = zi - mol_pos[j, 2]
            if periodic:
                dx -= box[0] * np.rint(dx / box[0])
                dy -= box[1] * np.rint(dy / box[1])
                dz -= box[2] * np.rint(dz / box[2])

            r2 = dx * dx + dy * dy + dz * dz
            r = np.sqrt(r2)
            cut = erfc(r - cutoff)
            acc += (qi * mol_q[j]) * (cut / r)

    return coulomb_coeff * acc


def coulomb_stream(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """Dense Coulomb, streaming distances (no r2 matrix, no masks)."""

    data_pos = np.asarray(data.positions, dtype=np.float64)
    mol_pos = np.asarray(molecule_data.positions, dtype=np.float64)
    data_q = np.asarray(data.charges, dtype=np.float64)
    mol_q = np.asarray(molecule_data.charges, dtype=np.float64)
    box = np.asarray(props.box, dtype=np.float64)
    periodic = props.periodic

    e_extra = _coulomb_extra_stream(
        data_pos,
        data_q,
        mol_pos,
        mol_q,
        box,
        periodic,
        float(model.coulomb_coeff),
        float(model.cutoff),
    )
    e_intra = _coulomb_intra_upper_stream(
        mol_pos,
        mol_q,
        box,
        periodic,
        float(model.coulomb_coeff),
        float(model.cutoff),
    )

    return float(np.array(e_extra + e_intra, dtype=FLOAT_DTYPE))
