"""Dense (no-mask) potentials.

These are for the special case where you can skip the extra/intra masks and
instead compute:

  - extra: full rectangular interaction (data x molecule)
  - intra: only upper triangle within the molecule

The intent is to avoid boolean masking and improve parallel efficiency.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from molecsim.data.contributors import Contribution
from molecsim.data.data import FLOAT_DTYPE, SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import distance_squares_parallel, erfc


# ----------------------------
# Lennard-Jones dense
# ----------------------------


@njit(fastmath=True, parallel=True)
def _lj_extra_dense(
    r2: np.ndarray,
    types_i: np.ndarray,
    types_j: np.ndarray,
    table: np.ndarray,  # (T,T,2) with A,B
) -> float:
    M, N = r2.shape
    acc = 0.0
    for i in prange(M):
        ti = types_i[i]
        local = 0.0
        for j in range(N):
            tj = types_j[j]
            rij2 = r2[i, j]
            Aij = table[ti, tj, 0]
            Bij = table[ti, tj, 1]
            r6 = rij2 * rij2 * rij2
            r12 = r6 * r6
            local += (Aij / r12) - (Bij / r6)
        acc += local
    return acc


@njit(fastmath=True, parallel=True)
def _lj_intra_upper_dense(
    r2: np.ndarray,
    types: np.ndarray,
    table: np.ndarray,
) -> float:
    N = r2.shape[0]
    acc = 0.0
    for i in prange(N):
        ti = types[i]
        local = 0.0
        for j in range(i + 1, N):
            tj = types[j]
            rij2 = r2[i, j]
            Aij = table[ti, tj, 0]
            Bij = table[ti, tj, 1]
            r6 = rij2 * rij2 * rij2
            r12 = r6 * r6
            local += (Aij / r12) - (Bij / r6)
        acc += local
    return acc


@njit(fastmath=True)
def _lennard_jones_dense_numba(
    extra_r2: np.ndarray,
    data_types: np.ndarray,
    mol_types: np.ndarray,
    intra_r2: np.ndarray,
    table: np.ndarray,
) -> float:
    return _lj_extra_dense(extra_r2, data_types, mol_types, table) + _lj_intra_upper_dense(
        intra_r2, mol_types, table
    )


def lennard_jones_dense(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """Dense LJ (no masks), signature-compatible wrapper."""
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    extra_r2 = distance_squares_parallel(
        data, molecule_data, props.box, periodic=props.periodic
    )
    intra_r2 = distance_squares_parallel(
        molecule_data, molecule_data, props.box, periodic=props.periodic
    )

    data_types = np.asarray(data.atoms, dtype=np.int64)
    mol_types = np.asarray(molecule_data.atoms, dtype=np.int64)
    table = np.asarray(model.ljtable.table, dtype=np.float64)

    e = _lennard_jones_dense_numba(
        np.asarray(extra_r2, dtype=np.float64),
        data_types,
        mol_types,
        np.asarray(intra_r2, dtype=np.float64),
        table,
    )
    return float(np.array(e, dtype=FLOAT_DTYPE))


# ----------------------------
# Coulomb dense
# ----------------------------


@njit(fastmath=True, parallel=True)
def _coulomb_extra_dense(
    r2: np.ndarray,
    q_i: np.ndarray,
    q_j: np.ndarray,
    coeff: float,
    cutoff: float,
) -> float:
    M, N = r2.shape
    acc = 0.0
    for i in prange(M):
        qi = q_i[i]
        local = 0.0
        for j in range(N):
            rij = np.sqrt(r2[i, j])
            cut = erfc(rij - cutoff)
            local += (qi * q_j[j]) * (cut / rij)
        acc += local
    return coeff * acc


@njit(fastmath=True, parallel=True)
def _coulomb_intra_upper_dense(
    r2: np.ndarray,
    q: np.ndarray,
    coeff: float,
    cutoff: float,
) -> float:
    N = r2.shape[0]
    acc = 0.0
    for i in prange(N):
        qi = q[i]
        local = 0.0
        for j in range(i + 1, N):
            rij = np.sqrt(r2[i, j])
            cut = erfc(rij - cutoff)
            local += (qi * q[j]) * (cut / rij)
        acc += local
    return coeff * acc


@njit(fastmath=True)
def _coulomb_dense_numba(
    extra_r2: np.ndarray,
    data_q: np.ndarray,
    mol_q: np.ndarray,
    intra_r2: np.ndarray,
    coeff: float,
    cutoff: float,
) -> float:
    return _coulomb_extra_dense(extra_r2, data_q, mol_q, coeff, cutoff) + _coulomb_intra_upper_dense(
        intra_r2, mol_q, coeff, cutoff
    )


def coulomb_dense(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """Dense Coulomb (no masks), signature-compatible wrapper."""
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    extra_r2 = distance_squares_parallel(
        data, molecule_data, props.box, periodic=props.periodic
    )
    intra_r2 = distance_squares_parallel(
        molecule_data, molecule_data, props.box, periodic=props.periodic
    )

    e = _coulomb_dense_numba(
        np.asarray(extra_r2, dtype=np.float64),
        np.asarray(data.charges, dtype=np.float64),
        np.asarray(molecule_data.charges, dtype=np.float64),
        np.asarray(intra_r2, dtype=np.float64),
        float(model.coulomb_coeff),
        float(model.cutoff),
    )
    return float(np.array(e, dtype=FLOAT_DTYPE))
