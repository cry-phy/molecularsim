"""Simple Numba-accelerated potentials.

These implementations keep the same physics as the classic versions, but push the
inner loops into Numba. They still compute distance matrices + masks, but the
energy accumulation is done in nopython mode.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from molecsim.data.contributors import Contribution
from molecsim.data.data import FLOAT_DTYPE, SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import (
    distance_squares,
    distance_squares_njit,
    erfc,
    extra_interaction_mask,
    intra_interaction_mask,
)


# ----------------------------
# Lennard-Jones (table-driven)
# ----------------------------


@njit(fastmath=True)
def _sum_lj_from_r2_mask_table(
    r2: np.ndarray,
    mask: np.ndarray,
    types_i: np.ndarray,
    types_j: np.ndarray,
    table: np.ndarray,
) -> float:
    acc = 0.0
    M, N = r2.shape
    for i in range(M):
        ti = types_i[i]
        for j in range(N):
            if not mask[i, j]:
                continue
            tj = types_j[j]

            Aij = table[ti, tj, 0]
            Bij = table[ti, tj, 1]
            if Aij == 0.0 and Bij == 0.0:
                continue

            rij2 = r2[i, j]
            if rij2 <= 0.0 or not np.isfinite(rij2):
                continue

            r6 = rij2 * rij2 * rij2
            r12 = r6 * r6
            acc += (Aij / r12) - (Bij / r6)
    return acc


@njit(fastmath=True)
def _lennard_jones_numba_table(
    extra_r2: np.ndarray,
    extra_mask: np.ndarray,
    data_types: np.ndarray,
    mol_types: np.ndarray,
    intra_r2: np.ndarray,
    intra_mask: np.ndarray,
    table: np.ndarray,
) -> float:
    e_extra = _sum_lj_from_r2_mask_table(
        extra_r2, extra_mask, data_types, mol_types, table
    )
    e_intra = _sum_lj_from_r2_mask_table(intra_r2, intra_mask, mol_types, mol_types, table)
    return e_extra + e_intra


def lennard_jones_njit(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """Numba-compatible Lennard-Jones potential with same signature as classic."""
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    extra_r2 = distance_squares_njit(
        data, molecule_data, props.box, periodic=props.periodic
    )
    extra_mask = extra_interaction_mask(data, molecule_data)

    intra_r2 = distance_squares_njit(
        molecule_data, molecule_data, props.box, periodic=props.periodic
    )
    intra_mask = intra_interaction_mask(molecule_data)

    # registry-aligned integer type indices 0..T-1
    table = np.asarray(model.ljtable.table, dtype=np.float64)

    e = _lennard_jones_numba_table(
        np.asarray(extra_r2, dtype=np.float64),
        np.asarray(extra_mask, dtype=np.bool_),
        np.asarray(data.atoms, dtype=np.int64),
        np.asarray(molecule_data.atoms, dtype=np.int64),
        np.asarray(intra_r2, dtype=np.float64),
        np.asarray(intra_mask, dtype=np.bool_),
        table,
    )

    return float(np.array(e, dtype=FLOAT_DTYPE))


# ----------------------------
# Coulomb (mask-driven)
# ----------------------------


@njit(fastmath=True)
def _sum_coulomb_from_r2_mask(
    r2: np.ndarray,
    mask: np.ndarray,
    charges_i: np.ndarray,
    charges_j: np.ndarray,
    coulomb_coeff: float,
    cutoff: float,
) -> float:
    acc = 0.0
    M, N = r2.shape
    for i in range(M):
        qi = charges_i[i]
        for j in range(N):
            if not mask[i, j]:
                continue

            rij2 = r2[i, j]
            if rij2 <= 0.0 or not np.isfinite(rij2):
                continue

            rij = np.sqrt(rij2)
            qj = charges_j[j]
            cut = erfc(rij - cutoff)  # matches classic: erfc(r - cutoff)
            acc += (qi * qj) * (cut / rij) * 0.5

    return coulomb_coeff * acc


@njit(fastmath=True)
def _coulomb_numba(
    extra_r2: np.ndarray,
    extra_mask: np.ndarray,
    data_charges: np.ndarray,
    mol_charges: np.ndarray,
    intra_r2: np.ndarray,
    intra_mask: np.ndarray,
    coulomb_coeff: float,
    cutoff: float,
) -> float:
    extra_part = _sum_coulomb_from_r2_mask(
        extra_r2,
        extra_mask,
        data_charges,
        mol_charges,
        coulomb_coeff,
        cutoff,
    )
    intra_part = _sum_coulomb_from_r2_mask(
        intra_r2,
        intra_mask,
        mol_charges,
        mol_charges,
        coulomb_coeff,
        cutoff,
    )
    return extra_part + intra_part


def coulomb_njit(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """Numba-compatible Coulomb potential with same signature as classic."""
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    extra_r2 = distance_squares(data, molecule_data, props.box, periodic=props.periodic)
    extra_mask = extra_interaction_mask(data, molecule_data)

    intra_r2 = distance_squares(
        molecule_data, molecule_data, props.box, periodic=props.periodic
    )
    intra_mask = intra_interaction_mask(molecule_data)

    e = _coulomb_numba(
        np.asarray(extra_r2, dtype=np.float64),
        np.asarray(extra_mask, dtype=np.bool_),
        np.asarray(data.charges, dtype=np.float64),
        np.asarray(molecule_data.charges, dtype=np.float64),
        np.asarray(intra_r2, dtype=np.float64),
        np.asarray(intra_mask, dtype=np.bool_),
        float(model.coulomb_coeff),
        float(model.cutoff),
    )

    return float(np.array(e, dtype=FLOAT_DTYPE))
