from molecsim.data.contributors import Contribution
from molecsim.data.data import SystemData, FLOAT_DTYPE
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import (
    distance_squares,
    distance_squares_parallel,
    extra_interaction_mask,
    intra_interaction_mask,
)
from numba import njit, prange
import numpy as np


def _sum_lj_block(
    r2_block: np.ndarray,
    mask_block: np.ndarray,
    A: float,
    B: float,
) -> float:
    """Sum LJ energy on a masked distance^2 block for fixed (A,B)."""
    if A == 0.0 and B == 0.0:
        return 0.0
    r2 = r2_block.copy()
    r2[~mask_block] = np.inf
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    block = (A / r12) - (B / r6)
    return float(np.sum(block, dtype=FLOAT_DTYPE))


def lennard_jones(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """LJ interaction using per-type A_ij,B_ij from model.lj_for, looping only over present types."""
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    r2 = distance_squares(data, molecule_data, props.box, periodic=props.periodic)
    mask = extra_interaction_mask(data, molecule_data)

    data_types = data.atoms
    mol_types = molecule_data.atoms
    uniq_i = np.unique(data_types)
    uniq_j = np.unique(mol_types)

    extra_sum = 0.0
    for ti in uniq_i:
        Ii = np.where(data_types == ti)[0]
        for tj in uniq_j:
            Aij, Bij = model.lj_for(int(ti), int(tj))
            if Aij == 0.0 and Bij == 0.0:
                continue
            Jj = np.where(mol_types == tj)[0]

            sub_r2 = r2[np.ix_(Ii, Jj)]
            sub_mask = mask[np.ix_(Ii, Jj)]
            extra_sum += _sum_lj_block(sub_r2, sub_mask, Aij, Bij)

    intra_r2 = distance_squares(
        molecule_data, molecule_data, props.box, periodic=props.periodic
    )
    intra_mask = intra_interaction_mask(molecule_data)

    intra_sum = 0.0
    uniq_m = uniq_j
    for ti in uniq_m:
        Ii = np.where(mol_types == ti)[0]
        for tj in uniq_m:
            Aij, Bij = model.lj_for(int(ti), int(tj))
            if Aij == 0.0 and Bij == 0.0:
                continue
            Jj = np.where(mol_types == tj)[0]

            sub_r2 = intra_r2[np.ix_(Ii, Jj)]
            sub_mask = intra_mask[np.ix_(Ii, Jj)]
            intra_sum += _sum_lj_block(sub_r2, sub_mask, Aij, Bij)

    return extra_sum + intra_sum


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
            # Guards are optional if your masks guarantee valid distances.
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
    e_intra = _sum_lj_from_r2_mask_table(
        intra_r2, intra_mask, mol_types, mol_types, table
    )
    return e_extra + e_intra


def lennard_jones_njit(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """No python compatible lennard jones potential."""
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

    table = model.ljtable.table

    extra_r2 = np.asarray(extra_r2, dtype=np.float64)
    intra_r2 = np.asarray(intra_r2, dtype=np.float64)
    extra_mask = np.asarray(extra_mask, dtype=np.bool_)
    intra_mask = np.asarray(intra_mask, dtype=np.bool_)
    table = np.asarray(table, dtype=np.float64)

    e = _lennard_jones_numba_table(
        extra_r2,
        extra_mask,
        data.atoms,
        molecule_data.atoms,
        intra_r2,
        intra_mask,
        table,
    )

    return float(np.array(e, dtype=FLOAT_DTYPE))


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
    return _lj_extra_dense(
        extra_r2, data_types, mol_types, table
    ) + _lj_intra_upper_dense(intra_r2, mol_types, table)


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

    # expects registry-aligned integer type indices 0..T-1
    data_types = np.asarray(data.atoms, dtype=np.int64)
    mol_types = np.asarray(molecule_data.atoms, dtype=np.int64)

    table = np.asarray(model.ljtable.table, dtype=np.float64)

    extra_r2 = np.asarray(extra_r2, dtype=np.float64)
    intra_r2 = np.asarray(intra_r2, dtype=np.float64)

    e = _lennard_jones_dense_numba(extra_r2, data_types, mol_types, intra_r2, table)
    return float(np.array(e, dtype=FLOAT_DTYPE))
