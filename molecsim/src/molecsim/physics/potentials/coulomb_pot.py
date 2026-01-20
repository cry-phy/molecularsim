from molecsim.data.contributors import Contribution
from molecsim.data.data import FLOAT_DTYPE, SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import (
    distance_squares,
    distance_squares_parallel,
    erfc,
    extra_interaction_mask,
    intra_interaction_mask,
)
from numba import njit, prange
import numpy as np


def coulomb(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """Calculate coulomb interaction between molecule_data atoms and data atoms."""
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    # need to calculate two parts since molecule might
    # be different from data after mutation, overhead is small
    # since molecule is small in comparison to data

    # Calculate interactions to atoms outside the molecule only
    r2 = distance_squares(data, molecule_data, props.box, periodic=props.periodic)
    dists = np.sqrt(r2)
    mask = extra_interaction_mask(data, molecule_data)
    dists[~mask] = np.inf
    charge_prods = data.charges[:, None] * molecule_data.charges[None, :]
    intra_cut = erfc(dists - model.cutoff)
    extra_potentials = charge_prods / dists * intra_cut / 2
    extra_part = np.sum(model.coulomb_coeff * extra_potentials)

    # Calculate interactions within the molecule only
    intra_r2 = distance_squares(
        molecule_data, molecule_data, props.box, periodic=props.periodic
    )
    intra_dists = np.sqrt(intra_r2)
    intra_mask = intra_interaction_mask(molecule_data)
    intra_dists[~intra_mask] = np.inf
    intra_charge_prods = molecule_data.charges[None, :] * molecule_data.charges[:, None]
    intra_cut = erfc(intra_dists - model.cutoff)
    intra_potentials = intra_charge_prods / intra_dists * intra_cut / 2
    intra_part = np.sum(model.coulomb_coeff * intra_potentials)

    return extra_part + intra_part


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
            # mask should already kill self terms; this is just safety.
            if rij2 <= 0.0 or not np.isfinite(rij2):
                continue

            rij = np.sqrt(rij2)
            qj = charges_j[j]

            # Match your original: erfc(dists - model.cutoff)
            cut = erfc(rij - cutoff)

            # Match your original: qiqj / r * cut / 2
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
    """Numba-compatible coulomb potential with same signature as coulomb()."""
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

    extra_r2 = np.asarray(extra_r2, dtype=np.float64)
    intra_r2 = np.asarray(intra_r2, dtype=np.float64)
    extra_mask = np.asarray(extra_mask, dtype=np.bool_)
    intra_mask = np.asarray(intra_mask, dtype=np.bool_)

    data_charges = np.asarray(data.charges, dtype=np.float64)
    mol_charges = np.asarray(molecule_data.charges, dtype=np.float64)

    e = _coulomb_numba(
        extra_r2,
        extra_mask,
        data_charges,
        mol_charges,
        intra_r2,
        intra_mask,
        float(model.coulomb_coeff),
        float(model.cutoff),
    )

    return float(np.array(e, dtype=FLOAT_DTYPE))


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
            # matches your previous: erfc(r - cutoff)
            cut = erfc(rij - cutoff)
            local += (qi * q_j[j]) * (cut / rij)
        acc += local
    # NOTE: no /2 here because extra is not symmetric and you likely want full contribution.
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
    return _coulomb_extra_dense(
        extra_r2, data_q, mol_q, coeff, cutoff
    ) + _coulomb_intra_upper_dense(intra_r2, mol_q, coeff, cutoff)


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

    data_q = np.asarray(data.charges, dtype=np.float64)
    mol_q = np.asarray(molecule_data.charges, dtype=np.float64)

    extra_r2 = np.asarray(extra_r2, dtype=np.float64)
    intra_r2 = np.asarray(intra_r2, dtype=np.float64)

    e = _coulomb_dense_numba(
        extra_r2,
        data_q,
        mol_q,
        intra_r2,
        float(model.coulomb_coeff),
        float(model.cutoff),
    )
    return float(np.array(e, dtype=FLOAT_DTYPE))
