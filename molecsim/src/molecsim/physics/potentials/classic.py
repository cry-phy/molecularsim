"""Classic (NumPy) potentials.

These are the original vectorized NumPy implementations (mask + distance matrix).
Keep these around as a correctness reference and for quick prototyping.
"""

from __future__ import annotations

import numpy as np

from molecsim.data.contributors import Contribution
from molecsim.data.data import FLOAT_DTYPE, SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import (
    distance_squares,
    erfc,
    extra_interaction_mask,
    intra_interaction_mask,
)


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
    """LJ interaction using per-type A_ij,B_ij from model.lj_for.

    This is the classic NumPy version:
    - compute distance matrix
    - apply interaction masks
    - sum only over present types (avoids full T^2 loops)
    """
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


def coulomb(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    contribution: Contribution | None = None,
) -> float:
    """Classic Coulomb interaction between molecule_data atoms and data atoms."""
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )

    # Extra (data vs molecule, excluding molecule-within-data indices)
    r2 = distance_squares(data, molecule_data, props.box, periodic=props.periodic)
    dists = np.sqrt(r2)
    mask = extra_interaction_mask(data, molecule_data)
    dists[~mask] = np.inf
    charge_prods = data.charges[:, None] * molecule_data.charges[None, :]
    intra_cut = erfc(dists - model.cutoff)
    extra_potentials = charge_prods / dists * intra_cut / 2
    extra_part = np.sum(model.coulomb_coeff * extra_potentials)

    # Intra (within molecule)
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

    return float(extra_part + intra_part)
