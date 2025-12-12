"""All force Calculations should be made using a function following PotContributor Signature."""

from molecsim.data.data import SystemData, FloatPrecision, FLOAT_DTYPE
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import (
    distance_squares,
    extra_interaction_mask,
    intra_interaction_mask,
)
import numpy.typing as npt
import numpy as np


def harmonic_well(
    data: SystemData,
    molecule_data: SystemData,
    curvature: npt.NDArray[FloatPrecision] = np.full(1, 1),
    center: npt.NDArray[FloatPrecision] = np.full(1, 0),
) -> float:
    """Quadratic well: sum_i ( (r_i - c)^2 * k ) over molecule_data."""
    rel = molecule_data.positions - center.astype(FLOAT_DTYPE, copy=False)
    single = (rel * rel) * curvature.astype(FLOAT_DTYPE, copy=False)
    return float(np.sum(single, dtype=FLOAT_DTYPE))


def harmonic_well_gpu(
    data: SystemData,
    curvature: npt.NDArray[FloatPrecision],
    center: npt.NDArray[FloatPrecision],
) -> float:
    """GPU: Quadratic well: sum_i ( (r_i - c)^2 * k ) over all atoms."""
    ...


def coulomb(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
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
    distances = np.sqrt(r2)
    mask = extra_interaction_mask(data, molecule_data)
    distances[~mask] = np.inf
    charge_prods = data.charges[:, None] * molecule_data.charges[None, :]
    extra_potentials = model.coulomb_coeff * charge_prods / distances
    extra_part = np.sum(extra_potentials)

    # Calculate interactions within the molecule only
    intra_r2 = distance_squares(
        molecule_data, molecule_data, props.box, periodic=props.periodic
    )
    intra_distances = np.sqrt(intra_r2)
    intra_mask = intra_interaction_mask(molecule_data)
    intra_distances[~intra_mask] = np.inf
    intra_charge_prods = molecule_data.charges[None, :] * molecule_data.charges[:, None]
    intra_potentials = model.coulomb_coeff * intra_charge_prods / intra_distances
    intra_part = np.sum(intra_potentials)

    return extra_part + intra_part


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
