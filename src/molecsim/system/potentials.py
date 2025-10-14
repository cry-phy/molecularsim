"""All force Calculations should be made using a function following PotContributor Signature."""

from molecsim.system.data import FloatPrecision, SystemData, FLOAT_DTYPE
import numpy.typing as npt
import numpy as np


def extra_interaction_mask(
    large_data: SystemData,
    small_data: SystemData,
) -> npt.NDArray[np.bool_]:
    """Generate mask for all pairs between small and large data.
    
    Only includese interactions between small_data and large_data, not within small_data."""
    mask = np.ones(shape=(len(large_data), len(small_data)), dtype=bool)
    sub_start, sub_end = small_data.data_interval()
    mask[sub_start:sub_end, :] = False
    return mask

def intra_interaction_mask(
    small_data: SystemData,
) -> npt.NDArray[np.bool_]:
    """Generate mask for all pairs within small_data only."""
    subset_mask = np.tril(np.ones(len(small_data), k=-1, dtype=bool))
    return subset_mask

def harmonic_well(
    data: SystemData,
    curvature: npt.NDArray[FloatPrecision],
    center: npt.NDArray[FloatPrecision],
) -> float:
    """Quadratic well: sum_i ( (r_i - c)^2 * k ) over all atoms."""
    rel = data.positions - center.astype(FLOAT_DTYPE, copy=False)
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
    coeff: float = 1,
) -> float:
    """Calculate coulomb interaction between molecule_data atoms and data atoms.

    This is linear in mem and can be bocome large MxN!!!
    """
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )
    # need to calculate two parts since molecule might
    # be different from data after mutation, overhead is small
    # since molecule is small in comparison to data

    # Calculate interactions to atoms outside the molecule only
    dist_vecs = data.positions[:, None, :] - molecule_data.positions[None, :, :]
    distances = np.sqrt(np.einsum("mni->mn", dist_vecs**2))
    mask = extra_interaction_mask(data, molecule_data)
    distances[~mask] = np.inf
    charge_prods = molecule_data.charges[:, None] * molecule_data.charges[None, :]
    extra_potentials = coeff * charge_prods / distances
    extra_part = np.sum(extra_potentials)
    
    # Calculate interactions within the molecule only
    intra_dist_vecs = molecule_data.positions[:, None, :] - molecule_data.positions[None, :, :]
    intra_distances = np.sqrt(np.einsum("mni->mn", intra_dist_vecs**2))
    intra_mask = intra_interaction_mask(molecule_data)
    intra_distances[~intra_mask] = np.inf
    intra_charge_prods = molecule_data.charges[:, None] * molecule_data.charges[None, :]
    intra_potentials = coeff * intra_charge_prods / intra_distances
    intra_part = np.sum(intra_potentials)
    return extra_part + intra_part


def lennard_jones(
    data: SystemData,
    molecule_data: SystemData,
    A: float = 1,
    B: float = 1,
) -> float:
    """Calculate lennard jones interaction between molecule_data atoms and data atoms.

    This is linear in mem and can be bocome large MxN!!!
    Also missing here is connected atom filtering since LJ should only apply to non-connected atoms.
    """
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )
    # need to calculate two parts since molecule might
    # be different from data after mutation, overhead is small
    # since molecule is small in comparison to data

    # Calculate interactions to atoms outside the molecule only
    dist_vecs = data.positions[:, None, :] - molecule_data.positions[None, :, :]
    r2 = np.einsum("mni->mn", dist_vecs**2)
    mask = extra_interaction_mask(data, molecule_data)
    r2[~mask] = np.inf
    r6 = r2**3
    r12 = r6**2
    potentials = A * r12 - B * r6
    extra_part = np.sum(potentials)
    
    # Calculate interactions within the molecule only
    intra_dist_vecs = molecule_data.positions[:, None, :] - molecule_data.positions[None, :, :]
    intra_r2 = np.einsum("mni->mn", intra_dist_vecs**2)
    intra_mask = intra_interaction_mask(molecule_data)
    intra_r2[~intra_mask] = np.inf
    intra_r6 = intra_r2**3
    intra_r12 = intra_r6**2
    intra_potentials = A * intra_r12 - B * intra_r6
    intra_part = np.sum(intra_potentials)
    
    return extra_part + intra_part