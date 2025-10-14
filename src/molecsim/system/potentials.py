"""All force Calculations should be made using a function following PotContributor Signature."""

from molecsim.system.data import FloatPrecision, SystemData, FLOAT_DTYPE
import numpy.typing as npt
import numpy as np


def full_interaction_mask(
    large_data: SystemData,
    small_data: SystemData,
) -> npt.NDArray[np.bool_]:
    """Generate full mask for all pairs between small and large data."""
    mask = np.ones(shape=(len(large_data), len(small_data)), dtype=bool)
    subset_mask = np.triu(np.ones(len(small_data), dtype=bool))
    sub_start, sub_end = small_data.data_interval()
    mask[sub_start:sub_end, :] = ~subset_mask
    return mask


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
    dist_vecs = data.positions[:, None, :] - molecule_data.positions[None, :, :]
    distances = np.sqrt(np.einsum("mni->mn", dist_vecs**2))
    mask = full_interaction_mask(data, molecule_data)
    distances[~mask] = np.inf
    charge_prods = molecule_data.charges[:, None] * molecule_data.charges[None, :]
    potentials = coeff * charge_prods / distances
    return np.sum(potentials)


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
    dist_vecs = data.positions[:, None, :] - molecule_data.positions[None, :, :]
    r2 = np.einsum("mni->mn", dist_vecs**2)
    mask = full_interaction_mask(data, molecule_data)
    r2[~mask] = np.inf
    r6 = r2**3
    r12 = r6**2
    potentials = A * r12 - B * r6
    return np.sum(potentials)