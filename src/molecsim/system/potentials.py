"""All force Calculations should be made using a function following PotContributor Signature."""

from molecsim.system.data import FloatPrecision, SystemData, FLOAT_DTYPE
import numpy.typing as npt
import numpy as np


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


def coulomb_bigmem(
    data: SystemData,
    molecule_data: SystemData,
    coeff: float = 1,
) -> float:
    """Calculate coulomb interaction between molecule_data atoms and data atoms.

    This is bigmem MxN!!!
    """
    if len(molecule_data) > len(data):
        raise ValueError(
            f"Data must be larger or equal to molecule_data: data {len(data)}, molecule_data: {len(molecule_data)}"
        )
    dist_vecs = data.positions[:, None, :] - molecule_data.positions[None, :, :]
    distances = np.sqrt(np.einsum("mni->mn", dist_vecs**2))

    # generate mask to mask i=j
    selection = np.zeros(len(data), dtype=bool)
    start, end = molecule_data.data_interval()
    selection[np.arange(start=start, stop=end)] = True
    sel_row = selection[:, None]
    sel_col = selection[None, :]
    cross_mask = sel_row ^ sel_col

    # include intermolecular
    subset_mask = np.triu(sel_row & sel_col, k=1)
    mask = cross_mask | subset_mask

    distances[np.triu_indices_from(distances)] = np.inf
    charge_prods = molecule_data.charges[:, None] * molecule_data.charges[None, :]
    potentials = coeff * charge_prods / distances
    return np.sum(potentials)
