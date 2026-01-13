from molecsim.data.data import SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import (
    distance_squares,
    extra_interaction_mask,
    intra_interaction_mask,
)
import numpy as np


def coulomb(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    *,
    cut_distance: float | None,
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
    if cut_distance is not None:
        close_indices = np.where(r2 <= cut_distance**2)
        r2 = r2[close_indices]
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
    if cut_distance is not None:
        intra_close_indices = np.where(r2 <= cut_distance**2)
        intra_r2 = intra_r2[intra_close_indices]
    intra_distances = np.sqrt(intra_r2)
    intra_mask = intra_interaction_mask(molecule_data)
    intra_distances[~intra_mask] = np.inf
    intra_charge_prods = molecule_data.charges[None, :] * molecule_data.charges[:, None]
    intra_potentials = model.coulomb_coeff * intra_charge_prods / intra_distances
    intra_part = np.sum(intra_potentials)

    return extra_part + intra_part
