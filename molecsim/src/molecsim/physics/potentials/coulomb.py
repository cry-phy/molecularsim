from molecsim.data.data import SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.physics.geometry import (
    distance_squares,
    erfc,
    extra_interaction_mask,
    intra_interaction_mask,
)
import numpy as np


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
    dists = np.sqrt(r2)
    mask = extra_interaction_mask(data, molecule_data)
    dists[~mask] = np.inf
    charge_prods = data.charges[:, None] * molecule_data.charges[None, :]
    intra_cut = erfc(dists - model.cutoff)
    extra_potentials = charge_prods / dists * intra_cut
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
    intra_potentials = intra_charge_prods / intra_dists * intra_cut
    intra_part = np.sum(model.coulomb_coeff * intra_potentials)

    return extra_part + intra_part
