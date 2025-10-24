from molecsim.system.data import SystemData, SystemProperties
from molecsim.system.mutate import mutate_rotate
from molecsim.system.system import SimSystem
from molecsim.utils.geometry import fuzzyfy, contain_to_box
from molecsim.utils.grids import simple_cubic
from molecsim.utils.rng import RNG
import numpy as np


def randomized_system(
    properties: SystemProperties, rng: RNG, grid_function=simple_cubic
) -> SimSystem:
    """Generate random System with grid function arrangement as basis."""
    total_molecules = sum(properties.molecules.values())
    total_atoms = sum(
        molecule.size() * number for molecule, number in properties.molecules.items()
    )
    nodes = grid_function(total_molecules, properties.box, rng, properties.periodic)
    fuzzy_nodes = fuzzyfy(nodes, properties.box, rng)
    molecule_types = [
        molecule
        for molecule, number in properties.molecules.items()
        for _ in range(number)
    ]
    data = SystemData(total_atoms)
    molecules = []
    for molec_sample, fuzzy_node in zip(molecule_types, fuzzy_nodes, strict=True):
        molecule_slice = data.allocate_molecule(molec_sample.size())
        print(
            "before",
            np.linalg.norm(
                molec_sample.data.positions[:, None, :]
                - molec_sample.data.positions[None, :, :],
                axis=-1,
            ),
        )
        rotated = mutate_rotate(molec_sample, properties, 1, 1, rng)
        print(
            "after",
            np.linalg.norm(
                rotated.data.positions[:, None, :] - rotated.data.positions[None, :, :],
                axis=-1,
            ),
        )
        rotated.data.positions += fuzzy_node
        contain_to_box(rotated.data, properties)
        rotated.move_to(molecule_slice)
        molecules.append(rotated)
    return SimSystem(data, molecules, properties)
