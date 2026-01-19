from molecsim.data.contributors import SystemPotentials
from molecsim.data.data import SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.data.system import SimSystem, SystemUpdater
from molecsim.physics.mutate import mutate_rotate
from molecsim.physics.geometry import contain_to_box, fuzzyfy
from molecsim.utils.grids import simple_cubic
from molecsim.data.rng import RNG


def randomized_system(
    properties: SystemProperties,
    model: ModelConstants,
    potentials: SystemPotentials,
    rng: RNG,
    grid_function=simple_cubic,
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
        rotated = mutate_rotate(molec_sample, properties, 1, rng)
        rotated.data.positions += fuzzy_node
        contain_to_box(rotated.data, properties)
        rotated.move_to(molecule_slice)
        molecules.append(rotated)
    updater = SystemUpdater(potentials)
    return SimSystem(data, molecules, properties, model, updater)
