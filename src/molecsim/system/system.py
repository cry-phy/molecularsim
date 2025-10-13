from __future__ import annotations
from collections.abc import Iterable
from molecsim.system.data import SystemData, SystemProperties
from molecsim.system.molecule import Molecule
from molecsim.system.mutate import mutate_rotate, mutate_translate
from molecsim.utils.geometry import fuzzyfy, simple_cubic
from molecsim.utils.rng import RNG


class SimSystem:
    """System contains the underlying data Structure and general Stucture information."""

    def __init__(
        self,
        contigous_data: SystemData,
        molecules: Iterable[Molecule],
        properties: SystemProperties,
    ) -> None:
        """Initialize Simulation System."""
        self._contigous_data: SystemData = contigous_data
        self._molecules: Iterable[Molecule] = molecules

        if len(self._contigous_data.positions[0]) != len(properties.box):
            raise ValueError(
                "Dimension of bounding box must match dimension of position."
            )
        self.properties = properties


def randomized_system(
    properties: SystemProperties, rng: RNG, grid_function=simple_cubic
) -> SimSystem:
    """Generate a system from properties with randomized initial data."""
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
        rotated_molecule = mutate_rotate(molec_sample, 1, rng)
        rotated_molecule.data.positions += fuzzy_node
        rotated_molecule.move_to(molecule_slice)
        molecules.append(rotated_molecule)
    return SimSystem(data, molecules, properties)
