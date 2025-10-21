from collections.abc import Callable, Sequence
from molecsim.system.data import SystemProperties
from molecsim.utils.geometry import contain_to_box
from molecsim.utils.rng import RNG
from molecsim.system.molecule import Molecule
from molecsim.system.system import SimSystem

type MutateFunction = Callable[[Molecule, float, RNG], Molecule]


def mutate_translate(molecule: Molecule, magnitude: float, rng: RNG) -> Molecule:
    """Mutate molecule by moving it."""
    new_data = molecule.data.deepcopy()
    shift = magnitude * rng.random_direction()
    new_data.positions += shift
    return Molecule(new_data)


def mutate_rotate(molecule: Molecule, magnitude: float, rng: RNG) -> Molecule:
    """Mutate molecule by Rotation around Molecule.center."""
    new_data = molecule.data.deepcopy()
    rel_positions = new_data.positions - molecule.center
    new_data.positions = rel_positions @ rng.rotation_matrix() + molecule.center
    return Molecule(new_data)


def propose_new(
    system: SimSystem,
    mutation_functions: Sequence[MutateFunction],
    magnitude: float,
    rng: RNG,
) -> tuple[Molecule, Molecule]:
    """Propose change in system based on change of one molecule.

    Returns tuple[old, new].
    """
    molecule_index = rng.randint(len(system.molecules))
    mutation_function_index = rng.randint(len(mutation_functions))
    old_molecule = system.molecules[molecule_index]
    mutation_fn = mutation_functions[mutation_function_index]
    new_molecule = mutation_fn(old_molecule, magnitude, rng)
    contain_to_box(new_molecule.data, system.properties)
    return old_molecule, new_molecule
