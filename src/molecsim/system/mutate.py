from collections.abc import Callable
from molecsim.utils.rng import RNG
from molecsim.system.molecule import Molecule

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
    new_data.positions = (
        rng.rotation_matrix(magnitude) @ rel_positions + molecule.center
    )
    return Molecule(new_data)


MUTATION_FUNCTIONS: list[MutateFunction] = [mutate_translate, mutate_rotate]
