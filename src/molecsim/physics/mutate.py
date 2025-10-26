from molecsim.data.system import SimSystem
from molecsim.data.properties import SystemProperties
from molecsim.data.molecule import Molecule
from molecsim.physics.geometry import contain_to_box, rotate_rigid_pbc
from molecsim.data.rng import RNG
from collections.abc import Callable, Sequence
import numpy as np
import numpy.typing as npt

type MutateFunction = Callable[[Molecule, SystemProperties, float, RNG], Molecule]


def mutate_translate(
    molecule: Molecule,
    props: SystemProperties,
    coeff: float,
    rng: RNG,
) -> Molecule:
    """Mutate molecule by moving it."""
    new_molecule = molecule.copy_data()
    shift = coeff * rng.random() * rng.random_direction()
    new_molecule.data.positions += shift
    return new_molecule


def mutate_rotate(
    molecule: Molecule,
    props: SystemProperties,
    coeff: float,
    rng: RNG,
) -> Molecule:
    """Mutate molecule by Rotation around Molecule.center."""
    new = molecule.copy_data()
    R = rng.rotation_matrix(coeff)
    pos = molecule.data.positions
    box = props.box
    rotated = (
        rotate_rigid_pbc(pos, box, R, iref=0)
        if props.periodic
        else ((pos - molecule.center) @ R.T + molecule.center)
    )
    np.copyto(new.data.positions, rotated)
    return new


def propose_new(
    system: SimSystem,
    mutation_functions: Sequence[MutateFunction],
    magnitude_coefficients: Sequence[float] | npt.NDArray,
    rng: RNG,
) -> tuple[Molecule, Molecule, int]:
    """Propose change in system based on change of one molecule.

    Returns tuple[old, new, mut_fn_index].
    """
    molecule_index = rng.randint(len(system.molecules))
    mutation_function_index = rng.randint(len(mutation_functions))
    mutation_function_coeff = magnitude_coefficients[mutation_function_index]
    old_molecule = system.molecules[molecule_index]
    mutation_fn = mutation_functions[mutation_function_index]
    new_molecule = mutation_fn(
        old_molecule,
        system.properties,
        mutation_function_coeff,
        rng,
    )
    contain_to_box(new_molecule.data, system.properties)
    return old_molecule, new_molecule, mutation_function_index
