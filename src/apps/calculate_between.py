"""Calculate interaction forces for system of n molecules loaded from data.

Usage example (called from project root):

python src/apps/calculate_between.py -f tests/data/water_zmatrix.com -n 2 -t 10 -b 10,10,10
"""

import time
from molecsim.system.calculations import (
    calculate_potential,
    full_potential,
    potential_breakdown,
)
from molecsim.system.data import make_potcall, SystemForces, SystemProperties
from molecsim.system.mutate import mutate_rotate, mutate_translate, propose_new
from molecsim.system.potentials import coulomb, lennard_jones
from molecsim.utils.mc import metropolis
from molecsim.utils.place import randomized_system
from molecsim.utils.parsers import parse_args, parse_com
from molecsim.utils.rng import RNG

import numpy as np

if __name__ == "__main__":
    args = parse_args()
    molecule_list = {
        parse_com(path): n for path, n in zip(args.files, args.numbers, strict=True)
    }
    # Hack until implementing charge loading from gaussian outputs
    prop_h20 = list(molecule_list.keys())[0]
    prop_h20.data.charges = np.array([-2, 1, 1])

    rng = RNG(10)  # Ensure RNG is seeded
    box = np.asarray(args.box)
    temp = args.temperature

    props = SystemProperties(molecule_list, box, temp)

    sim = randomized_system(props, rng)  # Ensure deterministic placement

    coulomb_cont = make_potcall(coulomb, props)
    lj_cont = make_potcall(lennard_jones, props)

    forces = SystemForces([coulomb_cont, lj_cont])

    breakdown = potential_breakdown(sim, forces)
    full = full_potential(sim, forces)

    print("Successfully calculated potentials:")
    for name, cont in breakdown:
        print(f"\tContributor {name}: {cont}")
    print(f"For a total of {full}!")

    steps = 100

    start = time.perf_counter_ns()
    mutation_functions = [mutate_rotate, mutate_translate]
    print(f"Now running system for: {steps}")
    for step in range(steps):
        print(f"Step: {step}")
        old_molecule, new_molecule = propose_new(sim, mutation_functions, 1, rng)

        # should be wrapped like full_potential into molecule_potential
        old_contribution = calculate_potential(sim.data, old_molecule.data, forces)
        new_contribution = calculate_potential(sim.data, new_molecule.data, forces)

        diff = new_contribution - old_contribution
        decider = metropolis(sim.properties, diff, rng)

        if decider:
            old_molecule.copy_from(new_molecule.data)
            print(f"\tDelta E / kT = {diff}/{props.heat_energy} was accepted.")
        else:
            print(f"\tDelta E / kT = {diff}/{props.heat_energy} was rejected.")

    end = time.perf_counter_ns()

    print(f"Time per interation: {(end - start) / steps * 1e-6} ms")
