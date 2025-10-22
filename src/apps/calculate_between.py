"""Calculate interaction forces for system of n molecules loaded from data.

Usage example (called from project root):

python src/apps/calculate_between.py -f tests/data/water_zmatrix.com -n 2 -t 10 -b 10,10,10
"""

from os import makedirs
import os
from pathlib import Path
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
from molecsim.utils.plot import plot_system
from molecsim.utils.rng import RNG
import pickle

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

    coulomb_cont = make_potcall(
        coulomb,
        props,
    )
    lj_cont = make_potcall(lennard_jones, props)

    forces = SystemForces([coulomb_cont, lj_cont])

    breakdown = potential_breakdown(sim, forces)
    full = full_potential(sim, forces)

    print("Successfully calculated potentials:")
    for name, cont in breakdown:
        print(f"\tContributor {name}: {cont}")
    print(f"For a total of {full}!")

    plot_system(sim).show()

    steps = int(1e10)
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    save_every_seconds = 60  # 600  # 10 minutes
    save_count = 1

    start = time.perf_counter_ns()
    last_save = start

    mutation_functions = [mutate_rotate, mutate_translate]
    recent_acceptance = 1
    alpha = 1e-3
    print(f"Now running system for: {steps} steps")
    for step in range(steps):
        if step % 1000 == 0:
            print(f"Step: {step}, recently {recent_acceptance}")

        now = time.perf_counter_ns()
        if (now - last_save) / 1e9 >= save_every_seconds:
            tag = f"{save_count:05d}"  # 00001..10000
            fname_tmp = outdir / f"state_{tag}.pickle.tmp"
            fname = outdir / f"state_{tag}.pickle"
            with open(fname_tmp, "wb") as f:
                pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(fname_tmp, fname)
            print(f"[checkpoint] saved {fname} at step {step}")
            save_count += 1
            last_save = now
        old_molecule, new_molecule = propose_new(sim, mutation_functions, 0.1, rng)

        # should be wrapped like full_potential into molecule_potential
        old_contribution = calculate_potential(sim.data, old_molecule.data, forces)
        new_contribution = calculate_potential(sim.data, new_molecule.data, forces)

        diff = new_contribution - old_contribution
        decider = metropolis(sim.properties, diff, rng)

        recent_acceptance += alpha * (decider - recent_acceptance)

        if decider:
            old_molecule.copy_from(new_molecule.data)

    end = time.perf_counter_ns()

    print(f"Time per interation: {(end - start) / steps * 1e-6} ms")
