"""Calculate interaction potentials for system of n molecules loaded from data.

Usage example (called from project root):

python src/apps/calculate_between.py -f tests/data/water_zmatrix.com -n 2 -t 10 -b 10,10,10
"""

import os
from pathlib import Path
import time
from molecsim.data.system import update_system
from molecsim.physics.calculations import (
    full_potential,
    molecule_potential,
    potential_breakdown,
)
from molecsim.data.contributors import make_potcall, SystemPotentials
from molecsim.data.properties import SystemProperties
from molecsim.physics.mutate import mutate_rotate, mutate_translate, propose_new
from molecsim.physics.potentials import coulomb, lennard_jones
from molecsim.physics.mc import metropolis
from molecsim.utils.models import tip3p
from molecsim.utils.place import randomized_system
from molecsim.utils.parsers import parse_args, parse_com
from molecsim_tools.plot import plot_system
from molecsim.data.rng import RNG
import pickle

import numpy as np

if __name__ == "__main__":
    args = parse_args()
    model = tip3p()

    # Build prototype -> count mappincalcg
    molecule_list = {
        parse_com(path, model): n
        for path, n in zip(args.files, args.numbers, strict=True)
    }

    rng = RNG(args.seed)

    props = SystemProperties(
        molecules=molecule_list,
        box=np.asarray(args.box, dtype=float),
        temperature=args.temperature,
        periodic=args.periodic,
    )
    coulomb_cont = make_potcall(coulomb, props, model)
    lj_cont = make_potcall(lennard_jones, props, model)

    potentials = SystemPotentials([coulomb_cont, lj_cont])

    sim = randomized_system(props, model, potentials, rng)

    breakdown = potential_breakdown(sim, potentials)
    full = full_potential(sim, potentials)

    print("Successfully calculated potentials:")
    for name, cont in breakdown:
        print(f"\tContributor {name}: {cont}")
    print(f"For a total of {full}!")

    fig, update = plot_system(sim)
    fig.show()
    fig.write_html("plot.html")
    # in case update is desired (only works in notebooks and streamlit apps!):
    fig = update(sim)

    steps = int(1e1)
    outdir = Path("out")
    outdir.mkdir(parents=True, exist_ok=True)

    save_every_seconds = 60
    save_count = 1

    start = time.perf_counter_ns()
    last_save = start

    mutation_functions = [mutate_rotate, mutate_translate]
    function_coefficients = [0.5, 1]
    recent_acceptance = np.ones(len(mutation_functions))
    alpha = 1e-3
    prev = sim.data.positions.copy()

    print(f"Now running system for: {steps} steps")
    for step in range(steps):
        # propose a move
        old_molecule, new_molecule, index = propose_new(
            sim, mutation_functions, function_coefficients, rng
        )

        # compute ΔE
        old_contribution = molecule_potential(sim, old_molecule, potentials)
        new_contribution = molecule_potential(sim, new_molecule, potentials)
        diff = new_contribution.potential - old_contribution.potential

        # accept/reject
        decider = metropolis(sim.properties, sim.model, diff, rng)
        recent_acceptance[index] += alpha * (decider - recent_acceptance[index])

        # apply & THEN update the plot
        if decider:
            update_system(sim, old_contribution, new_contribution)

        # now visualize the *current* state
        if step % 1000 == 0:
            print(f"Step: {step}, recently {recent_acceptance}")

        # checkpointing ...
        now = time.perf_counter_ns()
        if (now - last_save) / 1e9 >= save_every_seconds:
            tag = f"{save_count:05d}"
            fname_tmp = outdir / f"state_{tag}.pickle.tmp"
            fname = outdir / f"state_{tag}.pickle"
            with open(fname_tmp, "wb") as f:
                pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(fname_tmp, fname)
            print(f"[checkpoint] saved {fname} at step {step}")
            save_count += 1
            last_save = now

    end = time.perf_counter_ns()

    print(f"Time per interation: {(end - start) / steps * 1e-6} ms")
