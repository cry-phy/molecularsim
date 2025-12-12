"""Headless Monte Carlo runner.

- Loads prototype molecules and counts
- Randomly places the system (deterministic given seed)
- Runs N Metropolis steps (rotate/translate proposals)
- Saves the final `sim` object to a pickle

Usage example (from project root):
python src/apps/run_headless.py \
  -f tests/data/water_zmatrix.com \
  -n 2 \
  -t 10 \
  -b 10,10,10 \
  -s 10000 \
  -o out/state_final.pickle \
  --seed 10
"""

from __future__ import annotations

import pickle
import time

import numpy as np

from molecsim.physics.calculations import molecule_potential
from molecsim.utils.models import tip3p
from molecsim.utils.parsers import parse_args, parse_com
from molecsim.utils.place import randomized_system
from molecsim.data.rng import RNG
from molecsim.data.contributors import make_potcall, SystemPotentials
from molecsim.data.properties import SystemProperties
from molecsim.physics.mutate import mutate_rotate, mutate_translate, propose_new
from molecsim.physics.potentials import coulomb, lennard_jones
from molecsim.physics.mc import metropolis


def main() -> None:
    args = parse_args()

    model = tip3p()

    # Build prototype -> count mapping
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

    sim = randomized_system(props, model, rng)

    # Minimal force field: Coulomb + Lennard-Jones
    forces = SystemPotentials(
        [
            make_potcall(coulomb, props, model),
            make_potcall(lennard_jones, props, model),
        ]
    )

    mutation_functions = [mutate_rotate, mutate_translate]
    function_coefficients = [1.0, 1.0]

    start = time.perf_counter_ns()

    for _ in range(args.steps):
        old_molecule, new_molecule, _ = propose_new(
            sim, mutation_functions, function_coefficients, rng
        )

        old_e = molecule_potential(sim, old_molecule, forces)
        new_e = molecule_potential(sim, new_molecule, forces)
        dE = new_e - old_e

        if metropolis(sim.properties, sim.model, dE, rng):
            old_molecule.copy_from(new_molecule)

    end = time.perf_counter_ns()
    dt_ms = (end - start) / 1e6
    per_step_ms = dt_ms / max(1, args.steps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(
        f"Done: {args.steps} steps in {dt_ms:.1f} ms "
        f"({per_step_ms:.4f} ms/step). Saved: {args.output}"
    )


if __name__ == "__main__":
    main()
