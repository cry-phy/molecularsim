"""Calculate interaction forces for system of n molecules loaded from data.

Usage example (called from project root):

python src/apps/calculate_between.py -f tests/data/water_zmatrix.com -n 2 -t 10 -b 10,10,10
"""

from molecsim.system.calculations import full_potential, potential_breakdown
from molecsim.system.data import make_potcall, SystemForces, SystemProperties
from molecsim.system.potentials import coulomb, lennard_jones
from molecsim.system.system import randomized_system
from molecsim.utils.parsers import parse_com
from molecsim.utils.rng import RNG

import numpy as np
import argparse
from pathlib import Path


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for file list and read counts."""
    parser = argparse.ArgumentParser(
        prog="molecsim-app",
        description="Specify molecules to load and number of reads for each.",
    )
    parser.add_argument(
        "-f",
        "--files",
        required=True,
        help="Comma-separated list of molecule files to load (e.g. data1.xyz,data2.xyz)",
    )
    parser.add_argument(
        "-n",
        "--numbers",
        required=True,
        help="Comma-separated list of molecule counts corresponding to each file (e.g. 3,5)",
    )
    parser.add_argument(
        "-t",
        "--temperature",
        required=True,
        help="System Temperature in Kelvin",
        type=float,
    )
    parser.add_argument(
        "-b",
        "--box",
        required=True,
        help="Box size in three dimension in Angstrom (e.g. 10,10,10)",
    )
    args = parser.parse_args(argv)
    args.files = [Path(f.strip()) for f in args.files.split(",") if f.strip()]
    args.numbers = [int(n.strip()) for n in args.numbers.split(",") if n.strip()]
    args.box = [int(n.strip()) for n in args.box.split(",") if n.strip()]
    if len(args.files) != len(args.numbers):
        parser.error(
            f"number of files ({len(args.files)}) must match number of counts ({len(args.numbers)})"
        )
    missing = [f for f in args.files if not f.is_file()]
    if missing:
        parser.error(f"file(s) not found: {', '.join(str(f) for f in missing)}")

    return args


if __name__ == "__main__":
    args = parse_args()
    molecule_list = {
        parse_com(path): n for path, n in zip(args.files, args.numbers, strict=True)
    }
    # Hack until implementing charge loading from gaussian outputs
    prop_h20 = list(molecule_list.keys())[0]
    prop_h20.data.charges = np.array([-2, 1, 1])

    rng = RNG(10)
    box = np.asarray(args.box)
    temp = args.temperature

    props = SystemProperties(molecule_list, box, temp)

    sim = randomized_system(props, rng)

    coulomb_cont = make_potcall(coulomb, sim.data)
    lj_cont = make_potcall(lennard_jones, sim.data)

    forces = SystemForces([coulomb_cont, lj_cont])

    breakdown = potential_breakdown(sim, forces)
    full = full_potential(sim, forces)

    print("Successfully calculated potentials:")
    for name, cont in breakdown:
        print(f"\tContributor {name}: {cont}")
    print(f"For a total of {full}!")
