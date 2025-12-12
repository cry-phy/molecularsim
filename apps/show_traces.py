"""Interactive debug viewer.

python src/apps/debug_viewer.py \
  -f tests/data/water_zmatrix.com \
  -n 5 \
  -t 10 \
  -b 10,10,10 \
  --seed 1
"""

from __future__ import annotations

import numpy as np

from molecsim.utils.models import tip3p
from molecsim.utils.parsers import parse_args, parse_com
from molecsim.data.rng import RNG
from molecsim.utils.place import randomized_system
from molecsim_tools.plot import add_periodic_debug_traces, plot_system
from molecsim.data.properties import SystemProperties


def main() -> None:
    args = parse_args()

    model = tip3p()

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

    fig, _ = plot_system(sim)
    fig = add_periodic_debug_traces(fig, sim)

    fig.show()


if __name__ == "__main__":
    main()
