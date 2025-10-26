"""Interactive debug viewer.

python src/apps/debug_viewer.py \
  -f tests/data/water_zmatrix.com \
  -n 5 \
  -t 10 \
  -b 10,10,10 \
  --seed 1
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from molecsim.utils.parsers import parse_args, parse_com
from molecsim.utils.rng import RNG
from molecsim.utils.place import randomized_system
from molecsim.utils.plot import add_periodic_debug_traces, plot_system
from molecsim.system.data import SystemProperties


def main() -> None:
    args = parse_args()

    # Optional extras (robust even if not yet added to parse_args)
    seed = getattr(args, "seed", 1)
    add_traces = not getattr(args, "no_debug_traces", False)

    # Build prototype -> count mapping from parsed args
    molecule_list = {
        parse_com(path): n for path, n in zip(args.files, args.numbers, strict=True)
    }

    # Temporary charge hack to mirror earlier behavior (remove when charge IO is implemented)
    first_proto = next(iter(molecule_list))
    first_proto.data.charges = np.array([-2, 1, 1])

    rng = RNG(seed)

    props = SystemProperties(
        molecules=molecule_list,
        box=np.asarray(args.box, dtype=float),
        temperature=args.temperature,
    )

    sim = randomized_system(props, rng)

    fig, _update = plot_system(sim)
    if add_traces:
        fig = add_periodic_debug_traces(fig, sim)

    fig.show()


if __name__ == "__main__":
    main()
