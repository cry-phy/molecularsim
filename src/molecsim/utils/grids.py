import numpy as np
from molecsim.data.data import StructureArray
from molecsim.utils.rng import RNG


def simple_cubic(
    number_placements: int,
    box: StructureArray,
    rng: RNG,
    periodic: bool = False,
) -> StructureArray:
    """Generate simple cubic arrangement for a number of atoms inside a box.

    Will produce vacancies if no perfect fit can be achieved!
    """
    cell_volume = np.prod(box) / number_placements
    cell_width = cell_volume ** (1 / 3)
    wall_spacing_xyz = (box % cell_width) / 2
    num_cells_xyz = np.ceil(box / cell_width).astype(int)
    axes_xyz = [
        np.linspace(
            start=wall_spacing,
            stop=length - wall_spacing,
            num=num_cells,
            endpoint=not periodic,
        )
        for num_cells, length, wall_spacing in zip(
            num_cells_xyz, box, wall_spacing_xyz, strict=True
        )
    ]
    x, y, z = np.meshgrid(*axes_xyz)
    nodes = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    pick = rng.choose_indices(len(nodes), size=number_placements, replace=False)
    return nodes[pick]
