from __future__ import annotations
import typing
import numpy as np
import numpy.typing as npt
from molecsim.system.data import FloatPrecision, FLOAT_DTYPE, StructureArray

if typing.TYPE_CHECKING:
    from molecsim.utils.rng import RNG


def rot_matrix(bend_angle: float, dihedral_angle: float) -> npt.NDArray[FloatPrecision]:
    """Generate a 3D rotation matrix from bend and dihedral angles (radians)."""
    cos_bend = np.cos(bend_angle)
    sin_bend = np.sin(bend_angle)
    cos_dihedral = np.cos(dihedral_angle)
    sin_dihedral = np.sin(dihedral_angle)

    return np.array(
        [
            [cos_dihedral * cos_bend, -sin_dihedral, cos_dihedral * sin_bend],
            [sin_dihedral * cos_bend, cos_dihedral, sin_dihedral * sin_bend],
            [-sin_bend, 0.0, cos_bend],
        ],
        dtype=FLOAT_DTYPE,
    )


def rot_matrix_quaternions(
    u1: float, u2: float, u3: float
) -> npt.NDArray[FloatPrecision]:
    """Rotation matrix using quaternions."""
    # No real clue whats going on here but seems to work (ChatGPT)
    # Using method from "Graphics Gems" (Ken Shoemake)
    q = np.array(
        [
            np.sqrt(1 - u1) * np.sin(2 * np.pi * u2),
            np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
            np.sqrt(u1) * np.sin(2 * np.pi * u3),
            np.sqrt(u1) * np.cos(2 * np.pi * u3),
        ]
    )
    # Convert quaternion to rotation matrix
    w, x, y, z = q[3], q[0], q[1], q[2]
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return matrix


def wrap_periodic(x, a, b):
    """Wrap N-D positions x into the periodic interval [a, b)."""
    x = np.asarray(x)
    a = np.asarray(a)
    b = np.asarray(b)
    return a + np.mod(x - a, b - a)


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


def fuzzyfy(
    positions: StructureArray,
    box: StructureArray,
    rng: RNG,
    strength: float = 0.2,
) -> StructureArray:
    """Fuzzy move the positions based on density."""
    position_volume = np.prod(box) / len(positions)
    position_radius = position_volume ** (1 / 3)
    fuzz = rng.random_like(positions, limit=float(strength * position_radius))
    proposal = fuzz + positions
    return wrap_periodic(proposal, 0, box)
