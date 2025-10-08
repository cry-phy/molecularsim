import numpy as np
import numpy.typing as npt
from molecsim.system.data import FloatPrecision, FLOAT_DTYPE


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
