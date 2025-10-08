import numpy as np
import numpy.typing as npt

from molecsim.utils.geometry import rot_matrix_quaternions


class RNG:
    """Simulation RNG wrapper with reproducible seed and helpers."""

    def __init__(self, seed: int | None = None):
        """Initialize random number generator with seed for reproduceability."""
        self._rng = np.random.default_rng(seed)

    # --- convenience samplers ---
    def random_direction(self) -> npt.NDArray:
        """Uniform direction on unit sphere."""
        vec = self._rng.normal(0.0, 1.0, 3)
        return vec / np.linalg.norm(vec)

    def rotation_matrix(self, magnitude: float) -> npt.NDArray:
        """Random rotation matrix uniformly distributed over SO(3)."""
        u1, u2, u3 = self._rng.random(3) * magnitude
        return rot_matrix_quaternions(u1, u2, u3)
