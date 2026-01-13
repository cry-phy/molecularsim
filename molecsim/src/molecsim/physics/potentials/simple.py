from molecsim.data.data import SystemData, FloatPrecision, FLOAT_DTYPE
import numpy.typing as npt
import numpy as np


def harmonic_well(
    data: SystemData,
    molecule_data: SystemData,
    curvature: npt.NDArray[FloatPrecision] = np.full(1, 1),
    center: npt.NDArray[FloatPrecision] = np.full(1, 0),
) -> float:
    """Quadratic well: sum_i ( (r_i - c)^2 * k ) over molecule_data."""
    rel = molecule_data.positions - center.astype(FLOAT_DTYPE, copy=False)
    single = (rel * rel) * curvature.astype(FLOAT_DTYPE, copy=False)
    return float(np.sum(single, dtype=FLOAT_DTYPE))
