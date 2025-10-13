"""All force Calculations should be made using a function following PotContributor Signature."""

from molecsim.system.data import FloatPrecision, SystemData, FLOAT_DTYPE
import numpy.typing as npt
import numpy as np


def harmonic_well(
    data: SystemData,
    curvature: npt.NDArray[FloatPrecision],
    center: npt.NDArray[FloatPrecision],
) -> float:
    """Quadratic well: sum_i ( (r_i - c)^2 * k ) over all atoms."""
    rel = data.positions - center.astype(FLOAT_DTYPE, copy=False)
    single = (rel * rel) * curvature.astype(FLOAT_DTYPE, copy=False)
    return float(np.sum(single, dtype=FLOAT_DTYPE))


def harmonic_well_gpu(
    data: SystemData,
    curvature: npt.NDArray[FloatPrecision],
    center: npt.NDArray[FloatPrecision],
) -> float:
    """GPU: Quadratic well: sum_i ( (r_i - c)^2 * k ) over all atoms."""
    ...
