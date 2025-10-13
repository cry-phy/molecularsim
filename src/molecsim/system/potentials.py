"""All force Calculations should be made using a function following PotContributor Signature."""

from molecsim.system.data import FloatPrecision, SystemData
import numpy.typing as npt
import numpy as np


def harmonic_well(
    data: SystemData,
    curvature: npt.NDArray[FloatPrecision],
    center: npt.NDArray[FloatPrecision],
) -> float:
    """All particles in a potential well."""
    rel_postions = data.positions - center
    single_potentials = rel_postions**2 * curvature
    return np.sum(single_potentials)


def harmonic_well_gpu(
    data: SystemData,
    curvature: npt.NDArray[FloatPrecision],
    center: npt.NDArray[FloatPrecision],
) -> float:
    """All Particles in potential well with kernels."""
    ...
