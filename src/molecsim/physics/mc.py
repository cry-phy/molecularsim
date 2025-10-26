from molecsim.data.properties import SystemProperties
from molecsim.utils.rng import RNG
import numpy as np


def metropolis(properties: SystemProperties, difference: float, rng: RNG) -> bool:
    """Get metropolis decision based on system properties and energy difference.

    Carefull: units of properties and difference must match!
    """
    boltzmann = np.exp(-difference / properties.heat_energy)
    return rng.random() < boltzmann
