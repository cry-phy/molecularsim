from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
from molecsim.data.rng import RNG
import numpy as np


def metropolis(
    properties: SystemProperties, model: ModelConstants, difference: float, rng: RNG
) -> bool:
    """Get metropolis decision based on system properties and energy difference.

    Carefull: units of properties and difference must match!
    """
    boltzmann = np.exp(-difference / properties.temperature / model.k_b)
    return rng.random() < boltzmann
