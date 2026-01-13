"""Modlue for encapsulating all potentials.

All force Calculations should be made using a function following PotContributor Signature.
"""

from molecsim.physics.potentials.ewald import ewald
from molecsim.physics.potentials.simple import harmonic_well
from molecsim.physics.potentials.lj import lennard_jones
from molecsim.physics.potentials.coulomb import coulomb

__all__ = [
    "coulomb",
    "ewald",
    "harmonic_well",
    "lennard_jones",
]
