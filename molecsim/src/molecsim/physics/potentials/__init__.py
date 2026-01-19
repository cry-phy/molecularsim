"""Modlue for encapsulating all potentials.

All potential Calculations should be made using a function following PotContributor Signature.
If potentials need to be cache data it needs to be passed an object caching the data following the PotCache contributor, which is passed to the PotContributor as argument (reference ewald summation for example).
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
