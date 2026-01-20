"""Modlue for encapsulating all potentials.

All potential Calculations should be made using a function following PotContributor Signature.
If potentials need to be cache data it needs to be passed an object caching the data following the PotCache contributor, which is passed to the PotContributor as argument (reference ewald summation for example).
"""

from molecsim.physics.potentials.ewald import ewald
from molecsim.physics.potentials.gpu import coulomb_gpu, lennard_jones_gpu
from molecsim.physics.potentials.simple import harmonic_well

# from molecsim.physics.potentials.stream import coulomb_stream, lennard_jones_stream
from molecsim.physics.potentials.lj import (
    lennard_jones,
    lennard_jones_dense,
    lennard_jones_njit,
)
from molecsim.physics.potentials.coulomb_pot import (
    coulomb,
    coulomb_dense,
    coulomb_njit,
)

__all__ = [
    "coulomb",
    "coulomb_dense",
    "coulomb_gpu",
    "coulomb_njit",
    # "coulomb_stream",
    "ewald",
    "harmonic_well",
    "lennard_jones",
    "lennard_jones_dense",
    "lennard_jones_gpu",
    "lennard_jones_njit",
    # "lennard_jones_stream",
]
