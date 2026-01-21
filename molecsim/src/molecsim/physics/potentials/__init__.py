"""Modlue for encapsulating all potentials.

All potential Calculations should be made using a function following PotContributor Signature.
If potentials need to be cache data it needs to be passed an object caching the data following the PotCache contributor, which is passed to the PotContributor as argument (reference ewald summation for example).
"""

from molecsim.physics.potentials.ewald import ewald
from molecsim.physics.potentials.gpu import coulomb_gpu, lennard_jones_gpu
from molecsim.physics.potentials.simple_potentials import harmonic_well

from molecsim.physics.potentials.classic import coulomb, lennard_jones
from molecsim.physics.potentials.simple_njit import coulomb_njit, lennard_jones_njit
from molecsim.physics.potentials.dense import coulomb_dense, lennard_jones_dense
from molecsim.physics.potentials.stream import coulomb_stream, lennard_jones_stream

__all__ = [
    "coulomb",
    "coulomb_dense",
    "coulomb_gpu",
    "coulomb_njit",
    "coulomb_stream",
    "ewald",
    "harmonic_well",
    "lennard_jones",
    "lennard_jones_dense",
    "lennard_jones_gpu",
    "lennard_jones_njit",
    "lennard_jones_stream",
]
