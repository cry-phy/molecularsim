from numpy.random import exponential
from molecsim.data.data import StructureArray, SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.properties import SystemProperties
import numpy as np
import numpy.typing as npt


def _compute_reciprocals(
    box: StructureArray,
    real_cut: float,
    reciprocal_cut: float,
) -> tuple[StructureArray, StructureArray]:
    """Precompute k vectors and weights for Ewald summation."""
    k_xyz = np.ceil(reciprocal_cut * box / 2 / np.pi)
    axes = [np.arange(-k, k + 1) for k in k_xyz]

    x, y, z = np.meshgrid(*axes)
    positions = np.column_stack([x, y, z])
    inside_indices = np.where(np.linalg.norm(positions / box))

    inside_positions = positions[inside_indices]
    ks = 2 * np.pi * inside_positions

    volume = np.prod(box)

    ws = 2 * np.pi / volume * np.exp(-(ks**2) / 4 / real_cut**2) / ks**2
    return ks, ws


class EwaldProperties:
    """Contains information relevant to the calculations using Ewald Summation.

    Must have ks updated on accepted step.
    """

    def __init__(
        self,
        real_cut: float,
        reciprocal_cut: float,
        ks: StructureArray,
        ws: StructureArray,
        s: npt.NDArray[np.complex128],
    ) -> None:
        """Generate an instance for specific System."""
        self._ks = ks
        self._ws = ws
        self._real_cut = real_cut
        self._reciprocal_cut = reciprocal_cut
        self._s = s

    @property
    def ks(self):
        """The ks property."""
        return self._ks

    @property
    def ws(self):
        """The ws property."""
        return self._ws

    @property
    def s(self):
        """The s property."""
        return self._s

    @s.setter
    def s(self, value):
        self._s = value


def ewald(
    data: SystemData,
    molecule_data: SystemData,
    props: SystemProperties,
    model: ModelConstants,
    ewald_props: EwaldProperties,
) -> float:
    """Compute Coulomb contribution with ewald potential."""
    exponents = 1j * np.sum(ewald_props.ks * molecule_data.positions, axis=1)
    s_atoms = molecule_data.charges * np.exp(exponents) * model.coulomb_coeff
    s_molecule = np.sum(s_atoms, axis=1)

    s_rest = ewald_props.s - s_molecule

    mode_parts = s_molecule**2 + 2 * np.real(s_molecule * np.conjugate(s_rest))
    contribution = np.sum(ewald_props.ws * mode_parts)
    return contribution
