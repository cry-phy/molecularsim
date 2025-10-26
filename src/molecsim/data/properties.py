from __future__ import annotations
from molecsim.data.data import StructureArray

import typing
if typing.TYPE_CHECKING:
    from molecsim.system.molecule import Molecule

class SystemProperties:
    """Immutable wrapper for calculating and storing system properties.

    Stores NVT Parameters:
        N: Number molecules (Dict of types)
        V: Volume through box boundaries (Angstrom)
        T: Kelvin
    """

    _k_b = 1  # by definition
    # Coulomb coefficient:
    # 1/(4*pi*epsilon_0) = 8.98e9 N*m²/C²
    _coulomb_coeff = 1.671009e5  # K*A/e²

    # LJ parameters (TIP3P oxygen):
    # epsilon = 0.1521 kcal/mol
    # 0.1521*503.2 = 76.514 K
    # sigma = 3.15 A
    # A = 4*epsilon*sigma¹², B = 4*epsilon*sigma⁶
    _LJ_A = 2.928737e8  # K*A¹²
    _LJ_B = 2.994156e5  # K*A⁶

    def __init__(
        self,
        molecules: dict[Molecule, int],
        box: StructureArray,
        temperature: float,
        periodic: bool = True,
    ) -> None:
        """Initialize data."""
        self._molecules: dict[Molecule, int] = molecules  # Count
        self._box: StructureArray = box  # Angstrom
        self._temperature: float = temperature  # Kelvin
        self.periodic: bool = periodic

    @property
    def box(self):
        """The box property."""
        return self._box

    @property
    def molecules(self) -> dict[Molecule, int]:
        """The molecules property."""
        return self._molecules

    @property
    def temperature(self):
        """The temperature property."""
        return self._temperature

    @property
    def heat_energy(self):
        """The heat_energy property."""
        return self._temperature * self._k_b

    @property
    def coulomb_coeff(self):
        """The coulomb_coeff property."""
        return self._coulomb_coeff

    @property
    def lj_params(self):
        """The lj_params property."""
        return self._LJ_A, self._LJ_B
