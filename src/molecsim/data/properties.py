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

    # Boltzmann (kcal/mol/K)
    _k_b = 0.0019872041

    # Coulomb coefficient:
    # 1/(4*pi*epsilon_0) =
    _coulomb_coeff = 332.06371  # kcal*A/mol/e²

    # LJ parameters (TIP3P oxygen):
    _LJ_A = 5.8200266e5  # kcal*A¹²/mol
    _LJ_B = 5.9505497e2  # kcal*A⁶/mol

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
