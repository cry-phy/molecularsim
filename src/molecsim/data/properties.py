from __future__ import annotations
from molecsim.data.chargetable import ChargeTable
from molecsim.data.data import StructureArray

import typing

from molecsim.data.ljtable import LJTable
from molecsim.data.typeregistry import TypeRegistry

if typing.TYPE_CHECKING:
    from molecsim.data.molecule import Molecule


class SystemProperties:
    """Immutable wrapper for calculating and storing system properties.

    Stores NVT Parameters:
        N: Number molecules
        V: Volume through box boundaries
        T: Kelvin
    """

    def __init__(
        self,
        molecules: dict[Molecule, int],
        box: StructureArray,
        temperature: float,
        periodic: bool,
    ) -> None:
        """Initialize data."""
        self._molecules: dict[Molecule, int] = molecules  # Count
        self._box: StructureArray = box  # Angstrom
        self._temperature: float = temperature  # Kelvin
        self._periodic: bool = periodic

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
    def periodic(self):
        """The periodic property."""
        return self._periodic
