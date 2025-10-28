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

    # Boltzmann (kcal/mol/K)
    _k_b = 0.0019872041

    # Coulomb coefficient:
    # 1/(4*pi*epsilon_0) =
    _coulomb_coeff = 332.06371  # kcal*A/mol/e²

    def __init__(
        self,
        molecules: dict[Molecule, int],
        box: StructureArray,
        temperature: float,
        periodic: bool,
        registry: TypeRegistry,
        qtable: ChargeTable,
        ljtable: LJTable,
    ) -> None:
        """Initialize data."""
        self._molecules: dict[Molecule, int] = molecules  # Count
        self._box: StructureArray = box  # Angstrom
        self._temperature: float = temperature  # Kelvin
        self._periodic: bool = periodic
        self._registry: TypeRegistry = registry
        self._qtable: ChargeTable = qtable
        self._ljtable: LJTable = ljtable

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

    @property
    def heat_energy(self):
        """The heat_energy property."""
        return self._temperature * self._k_b

    @property
    def coulomb_coeff(self):
        """The coulomb_coeff property."""
        return self._coulomb_coeff

    @property
    def registry(self):
        """TypeRegistry (read-only handle)."""
        return self._registry

    @property
    def types(self) -> list[str]:
        """Canonical type names in registry order."""
        return list(self._registry.types)

    def type_index(self, name: str) -> int:
        """Type name -> registry index."""
        return self._registry.index_of(name)

    def type_name(self, idx: int) -> str:
        """Registry index -> type name."""
        return self._registry.types[idx]

    def charge_of(self, t: str) -> float:
        """Charge of a single type (by name or index)."""
        return self._qtable.charge_of_type(str(t))

    def lj_for(self, ti: int, tj: int) -> tuple[float, float]:
        """A_ij, B_ij for a single pair."""
        A, B = self._ljtable.table[ti, tj]
        return A, B
