from __future__ import annotations

import typing
from molecsim.data.chargetable import ChargeTable
from molecsim.data.data import StructureArray
from molecsim.data.ljtable import LJTable
from molecsim.data.typeregistry import TypeRegistry

if typing.TYPE_CHECKING:
    from molecsim.data.molecule import Molecule


class ModelConstants:
    """Force-field & typing (immutable API).

    Holds:
      - TypeRegistry (name <-> index)
      - ChargeTable (per-type charges)
      - LJTable (mixed A_ij, B_ij)
      - Cutoff parameters (Cut distance, Tolerance)
      - Physical constants (k_B, Coulomb coeff)
    """

    def __init__(
        self,
        registry: TypeRegistry,
        qtable: ChargeTable,
        ljtable: LJTable,
        tolerance: float = 1e-7,
        cutoff: float = 10,
        k_b: float = 0.0019872041,
        coulomb_coeff: float = 332.06371,
    ) -> None:
        self._registry = registry
        self._qtable = qtable
        self._ljtable = ljtable
        self._tolerance = tolerance
        self._cutoff = cutoff
        self._k_b = k_b
        self._coulomb_coeff = coulomb_coeff

    @property
    def registry(self) -> TypeRegistry:
        """Atom Type conversion table property."""
        return self._registry

    @property
    def qtable(self) -> ChargeTable:
        """Charge Table property."""
        return self._qtable

    @property
    def ljtable(self) -> LJTable:
        """Lennard Jones coefficient table property."""
        return self._ljtable

    @property
    def tolerance(self):
        """The tolerance property."""
        return self._tolerance

    @property
    def cutoff(self):
        """The cutoff property."""
        return self._cutoff

    @property
    def k_b(self) -> float:
        """Boltzmann constant in kcal/mol/K."""
        return self._k_b

    @property
    def coulomb_coeff(self) -> float:
        """Coulomb coefficient in kcal·Å/(mol·e²)."""
        return self._coulomb_coeff

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
        """Charge of a single type (by name)."""
        return self._qtable.charge_of_type(str(t))

    def lj_for(self, ti: int, tj: int) -> tuple[float, float]:
        """A_ij, B_ij for a single pair (or eps/sig per your LJTable)."""
        A, B = self._ljtable.table[ti, tj]
        return A, B
