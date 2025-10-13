from __future__ import annotations
from collections.abc import Iterable
from typing import Final
import typing
import numpy as np
import numpy.typing as npt

if typing.TYPE_CHECKING:
    from molecsim.system.molecule import Molecule

FLOAT_DTYPE: Final = np.float64

type FloatPrecision = np.float64
type StructureArray = npt.NDArray[FloatPrecision]
type StructureView = StructureArray


class SystemData:
    """Contains Data encoded for simulation system usage."""

    def __init__(self, n: int) -> None:
        """Generate zeroed System Data for *n* Atoms."""
        self._positions: StructureArray | StructureView = np.zeros(
            shape=(n, 3), dtype=FLOAT_DTYPE
        )
        self._charges: StructureArray | StructureView = np.zeros(n, dtype=FLOAT_DTYPE)
        self._atoms: npt.NDArray[np.object_] = np.array([""] * n)
        self._first_empty: int = 0

    @classmethod
    def as_owner(
        cls, atoms: Iterable[str], positions: StructureArray, charges: StructureArray
    ) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        instance = cls(len(charges))
        instance._atoms = np.asarray(atoms, copy=True)
        instance._positions = positions.copy()
        instance._charges = charges.copy()
        return instance

    def as_borrow(
        self,
        start: int,
        size: int,
    ) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        instance = SystemData(size)
        instance._atoms = self._atoms.view()[start : start + size]
        instance._positions = self._positions.view()[start : start + size]
        instance._charges = self._charges.view()[start : start + size]
        return instance

    def allocate_molecule(
        self,
        size: int,
    ) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        return self.as_borrow(self.first_empty, size)

    def copy_from(self, data: SystemData) -> None:
        """Copy data from another system into this location."""
        np.copyto(self._atoms, data._atoms)
        np.copyto(self._positions, data._positions)
        np.copyto(self._charges, data._charges)

    def deepcopy(self) -> SystemData:
        """Generate copy."""
        return SystemData.as_owner(
            self._atoms.copy(),
            self._positions.copy(),
            self._charges.copy(),
        )

    @property
    def positions(self):
        """The positions property."""
        return self._positions

    @positions.setter
    def positions(self, positions: StructureArray | StructureView) -> None:
        np.copyto(self._positions, positions)

    @property
    def first_empty(self):
        """The first_empty property."""
        return self._first_empty

    def __len__(self) -> int:
        """Return number of atoms."""
        return len(self._atoms)


class StructureNode:
    """Node in the Structure Tree."""

    def __init__(
        self, molecular_index: int, neighbors: Iterable[StructureNode]
    ) -> None:
        """Generate empty node."""
        self.molecular_index: int = molecular_index
        self.neighbors: list[StructureNode] = list(neighbors)


class SystemProperties:
    """Immutable wrapper for calculating and storing system properties."""

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
