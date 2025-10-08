from __future__ import annotations
from collections.abc import Iterable
from typing import Final
import numpy as np
import numpy.typing as npt

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

    @classmethod
    def as_borrow(
        cls,
        atoms: npt.NDArray[np.object_],
        positions: StructureArray | StructureView,
        charges: StructureArray | StructureView,
        start: int = 0,
        end: int = -1,
    ) -> SystemData:
        """Generate Data as a copy where ownership is with this instance."""
        instance = cls(len(charges))
        instance._atoms = atoms.view()[start:end]
        instance._positions = positions.view()[start:end]
        instance._charges = charges.view()[start:end]
        return instance

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
    """Wrapper for calculating and storing system properties."""

    def __init__(
        self,
    ) -> None:
        """Initialize data"""
