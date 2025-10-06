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
        instance._atoms = np.array(atoms)
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


class StructureNode:
    """Node in the Structure Tree."""

    def __init__(
        self, molecular_index: int, neighbors: Iterable[StructureNode]
    ) -> None:
        """Generate empty node."""
        self.molecular_index: int = molecular_index
        self.neighbors: list[StructureNode] = list(neighbors)
