from __future__ import annotations
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

STRUCTURE_DTYPE = np.float64
type StructureArray = npt.NDArray[
    STRUCTURE_DTYPE
]  # Centralize the type of Structure dtypes
type StructureView = StructureArray  # Views have the same dtype as Structure arrays


class SystemData:
    """Contains Data encoded for simulation system usage."""

    def __init__(self, n: int) -> None:
        """Generate zeroed System Data for *n* Atoms."""
        self._positions: StructureArray = np.zeros(shape=(n, 3), dtype=STRUCTURE_DTYPE)
        self._charges: StructureArray = np.zeros(n, dtype=STRUCTURE_DTYPE)
        self._atom_types: npt.NDArray[np.int32] = np.arange(n)


class AtomNode:
    """Node in the Structure Tree."""

    def __init__(self, molecular_index: int, neighbors: Iterable[AtomNode]) -> None:
        """Generate empty node."""
        self.molecular_index: int = molecular_index
        self.neighbors: list[AtomNode] = list(neighbors)
