from __future__ import annotations
from collections.abc import Iterable
from molecsim.system.data import (
    StructureArray,
    StructureNode,
    StructureView,
    SystemData,
)
from copy import deepcopy


class Molecule:
    """Holds structural information of single Molecule.

    Typically holds views to a SystemData object and its underlying Arrays. Beware of stale Data!
    """

    def __init__(
        self,
        data: SystemData,
        atom_nodes: Iterable[StructureNode] | None = None,
    ) -> None:
        """Initialize Molecule from list of connected atom nodes and corresponding views."""
        self._data_slices: SystemData = data
        self._structure_nodes: Iterable[StructureNode] | None = atom_nodes

        # default to first atom for now, later COM?
        self._center: StructureArray | StructureView = data.positions[0]

    def set(self, data: SystemData) -> None:
        """Set underlying data with copies of this data."""
        self._data_slices.copy_from(data)

    @property
    def center(self):
        """The center property."""
        return self._center

    @property
    def data(self):
        """The data property."""
        return self._data_slices

    def deepcopy(self) -> Molecule:
        """Generate deepcopy."""
        instance = Molecule(self.data.deepcopy(), deepcopy(self._structure_nodes))
        instance._center = self.center.copy()
        return instance

    def move_to(self, data: SystemData) -> None:
        """Move Molecule to SytemData as borrower."""
        data.copy_from(self._data_slices)
        self._data_slices = data

    def size(self) -> int:
        """Return number of atoms."""
        return len(self._data_slices)
