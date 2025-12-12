from __future__ import annotations
from molecsim.data.data import StructureArray, StructureView, SystemData
from collections.abc import Iterable
from copy import deepcopy


class StructureNode:
    """Node in the Structure Tree."""

    def __init__(
        self, molecular_index: int, neighbors: Iterable[StructureNode]
    ) -> None:
        """Generate empty node."""
        self.molecular_index: int = molecular_index
        self.neighbors: list[StructureNode] = list(neighbors)

    def __repr__(self) -> str:
        """Give simple representation."""
        return f"""<StructureNode: {self.molecular_index}, {[neigh.molecular_index for neigh in self.neighbors]}>"""


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

    def copy_from(self, molecule: Molecule) -> None:
        """Set underlying data with copies of this data."""
        self._data_slices.copy_from(molecule.data)

    @property
    def center(self):
        """The center property."""
        return self._center

    @property
    def data(self):
        """The data property."""
        return self._data_slices

    def copy_data(self) -> Molecule:
        """Generate deepcopy of molecule.

        Data shares atom conversions.
        """
        instance = Molecule(self.data.copy_data(), deepcopy(self._structure_nodes))
        instance._center = self.center.copy()
        return instance

    def move_to(self, data: SystemData) -> None:
        """Move Molecule to SytemData as borrower."""
        data.copy_from(self._data_slices)
        self._data_slices = data

    def size(self) -> int:
        """Return number of atoms."""
        return len(self._data_slices)

    @property
    def structure_nodes(self):
        """The structure_nodes property."""
        return self._structure_nodes
