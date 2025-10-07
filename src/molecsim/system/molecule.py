from collections.abc import Iterable
from molecsim.system.data import StructureNode, SystemData


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
