from collections.abc import Iterable
from system.data import AtomNode, StructureView, StructureArray
import numpy.typing as npt


class Molecule:
    """Holds structural information of single Molecule.

    Holds views to a SystemData object and its underlying Arrays. Beware of stale Data!
    """

    def __init__(
        self,
        atom_nodes: Iterable[AtomNode],
        view: StructureView | StructureArray,
    ) -> None:
        """Initialize Molecule from list of connected atom nodes and corresponding views."""
        self._view: StructureView | StructureArray = view
        self._atom_nodes: Iterable[AtomNode] = atom_nodes
