from __future__ import annotations
from collections.abc import Iterable
from molecsim.system.data import StructureArray, SystemData
from molecsim.system.molecule import Molecule


class SimSystem:
    """System contains the underlying data Structure and general Stucture information."""

    def __init__(
        self,
        contigous_data: SystemData,
        molecules: Iterable[Molecule],
        box_size: StructureArray,
    ) -> None:
        """Initialize Simulation System."""
        self._contigous_data: SystemData = contigous_data
        self._molecules: Iterable[Molecule] = molecules

        if len(self._contigous_data.positions[0]) != len(box_size):
            raise ValueError(
                "Dimension of bounding box must match dimension of position."
            )
        self._box_size: StructureArray = box_size
