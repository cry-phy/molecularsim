from __future__ import annotations
from collections.abc import Iterable
from molecsim.system.data import SystemData, SystemProperties
from molecsim.system.molecule import Molecule


class SimSystem:
    """System contains the underlying data Structure and general Stucture information."""

    def __init__(
        self,
        contigous_data: SystemData,
        molecules: Iterable[Molecule],
        properties: SystemProperties,
    ) -> None:
        """Initialize Simulation System."""
        self._contigous_data: SystemData = contigous_data
        self._molecules: Iterable[Molecule] = molecules

        if len(self._contigous_data.positions[0]) != len(properties.box):
            raise ValueError(
                "Dimension of bounding box must match dimension of position."
            )
        self.properties = properties


def randomized_system(
    properties: SystemProperties, grid_function=simple_cubic
) -> SimSystem:
    """Generate a system from properties with randomized initial data."""
    total_molecules = sum()
