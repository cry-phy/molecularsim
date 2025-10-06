from __future__ import annotations
from collections.abc import Iterable
from molecsim.system.data import SystemData
from molecsim.system.molecule import Molecule


class SimSystem:
    """System contains the underlying data Structure and general Stucture information."""

    def __init__(
        self,
        contigous_data: SystemData,
        molecules: Iterable[Molecule],
    ) -> None:
        """Initialize Simulation System."""
        self._contigous_data = contigous_data
        self._molecules = molecules
