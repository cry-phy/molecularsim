from __future__ import annotations
from molecsim.data.data import SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.molecule import Molecule
from molecsim.data.properties import SystemProperties
from collections.abc import Sequence


class SimSystem:
    """System contains the underlying data Structure and general Stucture information."""

    def __init__(
        self,
        contigous_data: SystemData,
        molecules: Sequence[Molecule],
        properties: SystemProperties,
        model: ModelConstants,
    ) -> None:
        """Initialize Simulation System."""
        self._contigous_data: SystemData = contigous_data
        self._molecules: Sequence[Molecule] = molecules

        if len(self._contigous_data.positions[0]) != len(properties.box):
            raise ValueError(
                "Dimension of bounding box must match dimension of position."
            )
        self._properties = properties
        self._model = model

    @property
    def molecules(self):
        """The molecules property."""
        return self._molecules

    @property
    def data(self):
        """The data property."""
        return self._contigous_data

    @property
    def properties(self):
        """The properties property."""
        return self._properties

    @property
    def model(self):
        """The model property."""
        return self._model
