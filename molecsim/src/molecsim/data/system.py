from __future__ import annotations
from molecsim.data.data import SystemData
from molecsim.data.model import ModelConstants
from molecsim.data.molecule import Molecule
from molecsim.data.properties import SystemProperties
from collections.abc import Sequence
from typing import Any
from molecsim.data.contributors import Contribution, SystemPotentials


class SimSystem:
    """System contains the underlying data Structure and general Stucture information."""

    def __init__(
        self,
        contigous_data: SystemData,
        molecules: Sequence[Molecule],
        properties: SystemProperties,
        model: ModelConstants,
        updater: SystemUpdater,
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
        self._updaters = updater

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

    @property
    def updaters(self):
        """The updaters property."""
        return self._updaters


class SystemUpdater:
    """Access point for the simulation to bind to specific updateable objects which might be shared between different forces calculations."""

    def __init__(self, potentials: SystemPotentials) -> None:
        """Initialize the lookup for the updaters."""
        self._updaters = potentials.get_updaters()

    def update(self, sim: SimSystem, old: Contribution, new: Contribution) -> None:
        """Apply all accepted-move updates using cached payloads in contributions."""
        for key, obj in self._updaters:
            obj.__update__(sim, old, new, key=key)

    def key_for(self, obj: Any, potentials: SystemPotentials) -> str:
        """Get key for object for tracing cached data."""
        return potentials.key_for(obj)


def update_system(sim: SimSystem, old: Contribution, new: Contribution):
    """Update simulation."""
    sim.updaters.update(sim, old, new)
    old.molecule.copy_from(new.molecule)
