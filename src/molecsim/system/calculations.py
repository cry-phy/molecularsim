# molecsim/system/calculations.py

from __future__ import annotations
from collections.abc import Iterable
from typing import Union

from molecsim.system.data import (
    INT_DTYPE,
    PotCall,
    StructureArray,
    SystemData,
    SystemForces,
    SystemProperties,
)
from molecsim.system.molecule import Molecule
from molecsim.system.potentials import distance_squares
from molecsim.system.system import SimSystem
import numpy.typing as npt
import numpy as np


def _iter_enabled(
    forces: Iterable[PotCall], *, include_disabled: bool
) -> Iterable[PotCall]:
    """Normalize any iterable of PotCall and optionally filter disabled terms."""
    if isinstance(forces, SystemForces):
        return forces if include_disabled else forces.enabled()
    return forces if include_disabled else (pc for pc in forces if pc.enabled)


def potential_breakdown(
    system: SimSystem,
    forces: Iterable[PotCall] | SystemForces,
    *,
    include_disabled: bool = False,
) -> list[tuple[str | None, float]]:
    """Evaluate each term and return (name, contribution) pairs for the whole system."""
    return calculate_breakdown(
        system.data, system.data, forces, include_disabled=include_disabled
    )


def calculate_breakdown(
    large_data: SystemData,
    small_data: SystemData,
    forces: Iterable[PotCall] | SystemForces,
    *,
    include_disabled: bool = False,
) -> list[tuple[str | None, float]]:
    """Evaluate each term and return (name, contribution) pairs.

    - name may be None when not provided.
    - Respects enabled flags unless include_disabled=True.
    """
    parts: list[tuple[str | None, float]] = []
    for fn, args, kwargs, name, enabled in _iter_enabled(
        forces, include_disabled=include_disabled
    ):
        if include_disabled or enabled:
            parts.append((name, float(fn(large_data, small_data, *args, **kwargs))))
    return parts


def full_potential(
    system: SimSystem,
    forces: Iterable[PotCall] | SystemForces,
    *,
    include_disabled: bool = False,
) -> float:
    """Sum the system potential over all (enabled) terms."""
    return calculate_potential(
        system.data, system.data, forces, include_disabled=include_disabled
    )


def calculate_potential(
    large_data: SystemData,
    small_data: SystemData,
    forces: Iterable[PotCall] | SystemForces,
    *,
    include_disabled: bool = False,
) -> float:
    """Sum over all potential contributions from the small data subset.

    - By default, disabled terms are skipped. This removes a sharp edge where callers forget `.enabled()`.
    - If you already filter upstream, behavior is identical.
    """
    total = 0.0
    for fn, args, kwargs, _, enabled in _iter_enabled(
        forces, include_disabled=include_disabled
    ):
        if include_disabled or enabled:
            total += float(fn(large_data, small_data, *args, **kwargs))
    return total


def radial_distances(
    data: SystemData,
    atoms_a: Iterable[str],
    atoms_b: Iterable[str],
    props: SystemProperties,
    num_bins: int,
) -> tuple[StructureArray, dict[tuple[str, str], npt.NDArray[INT_DTYPE]]]:
    """Calculate the distances between all atoms and bin them.

    Since we commonly want the full distributions, we accept all distanfces calculations.
    """
    idas = data.atom_str_to_ids(atoms_a)
    idbs = data.atom_str_to_ids(atoms_b)

    rdfs = {}
    bins = np.linspace(0, np.sum(props.box**2) ** 0.5, num=num_bins, endpoint=True)
    # bin in squared space to avoid square N**2 square roots
    squared_bins = bins**2

    r2 = distance_squares(data, data, props.box, props.periodic)
    for ida, stra in zip(idas, atoms_a, strict=True):
        a_mask = data.atoms == ida
        for idb, strb in zip(idbs, atoms_b, strict=True):
            b_mask = data.atoms == idb
            mask = a_mask[:, None] & b_mask[None, :]
            distances_squared = r2[mask]
            hist, _ = np.histogram(distances_squared, bins=squared_bins)
            rdfs[stra, strb] = hist

    return bins, rdfs
