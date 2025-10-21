# molecsim/system/calculations.py

from __future__ import annotations
from collections.abc import Iterable

from molecsim.system.data import PotCall, SystemData, SystemForces
from molecsim.system.molecule import Molecule
from molecsim.system.system import SimSystem


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
