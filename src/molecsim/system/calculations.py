# molecsim/system/calculations.py

from __future__ import annotations
from collections.abc import Iterable

from molecsim.system.data import PotCall, SystemForces
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
    """Evaluate each term and return (name, contribution) pairs.

    - name may be None when not provided.
    - Respects enabled flags unless include_disabled=True.
    """
    data = system.data
    parts: list[tuple[str | None, float]] = []
    for fn, args, kwargs, name, enabled in _iter_enabled(
        forces, include_disabled=include_disabled
    ):
        if include_disabled or enabled:
            parts.append((name, float(fn(data, *args, **kwargs))))
    return parts


def full_potential(
    system: SimSystem,
    forces: Iterable[PotCall] | SystemForces,
    *,
    include_disabled: bool = False,
) -> float:
    """Sum the system potential over all (enabled) terms.

    - By default, disabled terms are skipped. This removes a sharp edge where callers forget `.enabled()`.
    - If you already filter upstream, behavior is identical.
    """
    total = 0.0
    data = system.data
    for fn, args, kwargs, _, enabled in _iter_enabled(
        forces, include_disabled=include_disabled
    ):
        if include_disabled or enabled:
            total += float(fn(data, *args, **kwargs))
    return total
