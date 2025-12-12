from __future__ import annotations
from molecsim.data.data import FLOAT_DTYPE, SystemData, StructureArray
from molecsim.data.model import ModelConstants
from molecsim.data.molecule import Molecule
from molecsim.data.properties import SystemProperties
from molecsim.data.system import SimSystem
from molecsim.data.contributors import PotCall, SystemPotentials
from molecsim.physics.geometry import distance_squares
from collections.abc import Iterable
import numpy.typing as npt
import numpy as np


def _iter_enabled(
    forces: Iterable[PotCall], *, include_disabled: bool
) -> Iterable[PotCall]:
    """Normalize any iterable of PotCall and optionally filter disabled terms."""
    if isinstance(forces, SystemPotentials):
        return forces if include_disabled else forces.enabled()
    return forces if include_disabled else (pc for pc in forces if pc.enabled)


def potential_breakdown(
    system: SimSystem,
    forces: Iterable[PotCall] | SystemPotentials,
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
    forces: Iterable[PotCall] | SystemPotentials,
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
    forces: Iterable[PotCall] | SystemPotentials,
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
    forces: Iterable[PotCall] | SystemPotentials,
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


def molecule_potential(
    sim: SimSystem, molecule: Molecule, potentials: SystemPotentials
) -> float:
    """Calculate Potential contribution for single molecule."""
    return calculate_potential(sim.data, molecule.data, potentials)


def all_rdfs(
    sim: SimSystem, num_bins: int
) -> tuple[StructureArray, dict[tuple[str, str], StructureArray]]:
    """Generate all RDFs for a system."""
    return radial_distances(
        sim.data, sim.model.types, sim.model.types, sim.properties, sim.model, num_bins
    )


def radial_distances(
    data: SystemData,
    atoms_a: Iterable[str],
    atoms_b: Iterable[str],
    props: SystemProperties,
    model: ModelConstants,
    num_bins: int,
) -> tuple[StructureArray, dict[tuple[str, str], StructureArray]]:
    """Calculate g_AB(r): histogram pairs in r^2, then normalize to RDF."""
    if props.periodic:
        r_max = 0.5 * float(np.min(props.box))
    else:
        r_max = float(np.linalg.norm(props.box))
    edges = np.linspace(0.0, r_max, num_bins + 1)
    bins = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    shell_vol = 4.0 * np.pi * bins**2 * dr
    squared_bins = edges**2

    idas = [model.type_index(a) for a in atoms_a]
    idbs = [model.type_index(b) for b in atoms_b]

    r2 = distance_squares(data, data, props.box, True)

    volume = float(np.prod(props.box))
    rdfs: dict[tuple[str, str], StructureArray] = {}

    for ida, stra in zip(idas, atoms_a, strict=True):
        a_mask = data.atoms == ida
        num_a = int(np.count_nonzero(a_mask))
        if num_a == 0:
            for strb in atoms_b:
                rdfs[(stra, strb)] = np.zeros_like(bins, dtype=FLOAT_DTYPE)
            continue

        for idb, strb in zip(idbs, atoms_b, strict=True):
            b_mask = data.atoms == idb
            num_b = int(np.count_nonzero(b_mask))
            if num_b == 0:
                rdfs[(stra, strb)] = np.zeros_like(bins, dtype=FLOAT_DTYPE)
                continue

            pair_mask = a_mask[:, None] & b_mask[None, :]
            if ida == idb:
                np.fill_diagonal(pair_mask, False)

            d2 = r2[pair_mask]
            keep = (d2 > 0.0) & (d2 < r_max * r_max)
            d2 = d2[keep]

            counts, _ = np.histogram(d2, bins=squared_bins)

            if ida == idb:
                rhob = (num_a - 1) / volume
                norm = num_a * rhob
            else:
                rhob = num_b / volume
                norm = num_a * rhob

            with np.errstate(divide="ignore", invalid="ignore"):
                g = counts.astype(FLOAT_DTYPE) / (norm * shell_vol)
                g[~np.isfinite(g)] = 0.0

            rdfs[(stra, strb)] = g

    return bins, rdfs
