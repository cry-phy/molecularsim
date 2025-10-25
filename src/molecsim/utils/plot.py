from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from molecsim.system.data import FLOAT_DTYPE, FloatPrecision, StructureArray, SystemData
from molecsim.system.molecule import Molecule

import os
import sys
import plotly.io as pio

from molecsim.system.system import SimSystem

_PLOTLY_INIT_DONE = False


def _in_ipython() -> bool:
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython() is not None
    except Exception:
        return False


def _in_notebook_like() -> bool:
    if not _in_ipython():
        return False
    try:
        from IPython import get_ipython  # type: ignore

        return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _in_colab() -> bool:
    return "google.colab" in sys.modules


def _pick_renderer() -> str:
    if _in_colab():
        return "colab"
    if _in_notebook_like():
        return "notebook_connected"
    return "browser"


def _ensure_renderer_once() -> None:
    global _PLOTLY_INIT_DONE
    if _PLOTLY_INIT_DONE:
        return
    env_choice = os.getenv("PLOTLY_RENDERER")
    if env_choice:
        pio.renderers.default = env_choice
    elif not pio.renderers.default:
        pio.renderers.default = _pick_renderer()
    _PLOTLY_INIT_DONE = True


def _figure_class() -> type[go.Figure] | type[go.FigureWidget]:
    return go.FigureWidget if _in_notebook_like() else go.Figure


# Use common color scheme (RasMol)
# http://www.pdg.cnb.uam.es/cursos/BioInfo2002/pages/Farmac/CMU_course/Periodic/ElemList.htm
_ELEMENT_COLOR_RASMOL: dict[str, str] = {
    "DEFAULT": "#FF1493",
    "H": "#FFFFFF",
    "He": "#FFC8C8",
    "Li": "#A52A2A",
    "Be": "#FF1493",
    "B": "#00FF00",
    "C": "#C8C8C8",
    "N": "#8F8FFF",
    "O": "#F00000",
    "F": "#C8A518",
    "Ne": "#FF1493",
    "Na": "#0000FF",
    "Mg": "#2A802A",
    "Al": "#808090",
    "Si": "#C8A518",
    "P": "#FFA500",
    "S": "#FFC832",
    "Cl": "#00FF00",
    "Ca": "#808090",
    "Ti": "#808090",
    "Cr": "#808090",
    "Mn": "#808090",
    "Fe": "#FFA500",
    "Ni": "#A52A2A",
    "Cu": "#A52A2A",
    "Zn": "#A52A2A",
    "Br": "#A52A2A",
    "Ag": "#808090",
    "I": "#8F8FFF",
    "Ba": "#FFA500",
    "Au": "#C8A518",
}


def color_for(elem: str | Iterable[str]) -> str | list[str]:
    """Return RasMol-style hex color(s) for element symbol(s), fallback to DEFAULT."""

    def _norm(s: str) -> str:
        return (s[:1].upper() + s[1:].lower()) if s else ""

    if isinstance(elem, str):
        e = _norm(elem)
        return _ELEMENT_COLOR_RASMOL.get(e, _ELEMENT_COLOR_RASMOL["DEFAULT"])

    out: list[str] = []
    for single in elem:
        s = str(single) if not isinstance(single, str) else single
        e = _norm(s)
        out.append(_ELEMENT_COLOR_RASMOL.get(e, _ELEMENT_COLOR_RASMOL["DEFAULT"]))
    return out


def plot_system(sim: SimSystem) -> tuple[go.Figure, Callable[[SimSystem], go.Figure]]:
    """Return (figure, update_fn). Keeps view state persistent across updates."""
    _ensure_renderer_once()
    FigureT = _figure_class()

    # build initial traces once
    sticks = lines_to_sticks(extract_bonds(sim.molecules, sim.properties.box))
    atoms = positions_to_atoms(sim.data.positions, sim.data.atoms_as_str())
    box = make_box_trace_from_lengths(sim.properties.box)

    fig = FigureT(data=[sticks, atoms, box])

    dark, invis = "#0b0f19", {"visible": False}
    fig.update_layout(
        title=getattr(sim, "name", None) or "System",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        paper_bgcolor=dark,
        plot_bgcolor=dark,
        font=dict(color="#ECECEC"),
        scene=dict(
            xaxis=invis, yaxis=invis, zaxis=invis, aspectmode="data", bgcolor=dark
        ),
        uirevision="molmc",
    )
    fig.update_scenes(camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)))

    STICKS_I, ATOMS_I, BOX_I = 0, 1, 2

    def update(new_sim):
        # recompute geometry
        new_sticks = extract_bonds(new_sim.molecules, new_sim.properties.box)
        new_atoms_pos = np.asarray(new_sim.data.positions)
        new_box_trace = make_box_trace_from_lengths(new_sim.properties.box)

        # convert to flat coords for sticks
        if new_sticks.size == 0:
            xs, ys, zs = [], [], []
        else:
            n = new_sticks.shape[0]
            xs = np.empty(3 * n, dtype=new_sticks.dtype)
            ys = np.empty(3 * n, dtype=new_sticks.dtype)
            zs = np.empty(3 * n, dtype=new_sticks.dtype)
            xs[0::3] = new_sticks[:, 0, 0]
            xs[1::3] = new_sticks[:, 1, 0]
            xs[2::3] = np.nan
            ys[0::3] = new_sticks[:, 0, 1]
            ys[1::3] = new_sticks[:, 1, 1]
            ys[2::3] = np.nan
            zs[0::3] = new_sticks[:, 0, 2]
            zs[1::3] = new_sticks[:, 1, 2]
            zs[2::3] = np.nan

        # mutate traces IN PLACE (do not replace fig.data tuple)
        if hasattr(fig, "batch_update"):
            ctx = fig.batch_update()
        else:
            # dummy context manager
            from contextlib import nullcontext

            ctx = nullcontext()

        with ctx:
            # sticks
            fig.data[STICKS_I].x = xs
            fig.data[STICKS_I].y = ys
            fig.data[STICKS_I].z = zs

            # atoms
            fig.data[ATOMS_I].x = new_atoms_pos[:, 0]
            fig.data[ATOMS_I].y = new_atoms_pos[:, 1]
            fig.data[ATOMS_I].z = new_atoms_pos[:, 2]
            # (no marker/style changes -> less chance to reset view)

            # box
            fig.data[BOX_I].x = new_box_trace.x
            fig.data[BOX_I].y = new_box_trace.y
            fig.data[BOX_I].z = new_box_trace.z

            # title text only (doesn't affect camera with uirevision set)
            fig.layout.title = getattr(new_sim, "name", None) or "System"

        return fig

    return fig, update


def add_periodic_debug_traces(
    fig: go.Figure,
    sim,
    *,
    add_ghost_atoms: bool = True,
    add_ghost_boxes: bool = True,
    add_all_connections: bool = True,
) -> go.Figure:
    """Add periodic 'ghost' atoms in the 26 neighbor cells.

    Adds ghost box wireframes, and ALL pairwise MIC connections (i != j) as extra traces.
    Does not modify existing traces.
    """
    pos = np.asarray(sim.data.positions, float)  # (N,3)
    boxL = np.asarray(sim.properties.box, float)
    N = pos.shape[0]

    if add_ghost_atoms and N:
        shifts = [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)
        ]
        ghost_chunks = [pos + np.array(s, float) * boxL for s in shifts]
        ghost_pos = (
            np.concatenate(ghost_chunks, axis=0) if ghost_chunks else np.empty((0, 3))
        )
        fig.add_trace(
            go.Scatter3d(
                x=ghost_pos[:, 0],
                y=ghost_pos[:, 1],
                z=ghost_pos[:, 2],
                mode="markers",
                marker=dict(size=3),
                opacity=0.18,
                hoverinfo="skip",
                name="ghosts",
                showlegend=False,
            )
        )

    if add_ghost_boxes:

        def _box_wire_xyz(L):
            Lx, Ly, Lz = map(float, L)
            C = np.array(
                [
                    [0, 0, 0],
                    [Lx, 0, 0],
                    [0, Ly, 0],
                    [Lx, Ly, 0],
                    [0, 0, Lz],
                    [Lx, 0, Lz],
                    [0, Ly, Lz],
                    [Lx, Ly, Lz],
                ]
            )
            E = [
                (0, 1),
                (0, 2),
                (1, 3),
                (2, 3),
                (4, 5),
                (4, 6),
                (5, 7),
                (6, 7),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            xs, ys, zs = [], [], []
            for i, j in E:
                xs += [C[i, 0], C[j, 0], None]
                ys += [C[i, 1], C[j, 1], None]
                zs += [C[i, 2], C[j, 2], None]
            return xs, ys, zs

        bx, by, bz = _box_wire_xyz(boxL)
        gxs, gys, gzs = [], [], []
        for s in [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)
        ]:
            shift = np.array(s, float) * boxL
            for x, y, z in zip(bx, by, bz):
                if x is None:
                    gxs.append(None)
                    gys.append(None)
                    gzs.append(None)
                else:
                    gxs.append(x + shift[0])
                    gys.append(y + shift[1])
                    gzs.append(z + shift[2])

        fig.add_trace(
            go.Scatter3d(
                x=gxs,
                y=gys,
                z=gzs,
                mode="lines",
                line=dict(width=1),
                opacity=0.12,
                hoverinfo="skip",
                name="ghost-boxes",
                showlegend=False,
            )
        )

    if add_all_connections and N:
        disp = pos[:, None, :] - pos[None, :, :]
        if sim.properties.periodic:
            disp -= boxL.reshape(1, 1, 3) * np.round(disp / boxL.reshape(1, 1, 3))

        xs, ys, zs = [], [], []
        for i in range(N):
            a = pos[i]
            for j in range(N):
                if i == j:
                    continue
                b = a - disp[i, j]
                xs += [a[0], b[0], None]
                ys += [a[1], b[1], None]
                zs += [a[2], b[2], None]

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(width=2, color="#88ff88"),
                name="all-connections",
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig


def positions_to_atoms(
    positions: StructureArray,
    atoms: npt.NDArray[np.object_],
) -> go.Scatter3d:
    """Generate atom markers from positions and element symbols."""
    pos = np.asarray(positions)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must be (N,3), got {pos.shape}")

    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    colors = color_for(np.asarray(atoms, dtype=object).ravel())

    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=8, color=colors, line=dict(width=1, color="black")),
        name="atoms",
        showlegend=False,
    )


def lines_to_sticks(
    lines: StructureArray,
    width: int = 5,
) -> go.Scatter3d:
    """Convert endpoints array of shape (n, 2, 3) to a single Plotly Scatter3d lines trace."""
    if lines.size == 0:
        return go.Scatter3d(
            mode="lines",
            x=[],
            y=[],
            z=[],
            line=dict(width=width, color="#CFCFCF"),
            hoverinfo="skip",
            showlegend=False,
        )

    if lines.ndim != 3 or lines.shape[1] != 2 or lines.shape[2] != 3:
        raise ValueError(f"lines must be (n,2,3), got {lines.shape}")

    n = lines.shape[0]
    # interleave: p0, p1, sep for each segment
    xb = np.empty(3 * n, dtype=lines.dtype)
    yb = np.empty(3 * n, dtype=lines.dtype)
    zb = np.empty(3 * n, dtype=lines.dtype)

    xb[0::3] = lines[:, 0, 0]
    xb[1::3] = lines[:, 1, 0]
    xb[2::3] = np.nan
    yb[0::3] = lines[:, 0, 1]
    yb[1::3] = lines[:, 1, 1]
    yb[2::3] = np.nan
    zb[0::3] = lines[:, 0, 2]
    zb[1::3] = lines[:, 1, 2]
    zb[2::3] = np.nan

    return go.Scatter3d(
        x=xb,
        y=yb,
        z=zb,
        mode="lines",
        line=dict(width=width, color="#CFCFCF"),
        hoverinfo="skip",
        showlegend=False,
    )


def make_box_trace_from_lengths(
    box, *, name="Box", width=2.0, color="#66ccff"
) -> go.Scatter3d:
    """L = [Lx, Ly, Lz]; axis-aligned box with origin at (0,0,0)."""
    Lx, Ly, Lz = box
    corners = np.array(
        [
            [0, 0, 0],
            [Lx, 0, 0],
            [0, Ly, 0],
            [Lx, Ly, 0],
            [0, 0, Lz],
            [Lx, 0, Lz],
            [0, Ly, Lz],
            [Lx, Ly, Lz],
        ],
        dtype=float,
    )

    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),  # bottom
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),  # top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),  # sides
    ]
    xs, ys, zs = [], [], []
    for i, j in edges:
        xs += [corners[i, 0], corners[j, 0], None]
        ys += [corners[i, 1], corners[j, 1], None]
        zs += [corners[i, 2], corners[j, 2], None]

    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(width=width, color=color),
        name=name,
        hoverinfo="skip",
        showlegend=False,
    )


def extract_bonds(molecules: Sequence[Molecule], box: StructureArray) -> StructureArray:
    """Extract all bonds from a list of molecules at once using MIC (orthorhombic). Returns (n,2,3)."""
    if not molecules:
        return np.empty((0, 2, 3), dtype=FLOAT_DTYPE)
    parts: list[StructureArray] = []
    for mol in molecules:
        segs = extract_molecule_bonds(mol, box)
        if segs.size:
            parts.append(segs)
    if not parts:
        return np.empty((0, 2, 3), dtype=FLOAT_DTYPE)
    return np.concatenate(parts, axis=0)


def extract_molecule_bonds(molecule: Molecule, box: StructureArray) -> StructureArray:
    """Extract bonds as minimum-image segments between two points. Returns (n_bonds, 2, 3)."""
    if molecule.structure_nodes is None:
        return np.empty((0, 2, 3), dtype=FLOAT_DTYPE)

    def mic_delta(
        p0: npt.NDArray[FloatPrecision], p1: npt.NDArray[FloatPrecision]
    ) -> npt.NDArray[FloatPrecision]:
        d = p1 - p0
        d = d - box * np.round(d / box)
        return d.astype(FLOAT_DTYPE, copy=False)

    lines_points: list[
        tuple[npt.NDArray[FloatPrecision], npt.NDArray[FloatPrecision]]
    ] = []
    seen: set[tuple[int, int]] = set()
    pos = molecule.data.positions

    for node in molecule.structure_nodes:
        i = node.molecular_index
        p0 = pos[i]
        for nb in node.neighbors:
            j = nb.molecular_index
            a, b = (i, j) if i < j else (j, i)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            d = mic_delta(p0, pos[j])
            lines_points.append((p0, p0 + d))

    if not lines_points:
        return np.empty((0, 2, 3), dtype=FLOAT_DTYPE)
    return np.asarray(lines_points, dtype=FLOAT_DTYPE)
