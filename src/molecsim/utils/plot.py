from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go

from molecsim.system.data import FLOAT_DTYPE, FloatPrecision, StructureArray
from molecsim.system.molecule import Molecule
from molecsim.system.system import SimSystem

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


def plot_system(sim: SimSystem, fig: go.Figure | None = None) -> go.Figure:
    """Generate a Plotly Figure with all molecules and bonds (lines + circles)."""
    lines = extract_bonds(sim.molecules)  # (n,2,3) or (0,2,3)
    sticks = lines_to_sticks(lines)  # bonds first
    atoms = positions_to_atoms(sim.data.positions, sim.data.atoms_as_str())

    if fig is not None:
        fig.add_traces([sticks, atoms])
        return fig

    fig = go.Figure(data=[sticks, atoms])

    invis = {"visible": False}
    dark = "#0b0f19"

    fig.update_layout(
        title=getattr(sim, "name", None) or "System",
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        paper_bgcolor=dark,
        plot_bgcolor=dark,
        font=dict(color="#ECECEC"),
        scene=dict(
            xaxis=invis,
            yaxis=invis,
            zaxis=invis,
            aspectmode="data",
            bgcolor=dark,  # 3D canvas background
        ),
    )
    fig.update_scenes(camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)))
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


def extract_bonds(molecules: Sequence[Molecule]) -> StructureArray:
    """Extract all bonds from a list of molecules at once. Returns (n,2,3)."""
    parts: list[StructureArray] = []
    for molec in molecules:
        segs = extract_molecule_bonds(molec)
        if segs.size:
            parts.append(segs)
    if not parts:
        return np.empty((0, 2, 3), dtype=FLOAT_DTYPE)
    return np.concatenate(parts, axis=0)


def extract_molecule_bonds(molecule: Molecule) -> StructureArray:
    """Extract bonds as segments between two points. Returns (n, 2, 3)."""
    if molecule.structure_nodes is None:
        return np.empty((0, 2, 3), dtype=FLOAT_DTYPE)

    lines_points: list[
        tuple[npt.NDArray[FloatPrecision], npt.NDArray[FloatPrecision]]
    ] = []
    for node in molecule.structure_nodes:
        i = node.molecular_index
        p0 = molecule.data.positions[i]
        for nb in node.neighbors:
            j = nb.molecular_index
            lines_points.append((p0, molecule.data.positions[j]))

    if not lines_points:
        return np.empty((0, 2, 3), dtype=FLOAT_DTYPE)
    return np.asarray(lines_points, dtype=FLOAT_DTYPE)
