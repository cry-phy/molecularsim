from collections.abc import Iterable
from typing import Sequence
from molecsim.system.data import FLOAT_DTYPE, StructureArray
import numpy.typing as npt
import numpy as np
import plotly.graph_objects as go
from molecsim.system.molecule import Molecule
from molecsim.system.system import SimSystem

# Use common color scheme (RasMol)
# http://www.pdg.cnb.uam.es/cursos/BioInfo2002/pages/Farmac/CMU_course/Periodic/ElemList.htm
_ELEMENT_COLOR_RASMOL = {
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


def color_for(elem: str | Iterable[str]) -> str | Iterable[str]:
    """Return RasMol-style hex color for an element, fallback to deep pink."""
    if isinstance(elem, str):
        e = (elem[:1].upper() + elem[1:].lower()) if elem else ""
        return _ELEMENT_COLOR_RASMOL.get(e, _ELEMENT_COLOR_RASMOL["DEFAULT"])
    elif isinstance(elem, Iterable):
        es = []
        for single in elem:
            e = (single[:1].upper() + single[1:].lower()) if single else ""
            es.append(_ELEMENT_COLOR_RASMOL.get(e, _ELEMENT_COLOR_RASMOL["DEFAULT"]))
        return es


def plot_system(sim: SimSystem, fig: go.Figure | None = None) -> go.Figure:
    """Generate a Figure plotting all molecules and the bonds."""
    lines = extract_bonds(sim.molecules)
    sticks = lines_to_sticks(lines)
    atoms = positions_to_atoms(sim.data.positions, sim.data.atoms_as_str())
    if fig is not None:
        # Augment an existing figure (no layout mutations)
        fig.add_traces([sticks, atoms])  # bonds first, atoms second
        return fig

    # New figure: build + adjust layout
    fig = go.Figure(data=[sticks, atoms])
    invis_dict = {"visible": False}
    fig.update_layout(
        title=getattr(sim, "name", None) or "System",
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        scene={
            "xaxis": invis_dict,
            "yaxis": invis_dict,
            "zaxis": invis_dict,
            "aspectmode": "data",
        },
        showlegend=False,
    )
    fig.update_scenes(camera={"eye": {"x": 1.6, "y": 1.6, "z": 1.2}})
    return fig


def positions_to_atoms(
    positions: StructureArray,
    atoms: npt.NDArray[np.object_],
) -> go.Scatter3d:
    """Generate atoms plottable from positions and types."""
    x, y, z = positions.transpose()
    colors = color_for(atoms)
    return go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker={"size": 8, "color": colors},
        line={"width": 1, "color": "black"},
        name="atoms",
    )


def lines_to_sticks(
    lines: StructureArray,
    width: int = 5,
) -> go.Scatter3d:
    """Convert endpoints array of shape (n, 2, 3) to a single Plotly Scatter3d lines trace."""
    if lines.size == 0:
        return go.Scatter3d(
            mode="lines", x=[], y=[], z=[], line={"width": width}, showlegend=False
        )

    assert lines.ndim == 3 and lines.shape[1] == 2 and lines.shape[2] == 3, (
        "lines must be (n,2,3)"
    )

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
        line={"width": width},
        hoverinfo="skip",
        showlegend=False,
    )


def extract_bonds(molecules: Sequence[Molecule]) -> StructureArray:
    """Extract all bonds from a list of molecules at once."""
    bonds = [extract_molecule_bonds(molec) for molec in molecules]
    data = np.concat(bonds, axis=0)
    return data


def extract_molecule_bonds(molecule: Molecule) -> StructureArray:
    """Extract bonds as lines between two points.

    Returns as endpoints array shape (n, 2, dim).
    """
    lines_points = []
    if molecule.structure_nodes is None:
        return np.empty(0, dtype=FLOAT_DTYPE)
    for node in molecule.structure_nodes:
        line_start = molecule.data.positions[node.molecular_index]
        neighbor_lines = [
            (line_start, molecule.data.positions[neighbor.molecular_index])
            for neighbor in node.neighbors
        ]
        lines_points.extend(neighbor_lines)
    return np.array(lines_points, dtype=FLOAT_DTYPE)
