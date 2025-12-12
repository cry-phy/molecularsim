from pathlib import Path
import numpy as np
import pytest

from molecsim.utils.parsers import parse_z_matrix
from tests.conftest import mock_model

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"


def _get_symbols(mol) -> list[str]:
    """Try common attribute names/paths for atomic symbols on Molecule."""
    # direct attributes on Molecule
    for attr in ("atoms", "symbols", "elements", "atomic_symbols"):
        if hasattr(mol, attr):
            vals = getattr(mol, attr)
            return list(vals) if not isinstance(vals, list) else vals

    # nested: Molecule.data / Molecule._data / Molecule._data_slices
    for container_attr in ("data", "_data", "_data_slices"):
        if hasattr(mol, container_attr):
            container = getattr(mol, container_attr)
            for attr in ("atoms", "symbols", "elements", "atomic_symbols", "_atoms"):
                if hasattr(container, attr):
                    vals = getattr(container, attr)
                    return list(vals) if not isinstance(vals, list) else vals

    raise AttributeError("Could not find atomic symbols on Molecule.")


def _get_coords(mol) -> np.ndarray:
    """Try common attribute names/paths for coordinates on Molecule."""
    # direct attributes on Molecule
    for attr in ("coords", "coordinates", "positions"):
        if hasattr(mol, attr):
            arr = getattr(mol, attr)
            if isinstance(arr, np.ndarray):
                return arr

    # nested: Molecule.data / Molecule._data / Molecule._data_slices
    for container_attr in ("data", "_data", "_data_slices"):
        if hasattr(mol, container_attr):
            container = getattr(mol, container_attr)
            for attr in ("coords", "coordinates", "positions", "_positions"):
                if hasattr(container, attr):
                    arr = getattr(container, attr)
                    if isinstance(arr, np.ndarray):
                        return arr

            # method-style accessors
            for meth in (
                "positions",
                "coords",
                "coordinates",
                "get_positions",
                "get_coords",
            ):
                if hasattr(container, meth):
                    maybe = getattr(container, meth)
                    arr = maybe() if callable(maybe) else maybe
                    if isinstance(arr, np.ndarray):
                        return arr

    raise AttributeError("Could not find coordinates on Molecule.")


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return angle ABC in degrees given 3× vectors (A,B,C)."""
    v1 = a - b
    v2 = c - b
    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)
    cosang = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosang)))


@pytest.mark.parametrize("filename", ["water_cartesian.com", "water_zmatrix.com"])
def test_parse_water_shapes_and_types(filename: str, mock_model):
    """Test that parsing water files results in correct shapes and types."""
    path = DATA_DIR / filename
    mol = parse_z_matrix(str(path), mock_model)

    coords = _get_coords(mol)
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (3, 3)

    symbols = mock_model.names(mol.data.atoms).tolist()
    print(symbols)
    assert len(symbols) == 3
    assert symbols.count("O") == 1
    assert symbols.count("H") == 2


@pytest.mark.parametrize("filename", ["water_cartesian.com", "water_zmatrix.com"])
def test_parse_water_geometry_is_correct(filename: str, mock_model):
    """Test that parsing water files results in correct geometry."""
    path = DATA_DIR / filename
    mol = parse_z_matrix(str(path), mock_model)

    coords = _get_coords(mol)
    symbols = mock_model.names(mol.data.atoms).tolist()
    idx_O = symbols.index("O")
    idx_H = [i for i, s in enumerate(symbols) if s == "H"]
    assert len(idx_H) == 2

    o_coord = coords[idx_O]
    h1_coord = coords[idx_H[0]]
    h2_coord = coords[idx_H[1]]

    # O–H ~ 0.96 Å
    d1 = np.linalg.norm(h1_coord - o_coord)
    d2 = np.linalg.norm(h2_coord - o_coord)
    assert np.isclose(d1, 0.96, atol=2e-2)
    assert np.isclose(d2, 0.96, atol=2e-2)

    # H–O–H ~ 104.5°
    ang = _angle(h1_coord, o_coord, h2_coord)
    assert np.isclose(ang, 104.5, atol=0.5)


def test_unsupported_extension_raises(tmp_path: Path, mock_model):
    """Test that an unsupported file extension raises a ValueError."""
    bad = tmp_path / "foo.xyz"
    bad.write_text("not a gaussian file\n")
    with pytest.raises(ValueError):
        _ = parse_z_matrix(str(bad), mock_model)
