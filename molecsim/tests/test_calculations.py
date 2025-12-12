import pytest
import numpy as np
from molecsim.physics.calculations import radial_distances

from molecsim.data.data import SystemData
from molecsim.physics.potentials import coulomb
from molecsim.physics.geometry import extra_interaction_mask


def test_radial_distribution_function(simsys):
    """Test the radial distribution function calculation."""
    bins, rdfs = radial_distances(
        simsys.data, ["A"], ["B"], simsys.properties, simsys.model, num_bins=5
    )

    # Check that bins are correctly spaced
    assert len(bins) == 5
    assert np.all(np.diff(bins) > 0)

    # Check that the RDF dictionary contains the expected keys
    assert ("A", "B") in rdfs

    # Check that the RDF values are non-negative
    rdf_values = rdfs[("A", "B")]
    assert np.all(rdf_values >= 0)


def _make_large_system(mock_model):
    """Four atoms in a line, charges alternate: +1, -1, +1, -1."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    charges = np.array([+1.0, -1.0, +1.0, -1.0], dtype=np.float64)
    atoms = ["A", "B", "C", "D"]
    big = SystemData.as_owner(
        np.array([mock_model.registry.index_of(a) for a in atoms]),
        positions=positions,
        charges=charges,
    )
    return big


def test_extra_interaction_mask_is_correct_for_borrow(mock_model):
    """Borrowed subview (contiguous in big array) should exclude its own rows only."""
    big = _make_large_system(mock_model)
    small_borrow = big.as_borrow(start=2, size=2)  # atoms with parent rows 2,3

    mask = extra_interaction_mask(big, small_borrow)

    expected = np.ones((len(big), len(small_borrow)), dtype=bool)
    expected[2:4, :] = False  # exclude parent rows 2,3 only

    assert mask.shape == expected.shape
    assert np.array_equal(mask, expected), (
        "Borrow mask must exclude only the borrowed rows."
    )
