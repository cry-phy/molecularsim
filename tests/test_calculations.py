import pytest
import numpy as np
from molecsim.physics.calculations import radial_distances

from molecsim.data.data import SystemData
from molecsim.data.properties import SystemProperties
from molecsim.physics.potentials import coulomb
from molecsim.physics.geometry import extra_interaction_mask


def test_radial_distribution_function(simsys):
    """Test the radial distribution function calculation."""
    bins, rdfs = radial_distances(
        simsys.data, ["A"], ["B"], simsys.properties, num_bins=5
    )

    # Check that bins are correctly spaced
    assert len(bins) == 5
    assert np.all(np.diff(bins) > 0)

    # Check that the RDF dictionary contains the expected keys
    assert ("A", "B") in rdfs

    # Check that the RDF values are non-negative
    rdf_values = rdfs[("A", "B")]
    assert np.all(rdf_values >= 0)


def _make_large_system():
    """Four atoms in a line, charges alternate: +1, -1, +1, -1."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    charges = np.array([+1.0, -1.0, +1.0, -1.0], dtype=np.float64)
    atoms = ["A", "B", "C", "D"]
    big = SystemData.as_owner(atoms=atoms, positions=positions, charges=charges)
    return big


def test_extra_interaction_mask_is_correct_for_borrow():
    """Borrowed subview (contiguous in big array) should exclude its own rows only."""
    big = _make_large_system()
    small_borrow = big.as_borrow(start=2, size=2)  # atoms with parent rows 2,3

    mask = extra_interaction_mask(big, small_borrow)

    expected = np.ones((len(big), len(small_borrow)), dtype=bool)
    expected[2:4, :] = False  # exclude parent rows 2,3 only

    assert mask.shape == expected.shape
    assert np.array_equal(mask, expected), (
        "Borrow mask must exclude only the borrowed rows."
    )


@pytest.mark.xfail(
    strict=True,
    reason="Copied molecule uses data_interval()->(0,len), excludes wrong rows; carry parent indices.",
)
def test_extra_interaction_mask_mislabels_for_copy():
    """
    Copied molecule has _start_index=0; current extra_interaction_mask(blank-by-interval)
    excludes rows (0..len-1) instead of the true parent rows (2,3).
    This test documents the bug (should xfail until fixed).
    """
    big = _make_large_system()

    idx = np.array([2, 3])
    atoms_str = big.atom_ids_to_str(big.atoms[idx])
    small_copy = SystemData.as_owner(
        atoms=atoms_str,
        positions=big.positions[idx],
        charges=big.charges[idx],
        mapping=(
            big._atom_str_to_int,
            big._atom_int_to_str,
        ),  # share mapping dicts only
    )

    mask = extra_interaction_mask(big, small_copy)

    expected = np.ones((len(big), len(small_copy)), dtype=bool)
    expected[2:4, :] = False  # true parent rows

    assert np.array_equal(mask, expected), (
        "Mask for copied molecule must exclude parent rows (2,3)."
    )


def test_coulomb_energy_differs_for_copy_when_mask_is_wrong(mock_props):
    """
    Practical impact: with wrong mask, Coulomb energy for the same physical molecule
    differs between borrow vs copy. Uses your mock_props fixture.
    """
    big = _make_large_system()

    small_borrow = big.as_borrow(start=2, size=2)
    e_borrow = coulomb(big, small_borrow, mock_props)

    idx = np.array([2, 3])
    small_copy = SystemData.as_owner(
        atoms=big.atom_ids_to_str(big.atoms[idx]),
        positions=big.positions[idx],
        charges=big.charges[idx],
        mapping=(big._atom_str_to_int, big._atom_int_to_str),
    )
    e_copy = coulomb(big, small_copy, mock_props)

    # With current interval-based masking, these should NOT match.
    assert not np.isclose(e_borrow, e_copy), (
        "Energies unexpectedly equal; if you’ve fixed masking by parent indices, "
        "update this test to assert equality instead."
    )
