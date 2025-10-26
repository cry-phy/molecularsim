import pytest
import numpy as np
from molecsim.physics.calculations import radial_distances


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

