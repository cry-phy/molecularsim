import numpy as np
import pytest

from molecsim.system.data import SystemData
from molecsim.system.potentials import coulomb, lennard_jones


def test_coulomb_single_molecule():
    """Test Coulomb potential for a single molecule."""
    # Define a simple molecule with two atoms
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    charges = np.array([1.0, -1.0], dtype=np.float64)
    data = SystemData.as_owner(atoms=["A", "B"], positions=positions, charges=charges)
    # Calculate Coulomb potential
    potential = coulomb(data, data)
    # Expected potential: k * q1 * q2 / r, where k=1, q1=1, q2=-1, r=1
    expected_potential = -1.0
    assert np.isclose(potential, expected_potential)


def test_coulomb_two_molecules():
    """Test Coulomb potential for two interacting molecules."""
    # Define two molecules
    positions1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    charges1 = np.array([1.0, -1.0], dtype=np.float64)
    data1 = SystemData.as_owner(atoms=["A", "B"], positions=positions1, charges=charges1)

    positions2 = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
    charges2 = np.array([-1.0, 1.0], dtype=np.float64)
    data2 = SystemData.as_owner(atoms=["C", "D"], positions=positions2, charges=charges2)

    # Combine positions and charges into a single SystemData
    combined_positions = np.concatenate([positions1, positions2], axis=0)
    combined_charges = np.concatenate([charges1, charges2], axis=0)
    combined_atoms = ["A", "B", "C", "D"]
    combined_data = SystemData.as_owner(atoms=combined_atoms, positions=combined_positions, charges=combined_charges)

    # Calculate Coulomb potential between the two molecules
    potential = coulomb(combined_data, data1)

    # Calculate expected potential
    # Distances:
    # Intramolecular (within data1): A-B
    # Intermolecular (between data1 and data2): A-C, A-D, B-C, B-D
    def coulomb_interaction(q1, q2, r):
        return q1 * q2 / r

    # Intramolecular (A-B)
    intra_potential = coulomb_interaction(1.0, -1.0, 1.0)

    # Intermolecular (A-C, A-D, B-C, B-D)
    inter_potential = (
        coulomb_interaction(1.0, -1.0, 3.0) +  # A-C
        coulomb_interaction(1.0, 1.0, 4.0) +   # A-D
        coulomb_interaction(-1.0, -1.0, 2.0) + # B-C
        coulomb_interaction(-1.0, 1.0, 3.0)    # B-D
    )

    # Total potential
    expected_potential = intra_potential + inter_potential

    print("Calculated potential:", potential)
    print("Expected potential:", expected_potential)

    assert np.isclose(potential, expected_potential)


def test_lennard_jones_single_molecule():
    """Test Lennard-Jones potential for a single molecule."""
    # Define a simple molecule with two atoms
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    data = SystemData.as_owner(atoms=["A", "B"], positions=positions, charges=np.zeros(2))

    # Set Lennard-Jones parameters
    A = 1.0
    B = 1.0

    # Calculate Lennard-Jones potential
    potential = lennard_jones(data, data, A=A, B=B)

    # Expected potential: A/r^12 - B/r^6, where r = 1
    expected_potential = 0.0
    assert np.isclose(potential, expected_potential)


def test_lennard_jones_two_molecules():
    """Test Lennard-Jones potential for two interacting molecules."""
    # Define two molecules
    positions1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    data1 = SystemData.as_owner(atoms=["A", "B"], positions=positions1, charges=np.zeros(2))

    positions2 = np.array([[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
    data2 = SystemData.as_owner(atoms=["C", "D"], positions=positions2, charges=np.zeros(2))

    # Combine positions into a single SystemData
    combined_positions = np.concatenate([positions1, positions2], axis=0)
    combined_atoms = ["A", "B", "C", "D"]
    combined_data = SystemData.as_owner(atoms=combined_atoms, positions=combined_positions, charges=np.zeros(4))

    # Set Lennard-Jones parameters
    A = 1.0
    B = 1.0

    # Calculate Lennard-Jones potential between the two molecules
    potential = lennard_jones(combined_data, data1, A=A, B=B)

    # Calculate expected potential
    # Distances: A-B=1, A-C=3, A-D=4, B-C=2, B-D=3
    def lj(r, A, B):
        r6 = r**6
        r12 = r**12
        return A / r12 - B / r6

    # Intramolecular (A-B)
    intra_potential = lj(1, A, B)

    # Intermolecular (A-C, A-D, B-C, B-D)
    inter_potential = lj(3, A, B) + lj(4, A, B) + lj(2, A, B) + lj(3, A, B)

    # Total potential
    expected_potential = intra_potential + inter_potential

    print("Calculated potential:", potential)
    print("Expected potential:", expected_potential)

    assert np.isclose(potential, expected_potential)
