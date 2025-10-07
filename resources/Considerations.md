# Considerations for Implementations.

Feel free to add Considerations you want to discuss/have implemented here.

If urgent please connect this to an Issue.

## Computational Load

To Compute the base interactions for $n_f$ force terms for $N$ Molecules we need to compute $O(N^2)$ molecule-molecule pair terms u(i, j) to achieve the total energy $U$.

This is an expensive update why i would use the direct update method where we only recompute the energy difference $\Delta U$ for moving a molecule $i$ to position $i'$ with the equation:

$$ \Sum_{j\neqi}  \left[u(i', j) - u(i, j)\right]$$

which is linear in complexity (if we implement proximity even $O(1)$).

## Data Layout

To allow for simple and fast iteration over the atoms we want to keep them contigous in a SoA approach. Therefore it could be usefull to hold the Atoms and thjeir properties in a Simulation class. 
Then we separate the molecule update logic into a different Molecule Structures, that only reference the underlying data. Since we have static number of molecules and no additions/switching of atoms to/between molecules (limitations of MC simulations) it could prove usefull not to hold indices to the atoms but rather numpy views.

