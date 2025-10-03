# Considerations for Implementations.

Feel free to add Considerations you want to discuss/have implemented here.

If urgent please connect this to an Issue.

## Compuational Load

To Compute the base interactions for $n_f$ force terms for $N$ Molecules we need to compute $O(N^2)$ molecule-molecule pair terms u(i, j) to achieve the total energy $U$.

This is an expensive update why i would use the direct update method where we only recompute the energy difference $\Delta U$ for moving a molecule $i$ to position $i'$ with the equation:

$$ \Sum_{j\neqi}  \left[u(i', j) - u(i, j)\right]$$

which is linear in complexity (if we implement proximity even $O(1)$).


