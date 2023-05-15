# Lecture1 Introduction to WFC

## 1. Algorithm

### Definition

1. Read the input bitmap and count NxN patterns.
   1. (optional) Augment pattern data with rotations and reflections.
2. Create an array with the dimensions of the output (called "wave" in the source). Each element of this array represents a state of an NxN region in the output. A state of an NxN region is a superposition of NxN patterns of the input with boolean coefficients (so a state of a pixel in the output is a superposition of input colors with real coefficients). False coefficient means that the corresponding pattern is forbidden, true coefficient means that the corresponding pattern is not yet forbidden.
3. Initialize the wave in the completely unobserved state, i.e. with all the boolean coefficients being true.
4. Repeat the following steps:
   1. Observation:
      1. Find a wave element with the minimal nonzero entropy. If there is no such elements (if all elements have zero or undefined entropy) then break the cycle (4) and go to step (5).
      2. Collapse this element into a definite state according to its coefficients and the distribution of NxN patterns in the input.
   2. Propagation: propagate information gained on the previous observation step.
5. By now all the wave elements are either in a completely observed state (all the coefficients except one being zero) or in the contradictory state (all the coefficients being zero). In the first case return the output. In the second case finish the work without returning anything.

### WFC Algorithm

- Initial all the tiles $T_i$, with constraints in each edge (for a 2D tiles, it has four edges, each edge has some constraints)
- Initial tile map array (int) $T_{Map}[n][m]$, with size $n × m$, all the grids are $-1$ (denotes blank)
- Initial slot map array $T_{Slot}[n][m]$, each slot initially maintains the choices of all tiles $T_i$
- **while** (`FindMinimumEntropy`)
  - $i, j \leftarrow$ the grid has the minimum entropy
  - get the available slots of the grid: $s \leftarrow$ $T_{Slot}[i][j]$ 
  - $T_{Map}[i][j] \leftarrow $ random choose one tiles in the slot $s$
  - **for each** side of this slot $s$
    - $s' \leftarrow$ the adjacent side of slot $s$
    - `ReducePossibility(s', s)`

### Find Minimum Entropy

- **for each** slot $s$ in the slot map array, find $s$ with then minimum choices of tiles
- if have multiple $s$, random return one $s$ and its position $i,j$

### Reduce Possibility(s', s)

- $S_{constraint} \leftarrow$ Set() // all the possible constraint in slot $s'$
- **for each** available tile $T_i$ in the slot $s$
  - union the constraint of tile $T_i$
- $s'.constraint  \leftarrow s'.constraint  \wedge S_{constraint}$
- **for each** side of this tiles $s'$
  - $s'' \leftarrow$ the adjacent side of slot $s$
  - `ReducePossibility(s'', s')`




$$
\left(
 \begin{matrix}
   0 & 1 & 2 & 3
  \end{matrix}
  \right)
  \\
W=\left[
 \begin{matrix}
   1 & 1 & 1 & 1
  \end{matrix}
  \right]
$$





$$
\left[
 \begin{matrix}
   \text{up}\\
   \text{down}\\
   \text{left}\\
   \text{right}
  \end{matrix}
  \right]
\left[
 \begin{matrix}
   0 & 0 & 1 & 1 \\
   1 & 1 & 0 & 0 \\
   0 & 1 & 0 & 0 \\
   0 & 0 & 1 & 0
  \end{matrix}
  \right]
$$

$$
\text{Entropy}=\sum (w_i=1)
$$




