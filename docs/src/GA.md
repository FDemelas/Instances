A Generalized Assignment (GA) instance is defined by a set ``I`` of items and a set ``J`` of bins.
Each bin ``j`` is associated with a certain capacity ``c_j``.
For each item ``i \in I`` and each bin ``j \in J$, $p_{ij}`` is the profit of assigning item ``i`` to bin ``j``, and ``w_{ij}`` is the weight of item ``i`` inside bin ``j``.

Considering a binary variable ``x_{ij}`` for each item and each bin that is equal to one if and only if item ``i`` is assigned to bin ``j``, the GA problem can be formulated as:
```math
\begin{align}
& \max_{\bm{x}}  \sum_{i \in I} \sum_{j \in J} p_{ij}x_{ij} \\
& \sum_{j \in J} x_{ij}  \leq 1 && \forall i \in I  \\
& \sum_{i \in I} w_{ij} x_{ij}  \leq c_j && \forall j \in J \\
& x_{ij} \in \{0,1\} && \forall i \in I ,\; \forall j\in J .
\end{align}
```
