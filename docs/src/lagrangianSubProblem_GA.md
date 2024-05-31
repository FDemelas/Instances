The implemented Lagrangian relaxation of the GA problem is the one obtained by dualizing:

``\max_{\bm{x}}  \sum_{i \in I} \sum_{j \in J} p_{ij}x_{ij}``

For ``i \in I``, let ``\pi_{i} \geq 0`` be the Lagrangian multipliers associated to the constraint with item $i$.
For each bin ``j`` the subproblem becomes:
```math
\begin{align*}
(LR_j(\bm{\pi})) \quad & \max_{\bm{x}} \sum_{i \in I}\sum_{j \in J} (p_{ij}-\pi_{i}) x_{ij} \span \span\\
& \sum_{i \in I}w_{ij}x_{ij} \leq c_j\\
& x_{ij} \in \{0,1\} && \forall i \in I
\end{align*}
```

It corresponds to an integer knapsack with ``|I|`` binary variables. For ``\bm{\pi}\ge \bm{0}``, the Lagrangian bound ``LR(\bm{\pi})`` is:
 ```math
    LR(\bm{\pi}) = \sum_{j \in J}LR_j(\bm{\pi})+\sum_{i \in I}\pi_i.
```


The Lagrangian dual can be then written as:
```math
    \min_{\bm{\pi} \in \R^{|I|}_{\geq 0}}LR(\bm{\pi})
```


