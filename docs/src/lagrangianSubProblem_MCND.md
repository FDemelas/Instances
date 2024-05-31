
Let ``\pi_i^k`` be the Lagrangian multiplier associated with node ``i \in N`` and ``k \in K``. The Lagrangian relaxation decomposes by arcs and one obtains a subproblem for each arc ``(i,j)\in A`` of the form:

```math
    \begin{align*}
        L_{ij}(\pi) & \equiv\min_{x, y} L(\pi, x, y) = \min_{x, y} f_{ij}y_{ij} + \sum_{k \in K} r^k_{ij}(\pi)x_{ij}^k\\
        s.t. & \sum_{k \in K}x_{ij}^k\leq c_{ij}y_{ij},\\
        & 0\leq x^k_{ij}\leq q^k, &  \forall k \in K, \\
        &y_{ij} \in \{0,1\}, & 
    \end{align*}
```

where, defining `` K(ij) = \{k \in K \mid j \neq o(k) `` and `` i \neq d(k)\}``, the function ``r_{ij}^k(\pi)`` can be defined as:


```math
r_{ij}^k(\pi)=\left\{
\begin{align*}
    r_{ij}^k+\pi_{i}^k-\pi_{j}^k \; & \text{if} \; k \in K(ij),\\
    0 \qquad & \; \text{otherwise}.
\end{align*} \right.
```

For each ``(i,j) \in A``, computing ``L_{ij}(\pi)`` reduces to computing a continuous knapsack problem (case ``y_{ij} = 1``) and to compare with 0 (case ``y_{ij} = 0``).


Lagrangian duality implies that the value ``L(\pi)=\sum_{(i,j) \in A} L_{ij}(\pi)- \sum_{i \in N}\sum_{k \in K} \pi_i^k b^k_i`` is a lower bound for the MCND and the best one is obtained by solving the following Lagrangian dual problem:

```math
LD = \max_{\pi \in \mathbb{R}^{N \times K}} L(\pi)  
```
