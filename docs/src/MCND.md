
An instance of the multicommodity capacitated fixed-charge network design problem (FCNDP) is given by a directed simple graph ``D = (N,A)``, a set of commodities ``K``, an arc-capacity vector ``c`` and two cost vectors ``r`` and ``f``.
Each commodity ``k\in K`` corresponds to a triplet ``(o^k, d^k, q^k)`` where ``o^k\in N`` and ``d^k\in N`` are nodes corresponding to the origin and the destination of the demand and ``q^k \in \mathbb{N}^*`` is its volume. For each arc ``(i,j) \in A``, ``c_{ij}>0`` corresponds to the maximum amount of flows which can be routed through ``(i,j)`` and ``f_{ij} > 0`` corresponds to the fixed cost of using the arc ``(i,j)`` to route some flows. For each arc ``(i,j) \in A`` and each demand ``k \in K``, ``r_{ij}^k >0``
corresponds to the cost of routing one unit of commodity ``k \in K`` through the arc ``(i,j)``.

A solution of the multicommodity capacitated fixed-charge network design problem consists in routing the demands while ensuring the capacity of the arcs. The cost of a solution is determined by the sum of the routing costs associated with each commodity and the fixed costs associated with all the arcs used for routing a demand.


A standard model for MCND introduces two sets of variables: the flow variables ``x_{ij}^k`` representing the flow of commodity ``k`` that passes through the arc ``(i,j)`` and the variables ``y_{ij}`` representing design variables of the network, i.e. if we use or not a certain arc ``(i,j)``. Hence the model is:
```math
\begin{aligned}
    Z^{MCND} & =\min \sum_{(i,j) \in A} \left( \sum_{k \in K} r^k_{ij}x_{ij}^k+f_{ij}y_{ij}\right)\\
     s.t. & \sum_{j\in N^+_i}x_{ij}^k - \sum_{j\in N^-_i}x_{ji}^k=b^k_i& \forall i \in N, \forall k \in K, \\
     & \sum_{k \in K}x_{ij}^k\leq c_{ij}y_{ij},& \forall (i,j)\in A, \\
     & 0\leq x^k_{ij}\leq q^k  & \forall (i,j)\in A, \forall k \in K,\\
     &y_{ij} \in \{0,1\}, & \forall (i,j)\in A, 
\end{aligned}
```
where

```math 
b^k_i=\left\{
\begin{aligned}
    q^k & \; if \; i = o^k, \\
    -q^k & \; if \; i = d^k, \\
    0 \; & \; otherwise. 
\end{aligned}
\right.
```

The flows of each commodity can be restricted, without loss of generality, to not entering its origin node nor leaving its destination one. This can be done by adding the following equations in the model:

``  x_{ij}^k = 0`` for each arc ``(i,j) \in \delta^{\rm in}(o^k) \cup \delta^{\rm out}(d^k)``


