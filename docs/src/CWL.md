A CFL instance is defined by a set ``J`` of facilities and a set ``K`` of customers. With each facility ``j \in J`` is associated a capacity ``c_j`` and a fixed cost ``f_j``. A demand ``q^k`` is associated with each customer ``k \in K``. Finally, a service cost ``r_{j}^k`` is associated with each facility ``j \in J`` and each customer ``k \in K`` and corresponds to the cost of serving one unit of demand of customer ``k`` from facility ``j``.

A CFL solution consists in a subset of open facilities as well as the amount of demand served from these open facilities to each customer. Its cost is the sum of the fixed costs over the open facilities plus the sum over every facility ``j \in J`` and every customer ``k \in K`` of the unitary service cost ``r_{j}^k`` multiplied by the amount of demand of customer ``k`` served from facility ``j``.

## MILP formulation

A standard model for the CFL \citep{akinc77} introduces two sets of variables: the continuous variables ``x_j^k`` representing the amount of demand of customer ``k`` served from facility ``j``, and the binary variables ``y_j`` indicating whether facility ``j \in J`` is open. Hence, a formulation of the problem is:
```math
\begin{align}
	\min & \sum_{j \in J} \left(f_jy_j + \sum_{k \in K}r_j^kx_j^k\right) &    &  \\
	s.t. & \sum_{j \in J} x_j^k = q^k                 & \forall k \in K           &  \\
	 & \sum_{k \in K} x_j^k \leq c_jy_j     & \forall j \in J           &    \\
	     & 0 \leq x_j^k \leq q^k                      & \forall j \in J,\; \forall k \in K  \\
	     & y_j\in \{0,1\}                          & \forall j \in J
\end{align}
```
