A Lagrangian relaxation of the CFL problem is obtained by dualizing \eqref{cfl:customerService}. In that case, the Lagrangian dual gives a better bound than the continuous relaxation \citep{geoffrion_lagrangean_1974} while the relaxed problem can be solved efficiently as it decomposes by facility. Let ``\pi^k`` be the Lagrangian multiplier of constraint~\eqref{cfl:customerService} associated with customer ``k \in K``. The relaxed problem ``LR(\pi)`` is given by:
```math
(LR(\pi)) \quad  \min_{(x,y) \text{ satisfies }\eqref{cfl:capa}-\eqref{cfl:yBinary}} \sum_{j \in J}f_jy_j + \sum_{k \in K} q^k\pi^k +  \sum_{j \in J}  \sum_{k \in K} (r_j^k - \pi^k)x_j^k
```

The relaxed problem ``LR(\pi)`` can be decomposed by facility and the sub-problem associated with facility ``j\in J`` is the following:
```math
\begin{align*}
	(LR_{j}(\pi)) \quad & \min f_jy_j +\sum_{k \in K} (r_j^k - \pi^k)x_j^k   \\
	s.t.               & \sum_{k \in K} x_j^k \leq c_jy_j    \\
	                   & 0 \leq x_j^k \leq q^k   & \forall k \in K \\
	                   & y_{j} \in \{0,1\} 
\end{align*}
```

The value of the relaxed Lagrangian problem ``LR(\pi)`` is equal to ``\sum_{j \in J}  LR_{j}(\pi) +  \sum_{k \in K} q^k\pi^k``.
``LR(\pi)`` contains only one non-continuous variable. If ``y_j = 0`` equals 0, then ``x_j^k = 0`` by constraints \eqref{cfl_RLj:capa} and \eqref{cfl_RLj:xBounds}. If ``y_j = 1``, then the problem reduces to a continuous knapsack problem which can be solved by ordering the customers following decreasing values ``r_j^k - \pi^k`` and setting ``x_j`` to ``\max\{\min\{q^k, c_{j} - \sum_{k \in K(k)} q^k\}, 0\}`` where ``K(k)`` denotes the set of customers that precede ``k`` in the order. Solving ``LR_j(\pi)`` consists of choosing among these two solutions the one which is minimum so ``LR_j(\pi)`` can be solved in ``O(|K|\log(|K|))``.
% By Lagrangian duality theory, a lower bound of the CFL problem is provided by the value ``LR(\pi)=\sum_{j=1}^M\cdot \hat{LR}_j(\pi)-\sum_{i=1}^N\pi_i``.
The best Lagrangian lower bound for CFL can be found by solving the Lagrangian dual problem:
```math
	\max_{\pi\in \mathbb{R}^K}LR(\pi)
```
