We use the same formulation used in [^1], one of the standard formulations in literature, also called 3-binary variable formulation by [^2].

Given ``G`` generator and ``T`` number of time periods where decisions are taken the Unit Commitment problem consists asking that generators meet all the demand ``P^d_t`` in each time period ``t``, but also keeping a sufficient amount of backup of backup ``P^r_t`` that can be activated quickly.
It is further required that each generator power output has to be taken in its minimum/maximum limits ``P_g^{\min/\max}``, and their power outputs can change within ramp down/up rates ``P^{rd/ru}_g`` when the generator is operativeng and ``P^{su/sd}_g`` when the generator startup/shutdown.
Finally, if a generator is switched on/off it should stay on/off for a given time period ``T^{u/d}_g``

The solution should minimize a total cost composed by
- a no-load cost ``C^{nl}_g`` of generator ``g`` that we have to pay only to keep the generator is on
- a marginal cost ``C^{mr}_g`` of generator ``g`` that we pay for generating one unit of power with generator ``g``.
- a startup cost ``C^{up}_g``  that we pay to turn on the generator ``g`` 

We consider the following variables:
- for each generator ``g\in \{1,\cdots, G\}`` and for each period ``t\in\{1,\cdots, T\}`` a  binary variable ``\alpha_{g,t}`` to represent if the generator ``g`` is on in the period ``t``
- for each generator ``g\in \{1,\cdots, G\}`` and for each period ``t\in\{1,\cdots, T\}`` a binary variable  ``\gamma_{g,t}`` to represent if the generator ``g`` starts up in the preiod ``t``
- for each generator ``g\in \{1,\cdots, G\}`` and for each period ``t\in\{1,\cdots, T\}`` a binary variable ``\eta_{g,t}`` to represent if the generator shut down in the preiod ``ŧ``
- for each generator ``g\in \{1,\cdots, G\}`` and for each period ``t\in\{1,\cdots, T\}`` a non-negative variables ``p_{g,t}`` to represent the power output of the generator ``g`` in period ``t``.

and then the problem can be described as

```math
\begin{align*}
\min & \sum_{t=1}^T\sum_{g=1}^G\left( C_g^{nl}\alpha_{g,t} + C_g^{mr}p_{g,t} + C_g^{up}\gamma_{g,t} \right)\\
     & \sum_{g=1}^Gp_{g,t}\geq P^d_t & t=1,2,\cdots,T \\
     & \sum_{g=1}^G (P^{\max}_g\alpha_{g,t}-p_{g,t}) & t=1,2,\cdots,T \\
     & P_g^{\min}\leq p_{g,t} \leq P^{\max}_g\alpha_{g,t} & g=1,2,\cdots,G,\;t=2,\cdots,T\\
     & p_{g,t}-p_{g,t-1} \leq P^{ru}_g\alpha_{g,t-1}+P_g^{su}\gamma_{g,t} & g=1,2,\cdots,G,t=2,\cdots,T \\
     & p_{g,t-1}-p_{g,t} \leq P^{rd}_g\alpha_{g,t}+P^{sd}_g\eta_{gt} & g=1,2,\cdots,G,t=1,2,\cdots,T \\
     & \sum_{u=\max\{t-T^u_g+1,1\}}\gamma_{gu} \leq \alpha_{g,t} & g=1,2,\cdots,G,t=1,2,\cdots,T  \\
     & \sum_{u=\max\{t-T^u_g+1,1\}}\eta_{gu} \leq 1-\alpha_{g,t} & g=1,2,\cdots,G,t=1,2,\cdots,T  \\
     & \alpha_{g,t} - \alpha_{g,t-1} = \gamma_{g,t} - \eta_{g,t} & g=1,2,\cdots,G,t=2,3,\cdots,T \\
     & 1 \geq \gamma_{g,t}+\eta_{g,t} & g=1,2,\cdots,G,t=1,2,\cdots,T \\
     & \alpha_{g,t} \in \{0,1\} & g=1,\cdots,G,t=1,\cdots,T \\
     & \gamma_{g,t} \in \{0,1\} & g=1,\cdots,G,t=1,\cdots,T \\
     & \eta_{g,t} \in \{0,1\} & g=1,\cdots,G,t=1,\cdots,T \\
     & p_{g,t} \geq 0     & \alpha_{g,t} \in \{0,1\} & g=1,\cdots,G,t=1,\cdots,T
\end{align*}
```

[^1]: Sugishita, N., Grothey, A., and McKinnon, K. Use of Machine Learning Models to Warmstart Column Generation for Unit Commitment. INFORMS Journal on Computing, January 2024.
[^2]: Ostrowski, J., M. F. Anjos, and A. Vannelli (2012). “Tight mixed integer linear programming formulations for the unit commitment problem”. In: IEEE Transactions on Power Systems 27.1, pp. 39–46.
