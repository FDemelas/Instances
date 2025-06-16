We relax the constraints:

``
 \sum_{g=1}^Gp_{g,t}\geq P^d_t \qquad  t=1,2,\cdots,T
``
with multipliers ``\lambda\in \mathbb{R}_+^T``.
and

``
\sum_{g=1}^G (P^{\max}_g\alpha_{g,t}-p_{g,t}) \qquad t=1,2,\cdots,T
``
with multiliers ``\mu\in \mathbb{R}_+^T``,
obtaining a sub-problem that decomposes by generators.
For a given ``g \in G`` the associated sub-problem is:
                                       
```math
\begin{align*}
LR_g(\lambda,\mu) \equiv ~ \min ~ & \sum_{t=1}^T\left( C_g^{nl}\alpha_{g,t} + C_g^{mr}p_{g,t} + C_g^{up}\gamma_{g,t} \right)+ \sum_{t=1}^T\lambda_t p_{g,t}+ \sum_{t=1}^T \mu_t(P^{\max}_g\alpha_{g,t} - p_{g,t}) & t=1,2,\cdots,T \\
     & P_g^{\min}\leq p_{g,t} \leq P^{\max}_g\alpha_{g,t} & g=1,2,\cdots,G,\;t=2,\cdots,T\\
     & p_{g,t}-p_{g,t-1} \leq P^{ru}_g\alpha_{g,t-1}+P_g^{su}\gamma_{g,t} & t=2,\cdots,T \\
     & p_{g,t-1}-p_{g,t} \leq P^{rd}_g\alpha_{g,t}+P^{sd}_g\eta_{gt} & t=1,2,\cdots,T \\
     & \sum_{u=\max\{t-T^u_g+1,1\}}\gamma_{gu} \leq \alpha_{g,t} & t=1,2,\cdots,T  \\
     & \sum_{u=\max\{t-T^u_g+1,1\}}\eta_{gu} \leq 1-\alpha_{g,t} & t=1,2,\cdots,T  \\
     & \alpha_{g,t} - \alpha_{g,t-1} = \gamma_{g,t} - \eta_{g,t} & t=2,3,\cdots,T \\
     & 1 \geq \gamma_{g,t}+\eta_{g,t} & t=1,2,\cdots,T \\
     & \alpha_{g,t} \in \{0,1\} & g=1,\cdots,G,t=1,\cdots,T \\
     & \gamma_{g,t} \in \{0,1\} & g=1,\cdots,G,t=1,\cdots,T \\
     & \eta_{g,t} \in \{0,1\} & g=1,\cdots,G,t=1,\cdots,T \\
     & p_{g,t} \geq 0     & \alpha_{g,t} \in \{0,1\} & g=1,\cdots,G,t=1,\cdots,T   
\end{align*}
```

then the Lagrangian Dual bound can be computed as

```math
\max_{\lambda,\mu \in \mathbb{R}^T\times\mathbb{R}^T}\left( \sum_{g=1}^GLR_g(\lambda,\mu)+\sum_{t=1}^T\lambda_tP^d_t + \sum_{t=1}^T\mu_tP^r_t\right)
```
