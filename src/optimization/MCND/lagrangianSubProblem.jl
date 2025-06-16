"""
	cijk(ins::instance, e, k, π)

	# Arguments:
	- `ins`: instance of the problem
	- `e`: index of the edge
	- `k`: index of the commodity
	- `π`: a Lagrangian multipliers Vector

	Return the cost associated with demand k in the subproblem of L(π) related to arc e
"""
cijk(ins::cpuInstanceMCND, e, k, π) = isInKij(ins, k, e) ? routing_cost(ins, e, k) + π[k, tail(ins, e)] - π[k, head(ins, e)] : 0



"""
# Arguments:

	- `ins`: cpuInstanceMCND of the problem
	- `π`: a Lagrangian multipliers Vector
	- (`x`,`y`): two vector of the same size of the (primal) solution vector for the sub-problem, not initalized.
	- `demands`: a vector [1:K] with the (unsorted) incexes for the demands
	- `e`: index of the arc

	Compute LR_{ij}, the value of the Lagrangian problem associated with arc of index e.
	It requires the solution vector x,y as input and update them durnig the resolution.

	Demands is a permutation of 1:p.
"""
function LR(ins::cpuInstanceMCND, π, x, y, demands, e)

	u = capacity(ins, e)
	c̃ij = Float32[cijk(ins, e, k, π) for k in 1:sizeK(ins)]
	sort!(demands, by = k -> c̃ij[k])

	b = capacity(ins, e)
	cost = 0.0
	i = 1
	while b > 0 && i <= length(demands) && c̃ij[demands[i]] < 0
		vol_dem = min(b, volume(ins, demands[i]))
		b -= vol_dem
		cost += vol_dem * c̃ij[demands[i]]
		i += 1
	end

	# Update x
	x[:, e] .= 0

	if cost + fixed_cost(ins, e) < 0
		cost = cost + fixed_cost(ins, e)
		b = capacity(ins, e)
		i = 1
		while b > 0 && i <= length(demands) && c̃ij[demands[i]] < 0
			vol_dem = min(b, volume(ins, demands[i]))
			b -= vol_dem

			x[demands[i], e] = vol_dem
			i += 1
		end
		y[e] = 1
	else
		cost = 0.0
		y[e] = 0
	end


	return cost
end


"""
# Arguments:

- `ins`: cpuInstanceMCND of the problem
- `π`: a Lagrangian multipliers Vector

Compute LR.
regY is the regularization vector for y and α a multiplicative parameter.
It does not requires the solution vector x,y as input and it return them as output.
"""
function LR(ins::cpuInstanceMCND, π)
	x = zeros(Float32, sizeK(ins), sizeE(ins))
	y = zeros(Float32, sizeE(ins))
	
	demands = collect(1:sizeK(ins))

	LagBound = 0.0
	for e in 1:sizeE(ins)
		LagBound += LR(ins, π, x, y, demands, e)
	end
	return LagBound - constantLagrangianBound(ins, π), x, y
end

"""
# Arguments:

	- `ins`: cpuInstanceMCND of the problem
	- `π`: a Lagrangian multipliers Vector

	Compute the component of the bound that does not depends by the soluyion x,y of the sub-problem,
	but only by the lagrangian multipliers vector and the demands data.
"""
function constantLagrangianBound(ins::cpuInstanceMCND, π)
	cst = 0.0
	for k in 1:sizeK(ins)
		cst += volume(ins, k) * π[k, origin(ins, k)] - volume(ins, k) * π[k, destination(ins, k)]
	end
	return cst
end

"""
# Arguments:

	- `ins`: cpuInstanceMCND of the problem
	- `π`: Lagrangian multipliers
	- (`x`, `y`): Solution of the Lagrangian problem

Return the value of the Lagrangian sub-problem associated,
using the lagrangian multipliers vector π and the primal solution (x,y).
"""
value_LR(ins::cpuInstanceMCND, π, x, y) = sum(value_LR_a(ins, ia, π, x, y) for ia ∈ 1:sizeE(ins)) - constantLagrangianBound(ins, π)

"""
# Arguments:

	- `ins`: cpuInstanceMCND of the problem
	- `ia`: index of the arc
	- `π`: Lagrangian multipliers
	- (`x`, `y`): Solution of the Lagrangian problem


	Return the value of the Lagrangian problem associated with arc of index ia,
	using the lagrangian multipliers vector π and the primal solution (x,y).
"""
value_LR_a(ins::cpuInstanceMCND, ia, π, x, y) = fixed_cost(ins, ia) * y[ia] + sum((routing_cost(ins, ia, ik) + π[ik, tail(ins, ia)] - π[ik, head(ins, ia)]) * x[ik, ia] for ik ∈ 1:sizeK(ins) if isInKij(ins, ik, ia))
