"""
LR(ins::cpuInstanceGA,π)

#Arguments:
-`ins`: an instance structure of the Bin packing Problem
-`π`: a vector of Lagrangian Multipliers

This function solves the Lagrangian Knapsack Sub-Problem for the provided instance using the provided Lagrangian multipliers vector. 
"""
function LR(ins::cpuInstanceGA, π)
	π=Vector{Float64}(π[1,:])
	x = zeros(Float32, ins.I, ins.J)

	obj = -sum(π)
	obj1 = 0

	for j in 1:ins.J
		xp = zeros(Float32, ins.I)
		obj1 = solve_knapsack(ins.I, ins.p[:, j] + π, ins.w[:, j], ins.c[j], xp)
		x[:,j] = xp
        obj += obj1
	end

	return obj, x, nothing
end
