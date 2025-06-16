"""
LR(ins::cpuInstanceCWL,π)

#Arguments:
-`ins`: an instance structure of the Bin packing Problem
-`π`: a vector of Lagrangian Multipliers
# Arguments:
-`ins`: an instance structure of the Bin packing Problem,
-`π`: a vector of Lagrangian Multipliers,
- `unsplittable`: a boolean that say if true that is the unsplittable version of CWL where we have only binary variables.

This function solves the Lagrangian Knapsack Sub-Problem for the provided instance using the provided Lagrangian multipliers vector. 
"""
function LR(ins::cpuInstanceCWL, π; unsplittable = false)
	π=Vector{Float64}(π[1,:])
	x = zeros(Float32, ins.I, ins.J)
	y = zeros(Float32, ins.I)

	obj = -sum(π)
	obj1 = 0

	LRI = []
	for i in 1:ins.I
		xp = zeros(Float32, ins.J)

		if unsplittable
			obj1 = ins.f[i] - solve_knapsack(ins.J, -ins.c[i, :] - π, ins.d, ins.q[i], xp)
		else
			obj1 = ins.f[i] - solve_knapsack_continuous(ins.J, -ins.c[i, :] - π, ins.d, ins.q[i], xp)
		end
		if obj1 < 0
			y[i] = 1
			x[i, :] .= xp
		else
			obj1 = 0
		end
		append!(LRI, obj1)
		obj += obj1
	end

	return obj, x, y
end