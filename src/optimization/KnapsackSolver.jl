"""
# Arguments:
		- `n`: the number of objects 
		- `c`: the costs vector of the objects
		- `w`: the weights vector of the objects
		- `C`: the capacity of the knapsack
		- `x`: the decision variable vector, it should have size n, the solution will be contained in this vector after the call of the function.

Solve a continuous knapsack with the provided parameters, save the solution in the vector `x` and return the objective value.
"""
function solve_knapsack_continuous(n, c, w, C, x)
	if n == 0
		return 0
	end

	demands = collect(1:length(c))
	sort!(demands, by = k -> -c[k] / w[k])

	CC = C
	cost = 0.0
	i = 1
	while C > 0 && i <= length(demands) && c[demands[i]] > 0
		vol_dem = min(C, w[demands[i]])
		C -= vol_dem
		cost += vol_dem / w[demands[i]] * c[demands[i]]
		i += 1
	end

	# Update x
	x .= 0

	C = CC
	i = 1
	while C > 0 && i <= length(demands) && c[demands[i]] > 0
		vol_dem = min(C, w[demands[i]])
		C -= vol_dem

		x[demands[i]] = vol_dem / w[demands[i]]
		i += 1
	end

	return cost
end

"""
# Arguments:
		- `n`: the number of objects 
		- `c`: the costs vector of the objects
		- `w`: the weights vector of the objects
		- `C`: the capacity of the knapsack
		- `x`: the decision variable vector, it should have size n, the solution will be contained in this vector after the call of the function.

Solve a binary knapsack with the provided parameters, save the solution in the vector `x` and return the objective value.
"""
function solve_knapsack(n::Int64, c, w, C, x)
	if n == 0
		return 0
	end
	m = zeros(Float32, n + 1, C + 1)
	for i in 2:(n+1)
		for j in 2:(C+1)
			if w[i-1] > j - 1
				m[i, j] = m[i-1, j]
			else
				m[i, j] = max(m[i-1, j], m[i-1, j-Int64(w[i-1])] + c[i-1])
			end
		end
	end

	W = C + 1
	for i in (n+1:-1:2)
		if W <= 0
			break
		end
		if m[i, Int64(W)] <= m[i-1, Int64(W)]
			continue
		else
			x[i-1] = 1
			W -= w[i-1]
		end
	end
	return m[n+1, C+1]
end