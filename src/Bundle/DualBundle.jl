"""
Create the Dual Master Problem associated to the Bundle `B` using `t` as regularization parameter.
It is suggested to use this function only in the initialization and then consider to update the model instead of creating a new one.
"""
function create_DQP(B::DualBundle, t::Float64)
	model = Model(Gurobi.Optimizer)
	set_optimizer_attribute(model, "NonConvex", 2)
	set_attribute(model, "Threads", 1)
	set_attribute(model, "Method", 0)

	g = 0 < B.params.max_β_size < Inf ? B.G[:, 1:B.size] : B.G
	α = 0 < B.params.max_β_size < Inf ? B.α[1:B.size] : B.α

	@variable(model, 1 >= θ[1:size_Bundle(B)[1]] >= 0)

	@constraint(model, conv_comb, ones(size_Bundle(B)[1])' * θ == 1)

	if size(θ)[1] == 1
		quadratic_part = @expression(model, LinearAlgebra.dot(g .* θ, g .* θ))
		linear_part = @expression(model, LinearAlgebra.dot(α, θ))
		@objective(model, Min, (1 / 2) * quadratic_part .+ 1 / t * linear_part)
	else
		@objective(model, Min, (1 / 2) * LinearAlgebra.dot(g * θ, g * θ) + 1 / t * LinearAlgebra.dot(α, θ))
	end
	set_silent(model)
	set_time_limit_sec(model, 20.0)
	return model
end


"""
Solve the Dual Master Problem and update the associated objective function.
Note: the objective function is rescaled using the regularization parameter as it is scaled inversely in the Dual Master Problem formulation in order to allow faster re-optimization.
"""
function solve_DQP(B::DualBundle)
	optimize!(B.model)

	if dual_status(B.model) == NO_SOLUTION
		return 0
	end

	B.objB = B.params.t * JuMP.objective_value(B.model)

end

"""
Update the Dual Master Problem formulation.
We check if we add a new component or not.
The second case happens when we found a point for which the associated sub-gradient was equal to the one of an already visited point.
In this case we did not "add" a new point, but just update the information and so the reoptimization could be easier.
If we add a new component, then we add the associated varibale to the problem, by creating a new variable adding it to the simplex constraints and to the objective function.
If we changed the parameter t or the stabilization point we have to update all the linear part in the objective function.
"""
function update_DQP!(B::DualBundle, t_change = true, s_change = true)
	if !(length(B.model.obj_dict[:θ]) == B.size)
		for _ in 1:(size_Bundle(B)-length(B.model.obj_dict[:θ]))
			θ_tmp = @variable(B.model, upper_bound = 1, lower_bound = 0)
			set_normalized_coefficient(B.model.obj_dict[:conv_comb], θ_tmp, 1)
			push!(B.model.obj_dict[:θ], θ_tmp)
		end
		for tmp in 1:length(B.model.obj_dict[:θ])
			set_objective_coefficient(B.model, B.model.obj_dict[:θ][B.li], B.model.obj_dict[:θ][tmp], (1 / 2) * B.Q[B.li, tmp])
		end
	end
	if t_change || s_change
		set_objective_coefficient(B.model, B.model.obj_dict[:θ], 1 / B.params.t * B.α[1:B.size])
	else
		set_objective_coefficient(B.model, B.model.obj_dict[:θ][B.li], 1 / B.params.t * B.α[B.li])
	end
end

"""
Returns the linearization error associated to the stabilization point.
"""
function αS(B::DualBundle)
	return B.α[B.s]
end

"""
Returns the sub-gradient associated to the stabilization point.
"""
function gS(B::DualBundle)
	return B.G[:, B.s]
end


"""
Returns the stabilization point.
"""
function zS(B::DualBundle)
	return B.z[:, B.s]
end


"""
Returns the objective function associated to the stabilization point.
"""
function objS(B::DualBundle)
	return B.obj[B.s]
end

"""
Returns the size of the bundle.
"""
function size_Bundle(B::DualBundle)
	return B.size
end

"""
Returns the linearization error associated to the point in the i-th position in the bundle.
"""
function linearization_error(B::DualBundle, i::Int)
	if i == B.s
		return 0
	end
	return linearization_error(B.G[:, i], zS(B), B.z[:, i], objS(B), B.obj[i])
end

"""
Returns the linearization error associated to the point `z` (that has the gradient `g` and objective value `obj`) with respect to the stabilization point `zS` (that has objective value `objS`).
"""
function linearization_error(g::AbstractVector, zS::AbstractVector, z::AbstractVector, objS::Real, obj::Real)
	return g' * (zS - z) - (objS - obj)
end

"""
Updates all the linearization errors in the Bundle.
"""
function update_linearization_errors(B::DualBundle)
	B.α[B.s] = 0
	for i in 1:size_Bundle(B)
		B.α[i] = linearization_error(B, i)
	end
end

"""
Removes components from the Bundle that are not used for many consecutives iterations.
"""
function remove_outdated(B::DualBundle, ϵ = 1e-6)
	sB = size_Bundle(B)
	remove_idx = []
	keep_idx = []
	for i in 1:sB
		if i == B.s
			append!(keep_idx, i)
		else
			keep = 0
			how_much_iter_in = 0
			for j in eachindex(B.cumulative_θ)
				if i <= length(B.cumulative_θ[j])
					keep += (B.cumulative_θ[j][i] > ϵ)
					how_much_iter_in += 1
				end
			end
			if keep > 0 || (how_much_iter_in <= B.params.remotionStep)
				append!(keep_idx, i)
			else
				append!(remove_idx, i)
				B.size -= 1
			end
		end
	end
	if 0 < B.params.max_β_size < Inf
		B.size = length(keep_idx)
		while (B.size > B.params.max_β_size)
			θ = B.θ[keep_idx]
			i = argmin(θ)
			append!(remove_idx, keep_idx[i])
			popat!(keep_idx, keep_idx[i])
			B.size -= 1
		end
	end
	first_idxs = collect(1:length(keep_idx))
	if 0 < B.params.max_β_size < Inf
		first_idxs = collect(1:length(keep_idx))
		B.G[:, first_idxs] = B.G[:, keep_idx]
		B.Q[first_idxs, first_idxs] = B.Q[keep_idx, keep_idx]
		B.α[first_idxs] = B.α[keep_idx]
		B.z[:, first_idxs] = B.z[:, keep_idx]
		B.obj[first_idxs] = B.obj[keep_idx]
		B.θ = B.θ[keep_idx]
		B.size = length(keep_idx)
	else
		B.G = B.G[:, keep_idx]
		B.Q = B.Q[keep_idx, keep_idx]
		B.α = B.α[keep_idx]
		B.z = B.z[:, keep_idx]
		B.obj = B.obj[keep_idx]
		B.size = length(keep_idx)
		B.θ = B.θ[keep_idx]
	end
	sort!(remove_idx, rev = true)
	for h in remove_idx
		if h < B.li
			B.li -= 1
		end
		if h < B.s
			B.s -= 1
		elseif h == B.s
			println("Trying to remove stabilization point")
		end
		delete(B.model, B.model.obj_dict[:θ][h])
		deleteat!(B.model.obj_dict[:θ], h)
		for i in 1:size(B.cumulative_θ, 1)
			if h < size(B.cumulative_θ[i], 1)
				deleteat!(B.cumulative_θ[i], h)
			end
		end
	end
	B.cumulative_θ = B.cumulative_θ[max(1, end - B.params.remotionStep):end]

end

"""
Updates the solution `B.θ` of the Dual Master Problem. 
Compute the new searching direction `B.w`, obtained as convex combination of the gradients contained in the bundle with weights `B.θ`.
It also stock sever quantities that will be used in the condition that determinines if change the stailization point (i.e. make a Serious Step or a Null Step) and in the t-strategies.
"""
function compute_direction(B::DualBundle)
	B.θ = value.(B.model.obj_dict[:θ])
	push!(B.cumulative_θ, copy(B.θ))
	B.w = (0 < B.params.max_β_size < Inf) ? (B.G[:, 1:B.size] * B.θ) : (B.G * B.θ)
	B.linear_part = B.α[1:B.size]'B.θ
	B.quadratic_part = B.w'B.w
	B.vStar = (B.params.t * B.quadratic_part + B.linear_part)
	B.ϵ = B.linear_part + B.params.t_star * B.quadratic_part / 2
end

"""
Updates the Bundle information.
It adds to the bundle the information associated to the stabilization point `z`, knowing that the objective value in this point is `obj` and the sub-gradient is `g`.
"""
function update_Bundle(B::DualBundle, z, g, obj)
	z = reshape(z, :)
	g = Float32.(reshape(g, :))
	already_exists = false
	for j in 1:B.size
		#println(sum(abs.(B.G[:, j] - g))) 
		if (sum(abs.(B.G[:, j] - g)) < 1.0e-8)
			#println("already exists bundle size $(B.size) obj $(obj)")
			already_exists = true
			#if B.obj[j] < obj
			B.α[j] = linearization_error(g, zS(B), z, objS(B), obj)
			B.obj[j] = obj
			B.z[:, j] = z
			#end
			B.li = j
			push!(B.all_objs, obj)
			return
		end

	end

	if !(already_exists)
		if 0 < B.params.max_β_size < Inf
			q = B.G[:, 1:B.size]' * g
			i = B.size + 1
			B.obj[i] = obj
			B.z[:, i] = z
			B.G[:, i] = g
			B.Q[1:i-1, i] = q
			B.Q[1, 1:i-1] = q
			B.Q[i, i] = g'g
			B.α[i] = linearization_error(B, i)
		else
			B.z = hcat(B.z, z)
			q = B.G' * g
			B.G = hcat(B.G, g)
			B.Q = vcat(hcat(B.Q, q), vcat(q, g'g)')
			push!(B.obj, obj)
			α = linearization_error(B, size_Bundle(B) + 1)
			push!(B.α, α)
		end
		B.li = B.size + 1
		push!(B.all_objs, obj)
		B.size += 1
	end
end

"""
Return `true` is the stopping criteria is satisfied and `false` if not.
The stopping criteria require that the sum of the quadratic part (weighted with the hyper-parameter `t_star`) and the linear part are small.
"""
function stopping_criteria(B::DualBundle)
	return B.params.t_star * B.quadratic_part + B.linear_part <= B.params.ϵ * (max(0, objS(B)) + 1)
end

"""
Computes the new trial point by moving from the stabilization point `zS(B)` to a step `B.params.t` through the direction `B.w`.
"""
function trial_point(B::DualBundle)
	return zS(B) + B.params.t * B.w
end

"""
Main function for the Bundle.
It maximize the function `ϕ` using the bundle method, with a previously initialized bundle `B`.
It has two additional input parameters:
- `t_strat`: the t-Strategy, by default it is the contant t-strategy (i.e. the regularization parameter for the Dual Master Problem is always keeped fixed).
- `unstable`: if `true` we always change the stabilization point using the last visited point, by default `false` (change it only if you know what you are doing).
"""
function solve!(B::DualBundle, ϕ::AbstractConcaveFunction; t_strat::abstract_t_strategy = constant_t_strategy(), unstable::Bool = false)
	times = Dict("times" => [], "trial point" => [], "ϕ" => [], "update β" => [], "SS/NS" => [], "update DQP" => [], "solve DQP" => [], "remove outdated" => [])
	t0 = time()
	for epoch in 1:B.params.maxIt
		t1 = time()
		z = trial_point(B)#reshape(, sizeInputSpace(ϕ)) 
		append!(times["trial point"], (time() - t1))

		t0 = time()
		obj, g = value_gradient(ϕ, z) # to optimize
		append!(times["ϕ"], (time() - t0))

		t0 = time()
		s = B.s
		update_Bundle(B, z, g, obj)

		t = B.params.t
		t_strategy(B, B.li, t_strat, unstable)
		append!(times["update β"], (time() - t0))

		t0 = time()
		update_DQP!(B, t == B.params.t, s == B.s)
		append!(times["update DQP"], (time() - t0))

		t0 = time()
		solve_DQP(B) # to optimize
		append!(times["solve DQP"], (time() - t0))

		compute_direction(B)

		if B.params.log
			println("  $(objS(B)-obj <= B.objB) : $(objS(B)) - $(obj) = $(objS(B) - obj) $((objS(B)-obj <= B.objB) ? "<=" : ">") $(B.objB)  epoch number ", epoch, " optimal LR: ", objS(B), "  Bundle Size", size_Bundle(B)[1], " step t ", B.params.t)
		end

		t0 = time()
		if epoch >= B.params.remotionStep
			#remove_outdated(B)
		end
		append!(times["remove outdated"], (time() - t0))

		append!(B.ts, B.params.t)
		#if stopping_criteria(B)
		#	println("Satisfied stopping criteria")
		#	return true, times
		#end
		push!(B.memorized["times"], time() - t1)
	end
	times["times"] = B.memorized["times"]
	return false, times
end

"""
Function that handle is keep or change the stabilization point and then handle the increases or decreases of the regularization parameter.
"""
function t_strategy(B::DualBundle, i::Int, ts::abstract_t_strategy, unstable::Bool=false)
	if B.obj[i] - objS(B) >= B.params.m1 * B.vStar || unstable
		B.CSS += 1
		B.CNS = 0
		B.s = i
		update_linearization_errors(B)
		increment_t(B, ts)
	else
		B.CNS += 1
		B.CSS = 0
		decrement_t(B, ts)
	end
end