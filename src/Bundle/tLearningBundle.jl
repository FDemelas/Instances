"""
Asbtract type for Machine-Learning Based Bundles that specialize a DualBundle.
"""
abstract type AbstractMLBundle <: DualBundle end

"""
Structure for the Bundle Based on Machine Learning in which we still solve the Dual Master problem,
	but we use a neural network based t-strategy.
"""
mutable struct DeepBundle <: AbstractMLBundle
	Q::Matrix{Float32}
	G::Matrix{Float32}
	z::Matrix{Float32}
	α::Vector{Float32}
	s::Int64
	model::Model
	w::Vector{Any}
	θ::Vector{Float32}
	objB::Float32
	obj::Vector{Float32}
	cumulative_θ::Vector{Vector{Float32}}
	params::BundleParameters
	nn::Any
	lt::AbstractTModelFactory
	back_prop_idx::Any
	ws::Any
	θ2s::Any
	features::Any
	ts::Any
	CSS::Int64
	CNS::Int64
	ϕ0::AbstractVector
	size::Int
	all_objs::Vector{Float32}
	exactGrad::Bool
	li::Int
	memorized::Dict
	vStar::Float32
	ϵ::Float32
	linear_part::Float32
	quadratic_part::Float32
end

"""
Created and initialize a `DeepBundle`.
"""
function initializeBundle(bt::tLearningBundleFactory, ϕ, t, z, lt::AbstractTModelFactory, nn::Any = create_NN(lt), max_bundle_size = 500; exactGrad = true, instance_features::Bool = false)
	B = DeepBundle([;;], [;;], [;;], [], -1, Model(Gurobi.Optimizer), [], [], 0, [0], [Float32[]], BundleParameters(), nn, lt, [], [], [], [], [], 0, 0, [], 1, Float32[], exactGrad, 1, Dict("times" => []), 0.0, 0.0, 0.0, 0.0)
	B.params.max_β_size = max_bundle_size + 1
	Flux.reset!(nn)
	obj, g = value_gradient(ϕ, z)
	B.s = 1
	z = reshape(z, :)
	g = reshape(g, :)
	if 0 < B.params.max_β_size < Inf
		B.α = zeros(Float32, B.params.max_β_size)
		B.G = zeros(Float32, (length(g), B.params.max_β_size))
		B.z = zeros(Float32, (length(z), B.params.max_β_size))
		B.G[:, 1] = g
		B.z[:, 1] = z
		B.obj = zeros(Float32, B.params.max_β_size)
		B.obj[1] = obj
		B.Q = zeros(Float32, (B.params.max_β_size, B.params.max_β_size))
		B.Q[1, 1] = g'g
	else
		B.α = [0]
		B.G = reshape(g, (length(g), 1))
		B.Q = Float32[g'g;;]
		B.obj = [obj]
		B.z = z
	end

	B.model = create_DQP(B, t)

	B.w = zeros(length(z))
	B.θ = ones(1)

	B.params.t = t
	B.ϕ0 = instance_features ? Float32.(vcat(quantile(ϕ.inst.r), quantile(ϕ.inst.f), quantile([ϕ.inst.K[i][3] for i in eachindex(ϕ.inst.K)]), quantile(zeros(size(ϕ.inst.c))))) : zeros(Float32, 20)

	f = create_features(B, B.nn)

	B.params.t = cpu(B.nn(f, B))
	#push!(B.ts,B.params.t) 

	B.model = create_DQP(B, B.params.t)
	solve_DQP(B)
	compute_direction(B)

	B.w = B.G[:, 1]
	B.cumulative_θ = [Float32[1.0]]
	push!(B.ts, B.params.t)
	push!(B.ws, B.w)
	push!(B.θ2s, θ2(B))
	push!(B.features, ϕ)
	push!(B.all_objs, obj)
	return B
end

"""
Compute the new trial-point for the `DeepBundle`.
"""
function trial_point(B::DeepBundle)
	ϕ = create_features(B, B.nn)
	ϕ = device(ϕ)
	return zS(B) + cpu(B.nn(ϕ, B)) .* B.w
end

isNaN(x) = return (x === NaN)

function θ2(B)
	base = [i for i in eachindex(B.θ) if B.θ[i] > 10^(-12)]
	e = ones(length(base))
	θ2 = zeros(length(B.α))
	Qb = B.Q[base, base]
	if rank(Qb) >= size(Qb, 1)
		Qm1 = inv(Qb)
		θ2[base] = Qm1 * ((e' * Qm1 * B.α[base]) / (e' * Qm1 * e) * e - B.α[base])
	end
	return B.G * θ2
end

"""
Compute the trial direction d as `t*w`, by receiving as input the t-parameter `t`, the index `i` of the iteration and the Bundle `B`.
It computes then `B.ws[i] .* t`
"""
ws(t, i, B) = B.ws[i] .* t

"""
ChainRule for the back-ward computation of the function `ws`.
In particular if the parameter `B.exactGrad` is `true`, than we can provide a better approximation of `∂w/∂t` inspired by the KKT conditions.
"""
function ChainRulesCore.rrule(::typeof(ws), t, i, B)
	value = B.ws[i] .* t
	gs = B.ws[i]
	if B.exactGrad
		gs .-= (1 / t) * B.θ2s[i]
	end
	loss_pullback(dl) = (NoTangent(), gs' * dl, NoTangent(), NoTangent())
	return value, loss_pullback
end

"""
Computation of the bundle execution and the associated Sub-gradient.
"""
function Bundle_value_gradient!(B::DeepBundle, ϕ::AbstractConcaveFunction, sample = true, single_prediction::Bool = false)
	B.features = []
	B.ws = []
	B.ts = []
	B.θ2s = []
	B.back_prop_idx = []
	ϕ0 = device(create_features(B, B.nn))
	if (single_prediction)
		B.params.t = cpu(B.nn(ϕ0, B))
	end

	for epoch in 1:B.params.maxIt
		if !(single_prediction)
			f = device(create_features(B, B.nn))
			B.params.t = cpu(B.nn(f, B))
		else
			f = ϕ0
		end

		update_DQP!(B)
		solve_DQP(B)
		compute_direction(B)

		δ = LinearAlgebra.dot(B.α[1:B.size], B.θ)
		if epoch >= B.params.remotionStep
			remove_outdated(B)
		end

		z = cpu(zS(B) + B.params.t * B.w)

		obj, g = value_gradient(ϕ, z) # to optimize

		old_size, old_s, old_objS = B.size, B.s, B.obj[B.s]
		update_Bundle(B, z, g, obj)
		new_size, new_s = B.size, B.s

		if isnan(obj)
			println("Error the objective is NaN")
			return B
		end

		changed_s = false
		if B.obj[B.li] - old_objS > B.params.m1 * B.objB
			B.CSS += 1
			B.CNS = 0
			B.s = B.li
			update_linearization_errors(B)
			changed_s = true
		else
			if new_s == old_s && new_size == old_size
				changed_s = true
			end
			B.CNS += 1
			B.CSS = 0
		end


		push!(B.ws, B.w)
		push!(B.θ2s, θ2(B))
		push!(B.ts, B.params.t)
		push!(B.features, f)
		append!(B.all_objs, B.obj[B.li])
		if changed_s
			push!(B.back_prop_idx, epoch)
		end
	end
	return false
	#end
end

"""
Forward, Backward and parameters updates.
The gradient computation is still done (partially) 'at-hand' and possibly we can improve it by creating a properly well defined function completely differentiated using Zygote rules.
For the moment we compute the bundle execution and we stock some further informations that will be used to compute the backward pass.
The function that computes the forward and stock the information usefull for the backward is called `Bundle_value_gradient!` (this function stock the informations without computing the backward).
	The backward and the parameter update is made here in the function `train!`.
"""
function train!(B::DeepBundle, ϕ, state; samples = 1, telescopic = false, γ = 0.9, δ = 0.00001, normalization_factor = 1, oneshot = true, gold = zeros(1), single_prediction::Bool = false)
	sample = B.nn.sample
	par = Flux.trainable(B.nn)
	z0 = zS(B)

	t = B.params.t
	first_run = B.size <= 1 ? true : false

	#state = B.nn.model.layers[1].state
	Flux.reset!(B.nn)
	rng = B.nn.rng
	Bundle_value_gradient!(B, ϕ, sample)

	ϵs = B.nn.ϵs
	B.nn.rng = rng
	first_sample = true

	let v, vss, vns, grad
		for sample_idx in 1:samples
			#if B.back_prop_idx != []
			γS = 1
			if sample_idx > 1
				γS = 0.1
			end

			B.params.t = t
			#dev = zeros(size(B.ws[1]))
			fs = 1
			#ϵ = sample ? ϵs[1] : 0
			#if (1 in B.back_prop_idx) #&& first_run
			#	dev = B.params.t * B.ws[1]
			#	popat!(B.back_prop_idx,1)
			#end
			fs = 2

			Flux.reset!(B.nn)
			pred = z0# + dev

			ϵs = vcat([1], ϵs)

			if B.back_prop_idx != []
				if !telescopic
					v, grad = withgradient((m) -> -1 / normalization_factor * (#δ*ns_contribution(B,pred,[B.nn(B.features[i], B, sample ? ϵs[i] : 0) for i in eachindex(B.ws)],ϕ))
							+ϕ(reshape(
								z0 + sum([ws(m(B.features[i], B, sample ? ϵs[i] : 0), i, B) for i in eachindex(B.features)][B.back_prop_idx]), sizeInputSpace(ϕ)))), B.nn)
				else
					v, grad = withgradient((m) -> -1 / normalization_factor * (both_contributions(B, pred, [m(B.features[i], B, sample ? ϵs[i] : 0) for i in eachindex(B.ws)], ϕ, γ, δ)), B.nn)
				end
			else
				v, grad = withgradient((m) -> -1 / normalization_factor * (both_contributions(B, pred, [m(B.features[i], B, sample ? ϵs[i] : 0) for i in eachindex(B.ws)], ϕ, γ, δ)), B.nn)
			end
			Flux.update!(state, B.nn, grad[1])
			first_sample = false
		end
		B.nn.ϵs = []
		return -v
	end
end

"""
Compute the contributions of both serious steps and null steps.
"""
function both_contributions(B, pred, ts, ϕ, γ = 0.1, δ = 0.00001)
	wss = hcat(pred, [ws(ts[i], i, B) for i in eachindex(B.ws[1:end])]...)

	ns = [i for i in 2:size(wss, 2) if !(i in B.back_prop_idx .+ 1)]
	ss = vcat(1, B.back_prop_idx .+ 1)
	ss_ns = [maximum(vcat(1, [j for j in B.back_prop_idx if j <= i])) + 1 for i in ns]

	γs = [γ^i for i in 0:length(ss)-1]#:-1:1]
	γns = [γ^i for i in 0:length(ns)-1]#:-1:1]
	if γs != []
		δ = min(δ, minimum(γs))
	end
	loss_values_ns = [ϕ(reshape(sum(wss[:, i] for i in 1:ss_ns[j]) + wss[:, j], sizeInputSpace(ϕ))) for j in eachindex(ns)]
	loss_values_ss = [ϕ(reshape(sum(wss[:, i] for i in ss[1:j]), sizeInputSpace(ϕ))) for j in eachindex(ss)]
	return δ * sum(vcat(loss_values_ns, 0.0)) + sum(vcat(γs .* loss_values_ss, 0.0))
end

"""
Main function for the Bundle.
It maximize the function `ϕ` using the bundle method, with a previously initialized bundle `B`.
It has two additional input parameters:
- `t_strat`: the t-Strategy, by default it is the contant t-strategy (i.e. the regularization parameter for the Dual Master Problem is always keeped fixed).
- `unstable`: if `true` we always change the stabilization point using the last visited point, by default `false` (change it only if you know what you are doing).
"""
function bundle_execution(B::DeepBundle, ϕ::AbstractConcaveFunction; t_strat::abstract_t_strategy = constant_t_strategy(), unstable::Bool = false, force_maxIt::Bool = true, inference::Bool = false)
	times = Dict("times" => [], "trial point" => [], "ϕ" => [], "update β" => [], "SS/NS" => [], "update DQP" => [], "solve DQP" => [], "remove outdated" => [])
	ignore_derivatives() do
		t0 = time()
	end
	for epoch in 1:B.params.maxIt
		ignore_derivatives() do
			t1 = time()
		end
		# compute the new trial point
		z = trial_point(B)
		ignore_derivatives() do
			append!(times["trial point"], (time() - t1))
			t0 = time()
		end
		t0 = time()
		# compute the objective value and sub-gradient in the new trial-point
		obj, g = value_gradient(ϕ, z) # to optimize
		ignore_derivatives() do
			append!(times["ϕ"], (time() - t0))
			t0 = time()
			# update the bundle with the new information
			update_Bundle(B, z, g, obj)
		end

		# memorize the new regularization parameter
		t = B.params.t
		# memorize the current stabilization point
		s = B.s
		# update the regularization parameter and the stabilization point
		t_strategy(B, B.li, t_strat, unstable)

		ignore_derivatives() do
			append!(times["update β"], (time() - t0))

			t0 = time()
			#update the Dual Master Problem
			update_DQP!(B, t == B.params.t, s == B.s)
			append!(times["update DQP"], (time() - t0))

			t0 = time()
			#solve the Dual Master Problem
			solve_DQP(B) # to optimize
			append!(times["solve DQP"], (time() - t0))

			# Update the Dual Master problem solution and compute the new trial direction
			compute_direction(B)
		end
		B.w = ws()
		# remove outdated components, i.e. bundle components that are not used for many iterations
		ignore_derivatives() do
			t0 = time()
			if epoch >= B.params.remotionStep
				#remove_outdated(B)
			end
			append!(times["remove outdated"], (time() - t0))
			append!(B.ts, B.params.t)
		end
		# check the stopping criteria and if it is satisfied stop the execution
		# if force_maxIt is true than we force the algorithm to attain the maximum iterations
		# and no further stopping criteria is considered 
		ignore_derivatives() do
			if inference && !(force_maxIt) && stopping_criteria(B)
				println("Satisfied stopping criteria")
				return true, times
			end
		end
		ignore_derivatives() do
			push!(B.memorized["times"], time() - t1)
		end
	end
	ignore_derivatives() do
		times["times"] = B.memorized["times"]
	end
	if inference
		ignore_derivatives() do
			return false, times
		end
	else
		return both_contributions()


	end
end
