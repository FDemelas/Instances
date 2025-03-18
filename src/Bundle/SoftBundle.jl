"""
SoftBundle structure.
This variant does not computes the search direction as solution of the dual master problem, but it needs to use another model (generally based on neural networks).
"""
mutable struct SoftBundle <: AbstractSoftBundle
	G::Any
	z::Any
	α::Any
	s::Int64
	w::Any
	θ::Any
	γ::Any
	objB::Float32
	obj::Any
	cumulative_θ::Any
	lt::AbstractModelFactory
	back_prop_idx::Any
	CSS::Int64
	CNS::Int64
	ϕ0::Any
	li::Int
	info::Dict
	maxIt::Int
	t::Real
end

"""
Construct and initialize a SoftBundle structure.
Considering to optimize (maximize) the function `ϕ` and the staring point `z`.
`ϕs` and `z` must have the same number of components.
`lt` will be the factory of the model that we want consider for the prediction at the place of the Dual Master Problem.
The Bundle will be initialized to perform `maxIt` iterations (by default `10`).
"""

function initializeBundle(bt::SoftBundleFactory, ϕ::AbstractConcaveFunction, z::AbstractArray, lt::AbstractModelFactory, maxIt::Int = 10)
	B = SoftBundle([], [], [], -1, [], [], [], Inf, [Inf], [Float32[]], lt, [], 0, 0, [], 1, Dict(), maxIt, 0.0)
	#B.nn.h_representation = h_representation(nn)
	obj, g = value_gradient(ϕ, z)
	g = reshape(g, :)
	B.s = 1
	z = reshape(z, :)
	B.z = device(zeros(Float32, length(g), B.maxIt + 1))
	B.α = device(zeros(Float32, B.maxIt + 1))
	B.G = hcat(g, device(zeros(Float32, length(g), B.maxIt)))

	B.obj = vcat(obj, device(zeros(B.maxIt)))
	B.objB = g' * g
	B.w = g
	B.θ = zeros(B.maxIt + 1)
	B.θ[1] = 1
	B.ϕ0 = initializeϕ0(lt, ϕ)
	return B
end

"""
Reinitialize the Bundle before the execution.
This function allows to reuse the same Bundle multiple times without re-creating it.
It is particularly usefull to train models and it should be called before each Bundle execution (even without Backward pass).
"""
function reinitialize_Bundle!(B::SoftBundle, only_last::Bool = true)
	if only_last
		B.li = 1
		B.s = 1
	end
	B.obj = Zygote.bufferfrom(cpu(Float32.(B.obj)))
	B.G = Zygote.bufferfrom(device(Float32.(hcat(B.G[:, 1], zeros(size(B.G, 1), B.maxIt)))))
	B.z = Zygote.bufferfrom(device(Float32.(hcat(B.z[:, 1], zeros(size(B.z, 1), B.maxIt)))))
	B.α = cpu(B.obj[:] .- B.obj[B.s]) .- cpu(sum(B.G[:, :] .* (B.z[:, :] .- B.z[:,B.s]); dims = 1))'
	B.α = Zygote.bufferfrom(cpu(B.α))
	B.θ = device(ones(B.li) ./ B.li)
	B.w = device(B.w)
end

"""
Function to perform the execution of the BatchedSoftBundle `B` to optimize (maximize) the functions contained in the vector `Φ`.
If will use the model `m` to provide the search direction and the step size at the place of the resolution of the Dual Master Problem and the t-strategy.
It is throughed to be differentiable using the Automatic Differentiation and so it can be used demanding for an automatic computation of the backward gradient.
"""
function bundle_execution(
	B::SoftBundle,
	ϕ::AbstractConcaveFunction,
	m::AbstractModel;
	soft_updates = false,
	λ = 0.0,
	γ = 0.0,
	δ = 0.0,
	distribution_function = softmax,
	verbose::Int = 0,
	max_inst = Inf,
	metalearning = false,
	unstable = false,
	inference = false,
	z_bar = Zygote.bufferfrom(Float32.(vcat(B.z[:, B.s]))),
	z_new = Zygote.bufferfrom(B.z[:, B.li]))
	let xt, xγ, z_copy, LR_vec, Baseline, obj_new, obj_bar, g, t0, t1, times, maxIt, t, γs, θ
		ignore_derivatives() do
			times = Dict("init" => 0.0, "iters" => [], "model" => [], "distribution" => [], "features" => [], "stab_point" => [], "update_bundle" => [], "update_direction" => [], "update_point" => [], "lsp" => [])
			maxIt = B.maxIt
			t0 = time()
		end
		featG = function_features(B, B.lt)
		ignore_derivatives() do
			g = Zygote.bufferfrom(zeros(size(B.w)))
			obj_new = Float32[cpu(B.obj[B.li])]
			obj_bar = Float32[cpu(B.obj[B.s])]
			γs = [device(zeros(1, it)) for it in 1:maxIt]
			θ = [device(zeros(1, it)) for it in 1:maxIt]
			B.objB = 0
			times["init"] = time() - t0
		end
		for it in 1:maxIt
			ignore_derivatives() do
				t0 = time()
			end

			ignore_derivatives() do
				xt, xγ = device(create_features(B.lt, B; auxiliary = featG))
				if size(xt)[1] == length(xt)
					xt = reshape(xt, (length(xt), 1))
				end
				if size(xγ)[1] == length(xγ)
					xγ = reshape(xγ, (length(xγ), 1))
				end
			end
			ignore_derivatives() do
				append!(times["features"], time() - t0)
			end

			ignore_derivatives() do
				t1 = time()
			end


			res = m(xt, xγ)

			t = copy(res[1])
			B.t = sum(t)
			γs[it] = copy(reshape(res[2], size(γs[it])))  # Vérifie si res est un tuple
			min_idx = Int64(max(1, minimum(B.li) - max_inst))
			ignore_derivatives() do
				append!(times["model"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			θ[it] = distribution_function(γs[it][:, min_idx:B.li]; dims = 2)

			ignore_derivatives() do
				B.θ = θ[it]
				append!(times["distribution"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			B.w = vcat(B.G[:, min_idx:B.li] * θ[it][1, :])

			ignore_derivatives() do
				append!(times["update_direction"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			z_new[:] = z_bar[:] + vcat(B.t .* B.w[:])

			ignore_derivatives() do
				append!(times["update_point"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			v, g_tmp = value_gradient(ϕ, z_new[:])
			g[:] = cpu(g_tmp)
			obj_new = v

			ignore_derivatives() do
				append!(times["lsp"], time() - t1)
			end

			B.li += 1
			ignore_derivatives() do
				t1 = time()
			end
			if unstable
				z_bar = z_new
				obj_bar = obj_new
				B.s = B.li .* ones(Int64, length(B.s))
			else
				if !soft_updates

					z_bar[:] = (obj_new > obj_bar ? (z_new[:]) : (z_bar[:]))
					obj_bar = (obj_new > obj_bar ? obj_new[:] : obj_bar[:])
				else

					sm = softmax([obj_new, obj_bar])
					obj_bar = sm' * [obj_new, obj_bar]
					z_bar[:] = device(sm' * cpu.([z_new[:], z_bar[:]]))

				end
				B.s = obj_new > obj_bar ? B.li : B.s
			end
			ignore_derivatives() do
				append!(times["stab_point"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end
			B.G[:, B.li] = device(g[:])
			B.z[:, B.li] = z_new[:]
			B.obj[B.li]  = obj_new

			B.α[1:B.li] = (B.obj[1:B.li] .- obj_bar) .- cpu(sum(B.G[:, 1:B.li] .* (B.z[:, 1:B.li] .- z_bar[:]); dims = 1))'


			ignore_derivatives() do
				append!(times["update_bundle"], time() - t1)
				append!(times["iters"], time() - t0)
			end
		end

		if inference
			return ϕ[iϕ](reshape(z_bar, sizeInputSpace(ϕ))), times
		else
			vγ = (γ > 0 ? γ * mean([γ^(B.maxIt + 1 - i) for i in (B.maxIt+1):-1:1] .* [ϕ(z) for z in eachcol(B.z[:, 1:B.maxIt+1])]) : 0)
			vλ = (1 - λ) * ϕ(reshape(z_bar[:], sizeInputSpace(ϕ))) + (λ) * ϕ(reshape(z_new[:], sizeInputSpace(ϕ)))
			return vγ + vλ
		end
	end
end