"""
BatchedSoftBundle structure.
It works as SoftBundle structure, but allows to handle multiple bundle execution in parallel in order to perform batching.
This variant does not computes the search direction as solution of the dual master problem, but it needs to use another model (generally based on neural networks).
"""
mutable struct BatchedSoftBundle <: AbstractSoftBundle
    G::Any
    z::Any
    α::Any
    s::Vector{Int64}
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
    li::Int64
    info::Dict
    maxIt::Int
    t::Vector{Float32}
    idxComp::AbstractArray
end

"""
Construct and initialize a BatchedSoftBundle structure.
Considering the functions contained in the vector `ϕs` and the staring points contained in the vector (of vectors) `z`.
`ϕs` and `z` must have the same number of components.
`lt` will be the factory of the model that we want consider for the prediction at the place of the Dual Master Problem.
The Bundle will be initialized to perform `maxIt` iterations (by default `10`).
"""
function initializeBundle(bt::BatchedSoftBundleFactory,ϕs::Vector{<: AbstractConcaveFunction}, z::Vector{<: AbstractArray},lt,maxIt::Int=10)
    B = BatchedSoftBundle([], [], [], [-1], [], [], [], Inf, [Inf], [Float32[]], lt, [], 0, 0, [],1,Dict(),maxIt,zeros(length(ϕs)),[])
    #B.nn.h_representation = h_representation(nn)
    batch_size = length(ϕs)
    sLM=[]
    gs,objs,idxComp = [], [], []
    tmp=0
    for (idx,ϕ) in enumerate(ϕs)
        lLM=prod(sizeInputSpace(ϕ))
        push!(idxComp,(tmp+1,tmp+lLM))
        tmp+=lLM 
        append!(sLM,prod(lLM))
        obj,g = value_gradient(ϕ,reshape(z[idx],sizeInputSpace(ϕ)))
        g = reshape(g, :)
        append!(objs,obj)
        append!(gs,g)
    end

    B.s = ones(batch_size)
    B.z = zeros(Float32, sum(sLM), B.maxIt + 1)
    B.z[:,1] = vcat([zi for zi in z]...)
    B.α = zeros(Float32, batch_size, B.maxIt + 1)  
    B.G = zeros(Float32, sum(sLM), B.maxIt + 1)
    B.G[:,1] = gs
    B.obj = zeros(Float32, batch_size, B.maxIt+1)
    B.obj[:,1] = objs
    B.objB = gs'gs
    B.w = gs
	B.θ = ones(batch_size,1)
    B.idxComp = idxComp
    B.li=1
    B.s=ones(length(idxComp))
    return B
end

"""
Reinitialize the Bundle before the execution.
This function allows to reuse the same Bundle multiple times without re-creating it.
It is particularly usefull to train models and it should be called before each Bundle execution (even without Backward pass).
"""
function reinitialize_Bundle!(B::BatchedSoftBundle, only_last::Bool = true)
	if only_last
		B.li = 1
		B.s = ones(length(B.idxComp))
	end
	B.G = Zygote.bufferfrom(device(hcat([B.G[:, 1:B.li], zeros(size(B.G, 1), B.maxIt + 1 - B.li)]...)))
	B.z = Zygote.bufferfrom(device(hcat([B.z[:, 1:B.li], zeros(size(B.z, 1), B.maxIt + 1 - B.li)]...)))
	B.α = Zygote.bufferfrom(cpu(hcat(B.α[:, 1:B.li], zeros(size(B.α, 1), B.maxIt + 1 - B.li))))
	B.obj = Zygote.bufferfrom(cpu(hcat(B.obj[:, 1:B.li], zeros(size(B.obj, 1), B.maxIt + 1 - B.li))))
	B.θ = ones(length(B.idxComp), 1)
	B.w = device(zeros(size(B.z, 1)))
end

"""
Function to perform the execution of the BatchedSoftBundle `B` to optimize (maximize) the functions contained in the vector `Φ`.
If will use the model `m` to provide the search direction and the step size at the place of the resolution of the Dual Master Problem and the t-strategy.
It is throughed to be differentiable using the Automatic Differentiation and so it can be used demanding for an automatic computation of the backward gradient.
"""
function bundle_execution(
	B::BatchedSoftBundle,
	ϕ::Vector{<:AbstractConcaveFunction},
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
	z_bar = Zygote.bufferfrom(Float32.(vcat([B.z[s:e, B.s[i]] for (i, (s, e)) in enumerate(B.idxComp)]...))),
	z_new = Zygote.bufferfrom(B.z[:, B.li]),)
	let xt, xγ, z_copy, LR_vec, Baseline, obj_new, obj_bar, g, t0, t1, times, maxIt,t, γs,θ
		ignore_derivatives() do
			times = Dict("init" => 0.0, "iters" => [], "model" => [], "distribution" => [], "features" => [], "stab_point" => [], "update_bundle" => [], "update_direction" => [], "update_point" => [], "lsp" => [])
			maxIt = B.maxIt
			t0 = time()
		end
		featG = function_features(B, B.lt)
		ignore_derivatives() do
			g = Zygote.bufferfrom(zeros(size(B.w)))
			obj_new = Zygote.bufferfrom(cpu(B.obj[B.li.*ones(Int64, length(B.s))]))
			obj_bar = obj_new
			γs = Zygote.bufferfrom([device(zeros(length(B.idxComp),it)) for it in 1:maxIt])
			θ = Zygote.bufferfrom([device(zeros(length(B.idxComp),it)) for it in 1:maxIt])
			B.objB = 0
			times["init"] = time() - t0
		end
		for it in 1:maxIt
			ignore_derivatives() do
				t0 = time()
			end

			ignore_derivatives() do
				xt, xγ = device(create_features(B.lt, B; auxiliary = featG))
				if size(xt)[1]==length(xt)
					xt=reshape(xt,(length(xt),1))
				end
				if size(xγ)[1]==length(xγ)
					xγ=reshape(xγ,(length(xγ),1))
				end
			end
			ignore_derivatives() do
				append!(times["features"], time() - t0)
			end

			ignore_derivatives() do
				t1 = time()
			end
			t, γs[it] = m(xt, xγ)
			B.t=device(reshape(t,:))
			
			min_idx = Int64(max(1, minimum(B.li) - max_inst))
			ignore_derivatives() do
				append!(times["model"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end
			
			θ[it] = distribution_function(γs[it][:, min_idx:B.li]; dims = 2)
			
			ignore_derivatives() do
				B.θ=θ[it]
				append!(times["distribution"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end
			B.w = vcat([B.G[s:e, min_idx:B.li] * θ[it][i, :] for (i, (s, e)) in enumerate(B.idxComp)]...)
			ignore_derivatives() do
				append!(times["update_direction"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end

			z_new[:] = z_bar[:] + vcat([B.t[i] * B.w[s:e] for (i, (s, e)) in enumerate(B.idxComp)]...)
			ignore_derivatives() do
				append!(times["update_point"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end
			for (i, (s, e)) in enumerate(B.idxComp)
				v, g_tmp = value_gradient(ϕ[i], z_new[s:e])
				g[s:e] = cpu(g_tmp)
				obj_new[i] = v
			end
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
					for (i, (s, e)) in enumerate(B.idxComp)
						z_bar[s:e] = (obj_new[i] > obj_bar[i] ? (z_new[s:e]) : (z_bar[s:e]))
						obj_bar[i] = (obj_new[i] > obj_bar[i] ? obj_new[i] : obj_bar[i])
					end
				else
					for (i, (s, e)) in enumerate(B.idxComp)
						sm = softmax([obj_new[i], obj_bar[i]])
						obj_bar[i] = sm' * [obj_new[i], obj_bar[i]]
						z_bar[s:e] = device(sm' * cpu.([z_new[s:e], z_bar[s:e]]))
					end
				end
				B.s = vcat([obj_new[i] > obj_bar[i] ? B.li : B.s[i] for i in 1:length(B.s)]...)
			end
			ignore_derivatives() do
				append!(times["stab_point"], time() - t1)
			end

			ignore_derivatives() do
				t1 = time()
			end
			B.G[:, B.li] = device(g[:])
			B.z[:, B.li] = z_new[:]
			B.obj[:, B.li] = obj_new[:]

			for (i, (s, e)) in enumerate(B.idxComp)
				B.α[i, :] = (B.obj[i, :] .- obj_bar[i, :]) .- cpu(sum(B.G[s:e, :] .* (B.z[s:e, :] .- z_bar[s:e]); dims = 1))'
			end

			ignore_derivatives() do
				append!(times["update_bundle"], time() - t1)
				append!(times["iters"], time() - t0)
			end
		end

		if inference
			return mean(ϕ[iϕ](reshape(z_bar[s:e], sizeInputSpace(ϕ[iϕ]))) for (iϕ, (s, e)) in enumerate(B.idxComp)), times
		else
			vγ = (γ > 0 ? mean(γ * mean([γ^(B.maxIt + 1 - i) for i in (B.maxIt+1):-1:1] .* [ϕ[iϕ](reshape(z[s:e], sizeInputSpace(ϕ[iϕ]))) for z in eachcol(B.z[:, 1:B.maxIt+1])]) for (iϕ, (s, e)) in enumerate(B.idxComp)) : 0)
			vλ = mean((1 - λ) * ϕ[iϕ](reshape(z_bar[s:e], sizeInputSpace(ϕ[iϕ]))) + (λ) * ϕ[iϕ](reshape(z_new[s:e], sizeInputSpace(ϕ[iϕ]))) for (iϕ, (s, e)) in enumerate(B.idxComp))

			return vγ + vλ 
		end
	end
end