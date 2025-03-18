mutable struct RnnTModel <: AbstractTModel
	model::Any
	rng::Any
	sample::Bool
	ϵs::AbstractVector
	deviation::AbstractDeviation
	train_mode::Bool
	RnnTModel(model, rng, sample = false, es = [], deviation = NothingDeviation(),train_mode=true) = new(model, rng, sample, es, deviation,train_mode)
end

struct RnnTModelfactory <: AbstractTModelFactory end



function create_features(B::DualBundle, _::RnnTModel)
	φ=vcat(B.ϕ0, features_vector_i(B))
	return reshape(φ,(length(φ),1))
end

function features_vector_i(B::DualBundle)
	g = gS(B)
	α = αS(B)
	αl = B.α[B.li]
	w = B.w
	z = zS(B)
	realObj = B.obj[B.li]
	sObj = objS(B)
	approxObj = B.objB
	αs = B.α[1:min(length(B.θ), B.size)]
	ϕ = Float32[B.params.t,
		B.size/B.params.maxIt,
		B.CSS/B.params.maxIt,
		B.CNS/B.params.maxIt,
		B.s==B.li,
		realObj<sObj,
		realObj<sObj,
		sqrt(w' * w)/2<sum(αs' * B.θ),
		B.params.t*sqrt(w' * w)/2<sum(αs' * B.θ),
		sign(realObj),
		sign(sObj),
		sign(approxObj),
		log(1+abs(realObj)),
		log(1+abs(sObj)),
		log(1+abs(approxObj)),
		sObj-realObj <= approxObj,
		log(1+α),
		log(1+αl),
		log(1+maximum(B.α)),
		log(1+B.z[:, B.li]' * B.z[:, B.li]),
		log(1+z' * z), 
		log(1+g' * g),
		log(1+B.G[:, B.li]' * B.G[:, B.li]),
		log(1+w' * w),
		log(1+B.params.t * w' * w), 
		log(1+sum(αs' * B.θ)),
		B.params.t*B.params.t_incr<B.params.t_max]
	return ϕ
end

function size_features(lt::RnnTModel)
	return 27+20#20
end


function size_features(lt::RnnTModelfactory)
	return 27+20#20
end


function (m::RnnTModel)(ϕ, B, ϵ = randn(B.nn.rng, Float32, 1))
	f=m.model(ϕ)
	μ, σ2 = Flux.MLUtils.chunk(f, 2, dims = 1)
	if m.sample
		σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
		σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
		σ2 = exp.(σ2)
		sigma = sqrt.(σ2) / 100
		if m.train_mode
			ignore_derivatives() do
				push!(m.ϵs, ϵ)
			end
		end
		dev = (μ .+ ϵ .* sigma)
	else
		dev = μ
	end
	t = m.deviation(B.params.t, dev)
	t = min.(B.params.t_max, abs.(t) .+ B.params.t_min)
	return mean(t)
end

Flux.@layer RnnTModel


function size_output(_::RnnTModelfactory)
	return 2
end

function create_NN(lt::RnnTModelfactory,recurrent_layer=GRUv3, h_decoder::Vector{Int} = [512, 128], h_act = tanh, h_representation::Int = 64, seed::Int = 1, norm::Bool = true)
	f_norm(x) = Flux.normalise(x)
	rng = MersenneTwister(seed)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.01)
	encoder_layer = recurrent_layer(size_features(lt) => h_representation)

	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => size_output(lt); init)
	decoder = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	model = Chain(encoder_layer, decoder)
	return RnnTModel(model, rng, false)
end
