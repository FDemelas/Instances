"""
Structure for `RnnTModel`.
The fields are:
- `model`: a neural network model;
- `rng`: a random number generator;
- `sample`: a boolean that say if use (or not) a sampling mechanism;
- `ϵs`: a vector to store the random component in sampling, i.e. a normal vector N(0,1);
- `deviation`: a structure of type `AbstractDeviation` that handle which type of deviation consider for the prediction of consecutives `t`, see deviations.jl for more details;
- `train_mode`: a boolean that say if the model is used in training (if `true`) or inference (if `false`).
"""
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

"""
Creates the features vector for the current bundle.
"""
function create_features(B::DualBundle, _::RnnTModel)
	# append the initial features and the ones of the current iterations
	φ=vcat(B.ϕ0, features_vector_i(B))
	# reshape of a proper dimension
	return reshape(φ,(length(φ),1))
end

"""
Creates the features for the current iteration.
"""
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

"""
Returns the size of the features for the model. The input is the model itself.
"""
function size_features(_::RnnTModel)
	return 47
end


"""
Returns the size of the features for the model. The input is the model factory.
"""
function size_features(_::RnnTModelfactory)
	return 47
end

"""
Forward computation for a model of type `RnnTModel`. The Backward is performed by Automatic Differentiation.
"""
function (m::RnnTModel)(ϕ, B, ϵ = randn(B.nn.rng, Float32, 1))
	# compute the output of the inner model using the features ϕ 
	f=m.model(ϕ)
	# the inner model provides the mean and variance of a gaussian distribution.
	μ, σ2 = Flux.MLUtils.chunk(f, 2, dims = 1)
	if m.sample
		# if we use a reparametrization trick, then compute the variance
		σ2 = 2.0f0 .- softplus.(2.0f0 .- σ2)
		σ2 = -6.0f0 .+ softplus.(σ2 .+ 6.0f0)
		σ2 = exp.(σ2)
		sigma = sqrt.(σ2) / 100
		if m.train_mode
			# during training we need to save the random part `ϵ` ~ N(0,1)
			ignore_derivatives() do
				push!(m.ϵs, ϵ)
			end
		end
		# sample the deviation using a reparametrization trick: given X~N(μ,sigma) and Y~N(0,1) we have X = μ + sigma ⋅ Y
		dev = (μ .+ ϵ .* sigma)
	else
		# if no sample mechanism is used just take the mean
		dev = μ
	end
	# the parameter `t` is then obtained by applying a deviation to the sampled vector
	t = m.deviation(B.params.t, dev)
	# keep `t` into the good intervall
	t = min.(B.params.t_max, abs.(t) .+ B.params.t_min)
	return mean(t)
end

# Define the `RnnTModel` models as Flux layer (to allow automatic differentiation)
Flux.@layer RnnTModel

"""
Size of the output for a `RnnTModel` created using a `RnnTModelfactory`.
"""
function size_output(_::RnnTModelfactory)
	return 2
end

"""
Function to create a `RnnTModel` model given its hyper-parameters.
"""
function create_NN(lt::RnnTModelfactory,recurrent_layer=GRUv3, h_decoder::Vector{Int} = [512, 256], h_act = tanh, h_representation::Int = 128, seed::Int = 1, norm::Bool = false)
	# normalize or not the input
	f_norm(x) = norm ? Flux.normalise(x) : identity(x)

	# random number generator and model parameter initializer
	rng = MersenneTwister(seed)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.01)

	# encoder composed by a recurrent layer
	encoder_layer = recurrent_layer(size_features(lt) => h_representation)

	# decoder composed by a Multi-Layer Perceptron
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => size_output(lt); init)
	decoder = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# the model just consider a chain of encoder and decoder
	model = Chain(encoder_layer, decoder)

	# return the proper `RnnTModel`
	return RnnTModel(model, rng, false)
end
