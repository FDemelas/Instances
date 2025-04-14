"""
Structure for `RnnTModelSampleInside`.
The fields are:
- `model`: a neural network model;
- `rng`: a random number generator;
- `sample`: a boolean that say if use (or not) a sampling mechanism;
- `ϵs`: a vector to store the random component in sampling, i.e. a normal vector N(0,1);
- `deviation`: a structure of type `AbstractDeviation` that handle which type of deviation consider for the prediction of consecutives `t`, see deviations.jl for more details;
- `train_mode`: a boolean that say if the model is used in training (if `true`) or inference (if `false`).
"""
mutable struct RnnTModelSampleInside <: AbstractTModel
	model::Any
	rng::Any
	sample::Bool
	ϵs::AbstractVector
	deviation::AbstractDeviation
	train_mode::Bool
	RnnTModelSampleInside(model, rng, sample = false, es = [], deviation = NothingDeviation(),train_mode=true) = new(model, rng, sample, es, deviation,train_mode)
end

struct RnnTModelSampleInsidefactory <: AbstractTModelFactory end

"""
Creates the features vector for the current bundle.
"""
function create_features(B::DualBundle, _::RnnTModelSampleInside)
	# append the initial features and the ones of the current iterations
	φ=features_vector_i(B)
	# reshape of a proper dimension
	return reshape(φ,(length(φ),1))
end

"""
Returns the size of the features for the model. The input is the model itself.
"""
function size_features(_::RnnTModelSampleInside)
	return 34
end


"""
Returns the size of the features for the model. The input is the model factory.
"""
function size_features(_::RnnTModelSampleInsidefactory)
	return 34
end

"""
Forward computation for a model of type `RnnTModelSampleInside`. The Backward is performed by Automatic Differentiation.
"""
function (m::RnnTModelSampleInside)(ϕ, B, ϵ = randn(B.nn.rng, Float32, 1))
	# compute the output of the inner model using the features ϕ 
	f=m.model[1](ϕ)
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
	t = m.deviation(B.params.t, m.model[2](dev))
	# keep `t` into the good intervall
	t = min.(B.params.t_max, abs.(t) .+ B.params.t_min)
	return mean(t)
end

# Define the `RnnTModelSampleInside` models as Flux layer (to allow automatic differentiation)
Flux.@layer RnnTModelSampleInside

"""
Size of the output for a `RnnTModelSampleInside` created using a `RnnTModelSampleInsidefactory`.
"""
function size_output(_::RnnTModelSampleInsidefactory)
	return 1
end

"""
Function to create a `RnnTModelSampleInside` model given its hyper-parameters.
"""
function create_NN(lt::RnnTModelSampleInsidefactory,recurrent_layer=LSTM, h_decoder::Vector{Int} = [512, 256], h_act = softplus, h_representation::Int = 128, seed::Int = 1, norm::Bool = false)
	# normalize or not the input
	f_norm(x) = norm ? Flux.normalise(x) : identity(x)

	# random number generator and model parameter initializer
	rng = MersenneTwister(seed)
	init = Flux.truncated_normal(Flux.MersenneTwister(seed); mean = 0.0, std = 0.01)

	# encoder composed by a recurrent layer
	encoder_layer = recurrent_layer(size_features(lt) => 2*h_representation)

	# decoder composed by a Multi-Layer Perceptron
	i_decoder_layer = Dense(h_representation => h_decoder[1], h_act; init)
	h_decoder_layers = [Dense(h_decoder[i] => h_decoder[i+1], h_act; init) for i in 1:length(h_decoder)-1]
	o_decoder_layers = Dense(h_decoder[end] => size_output(lt),softplus; init)
	decoder = Chain(i_decoder_layer, h_decoder_layers..., o_decoder_layers)

	# the model just consider a chain of encoder and decoder
	model = Chain(encoder_layer, decoder)

	# return the proper `RnnTModelSampleInside`
	return RnnTModelSampleInside(model, rng, true)
end
