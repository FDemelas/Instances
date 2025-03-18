"""
Bundle Hyper-Parameters used for the tLearningBundle and VanillaBundle variants.
The majority are useful only for the VanillaBundle.
The Hyper-Parameters are:
- `maxIt`: maximum number of iterations, by default `1000`
- `remotionStep`: number of consecutive iterations that should be performed without using a certain component in order to fefmove it from the bundle. By default `50`.
- `t`: parameter used as regularization weight (in the quadratic part) of the Dual Master Probem objective function. By default `100`.
- `t_max`: maximum value that can be assumed to the parameter `t`. By default `10000`.
- `t_min`: minimum value that can be assumed to the parameter `t`. By default `1e-5`.
- `m1`: Parameter used in the condition Serious-Step / Null-Step. By default `0.01`. It should be ∈ [0,1[.
- `t_incr`: multiplicative factor used to increment `t` as `t*t_incr`.By default `1.1`. It should be greater than one.
- `t_decrement`: multiplicative factor used to decrement `t` as `t*t_decrement`.By default `0.9`. It should be greater than zero and lower than one.
- `minSS`: Minimum number of consecutive serious steps before considering an increment of `t`. By default `1`.
- `t_star`: Parameter for some t-strategies and the stopping criteria. It is used at the place of `t` as weight for the quadratic part in differemt contexts.By default `100000.0`.
- `ϵ`: Parameter for the stopping criteria. By default `1e-6`.
- `log`: A boolean value that say if memorize logs. By default `false`.
- `max_β_size`: Maximum bundle size. By default `200`.
- `tSPar2`: Parameter for different long term t-strategies.By default `0.01`.
"""
Base.@kwdef mutable struct BundleParameters
	maxIt::Int64=1000
	remotionStep::Int64=50
	t::Float64=100.0
	t_max::Int64=10000
	t_min::Float64=1e-5
	m1::Float64=0.01
	t_incr::Float64=1.1
	t_decrement::Float64=0.9
	minSS::Int64=1
	t_star::Float64=100000.0
	ϵ::Float64=1e-6
	log::Bool=false
	max_β_size::Int=200
	tSPar2::Float32=0.01
end

