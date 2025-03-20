"""
Abstact type for deviations used in model that predicts only `t`.
"""
abstract type AbstractDeviation
end

"""
Additive deviations consider `t_new = t_old + dev` where `dev` is predicted by the model.
"""
struct AdditiveDeviation <: AbstractDeviation end

(dt::AdditiveDeviation)(t,dev) = t .+ dev

Flux.@layer AdditiveDeviation


"""
Multiplicative deviations consider `t_new = t_old * dev` where `dev` is predicted by the model.
"""
struct MultiplicativeDeviation <: AbstractDeviation end

(dt::MultiplicativeDeviation)(t,dev) = t .* dev

Flux.@layer MultiplicativeDeviation


"""
Nothing deviations consider `t_new = dev` where `dev` is predicted by the model.
"""
struct NothingDeviation <: AbstractDeviation end

(dt::NothingDeviation)(_,dev) = dev
	
Flux.@layer NothingDeviation