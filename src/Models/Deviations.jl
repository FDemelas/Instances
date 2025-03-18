
abstract type AbstractDeviation
end

struct AdditiveDeviation <: AbstractDeviation end

struct MultiplicativeDeviation <: AbstractDeviation end

struct NothingDeviation <: AbstractDeviation end

(dt::AdditiveDeviation)(t,dev) = t .+ dev
(dt::MultiplicativeDeviation)(t,dev) = t .* dev
(dt::NothingDeviation)(_,dev) = dev


Flux.@layer AdditiveDeviation
	
Flux.@layer MultiplicativeDeviation
	
Flux.@layer NothingDeviation