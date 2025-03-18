abstract type abstract_t_strategy end

abstract type abstract_middle_term_t_strategy end

struct constant_t_strategy <: abstract_t_strategy end

struct nn_t_strategy <: abstract_t_strategy end

struct heuristic_t_strategy_1 <: abstract_middle_term_t_strategy end

struct heuristic_t_strategy_2 <: abstract_middle_term_t_strategy end

struct soft_long_term_t_strategy <: abstract_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

struct hard_long_term_t_strategy <: abstract_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

struct balancing_long_term_t_strategy <: abstract_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

struct medium_term_t_strategy <: abstract_t_strategy end

function increment_t(B::AbstractBundle,ts::constant_t_strategy)
end

function decrement_t(B::AbstractBundle,ts::constant_t_strategy)
end    

function increment_t(B::AbstractBundle,ts::heuristic_t_strategy_1)
    B.params.t = max(B.params.t,min(B.params.t * B.params.t_incr,B.params.t_max))
end

function decrement_t(B::AbstractBundle,ts::heuristic_t_strategy_1)
    B.params.t = min(B.params.t,max(B.params.t * B.params.t_decrement,B.params.t_min))
end    

function increment_t(B::AbstractBundle,ts::heuristic_t_strategy_2)
    B.params.t = max(B.params.t, min(B.params.t * B.params.t_incr,B.params.t_max, 2*B.params.t* B.ϵ * ( B.ϵ-(B.obj[end]-objS(B))) ))
end

function decrement_t(B::AbstractBundle,ts::heuristic_t_strategy_2)
    if B.α[end] > 2*B.w'B.w
        B.params.t = min(B.params.t, max(B.params.t * B.params.t_decrement,B.params.t_min,  ( B.α[end] +(B.obj[end]-objS(B))) /(2*B.α[end])))
    end
end    

function increment_t(B::AbstractBundle,ts::soft_long_term_t_strategy)
    increment_t(B,ts.middle_term_strategy)
end

function decrement_t(B::AbstractBundle,ts::soft_long_term_t_strategy)
    if B.vStar > B.params.tSPar2 * B.ϵ
        decrement_t(B,ts.middle_term_strategy)
    end
end    

function increment_t(B::AbstractBundle,ts::hard_long_term_t_strategy)
    increment_t(B,ts.middle_term_strategy)
end

function decrement_t(B::AbstractBundle,ts::hard_long_term_t_strategy)
    if B.vStar > B.params.tSPar2 * B.ϵ
    #    decrement_t(B,ts.middle_term_strategy)
    else
        increment_t(B,ts.middle_term_strategy)
    end
end    

function increment_t(B::AbstractBundle,ts::balancing_long_term_t_strategy)
    if B.params.t_star * B.w' * B.w/2 > B.params.tSPar2 *B.α[1:length(B.θ)]' * B.θ
        increment_t(B,ts.middle_term_strategy)
    end
end

function decrement_t(B::AbstractBundle,ts::balancing_long_term_t_strategy)
    if B.params.tSPar2 * B.params.t_star * B.w' * B.w/2 <  B.α[1:length(B.θ)]' * B.θ
        decrement_t(B,ts.middle_term_strategy)
    end
end    

function increment_t(B::AbstractBundle,ts::nn_t_strategy)
    ϕ=create_features(B,B.nn)
    ϕ=device(ϕ)
    B.params.t = cpu(B.nn( ϕ, B ))
end


function decrement_t(B::AbstractBundle,ts::nn_t_strategy)
    ϕ=create_features(B,B.nn)
    ϕ=device(ϕ)
    B.params.t = cpu(B.nn( ϕ, B ))
end