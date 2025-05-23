"""
Abstract type for modelize the 't-strategies', i.e. the heuristics to handle increments and decrements of the parameter, here called `t`, that works as regularization term for the quadratic component of the loss in the Dual Master Problem and also as step size.
"""
abstract type abstract_t_strategy end

"""
Abstract type that modelize the long term component in the t-strategies.
"""
abstract type abstract_long_term_t_strategy <: abstract_t_strategy end

"""
Abstract type that modelize the middle term component in the t-strategies.
"""
abstract type abstract_middle_term_t_strategy end

"""
Constant t-strategies. No increment or decrement. `t` in this case is just keeped fixed.
"""
struct constant_t_strategy <: abstract_t_strategy end

"""
This is a 't-strategy' that simply use a neural network to provide the new `t` parameter.
In this case the neural network compose the whole strategy, i.e. we does not have the sub-division in short/middle/long term t-strategies.
"""
struct nn_t_strategy <: abstract_t_strategy end

"""
Structure that define a middle term t-strategy.
It simply keep `t` in between a certain intervall and increment/decrement it using fixed factors. 
See the functions `increment_t` and `decrement_t` for more informations.
"""
struct heuristic_t_strategy_1 <: abstract_middle_term_t_strategy end

"""
Soft Long term t-strategy.
Allows decrements only if a certain condition is satisfied.
See the functions `increment_t` and `decrement_t` for more informations.
"""
struct soft_long_term_t_strategy <: abstract_long_term_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

"""
Hard Long term t-strategy.
Similar to the Soft Long term t-strategy. The difference is that it incentives increments when the Soft inhibits decrements (when a decrement should be done).
See the functions `increment_t` and `decrement_t` for more informations.
"""
struct hard_long_term_t_strategy <: abstract_long_term_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

"""
Balancing Long term t-strategy.
It tries to keep the quadratic part and the linear part to the "same" magnitude. 
See the functions `increment_t` and `decrement_t` for more informations.
"""
struct balancing_long_term_t_strategy <: abstract_long_term_t_strategy 
    middle_term_strategy::abstract_middle_term_t_strategy
end

"""
The constant just keep `t` constant and so this function do nothing.
"""
function increment_t(B::AbstractBundle,ts::constant_t_strategy)
end

"""
The constant just keep `t` constant and so this function do nothing.
"""
function decrement_t(B::AbstractBundle,ts::constant_t_strategy)
end    

"""
The middle term t-strategy `heuristic_t_strategy_1`, when an increment is demanded allows it only if we does not depass the upper bound provided by `B.params.t_max`.
In this case it increments `t` as `t * B.params.t_incr` with `B.params.t_incr > 1`.
"""
function increment_t(B::AbstractBundle,ts::heuristic_t_strategy_1)
    B.params.t = max(B.params.t,min(B.params.t * B.params.t_incr,B.params.t_max))
end

"""
The middle term t-strategy `heuristic_t_strategy_1`, when a decrement is demanded allows it only if we does not depass the lower bound provided by `B.params.t_min`.
In this case it increments `t` as `t * B.params.t_decrement` with `0 < B.params.t_decrement < 1`.
"""
function decrement_t(B::AbstractBundle,ts::heuristic_t_strategy_1)
    B.params.t = min(B.params.t,max(B.params.t * B.params.t_decrement,B.params.t_min))
end    

"""
The Soft Long-term t-strategy just call the Middle-term t-strategy for increments.
"""
function increment_t(B::AbstractBundle,ts::soft_long_term_t_strategy)
    increment_t(B,ts.middle_term_strategy)
end


"""
The Soft Long-term t-strategy allows decrements only if `B.vStar > B.params.tSPar2 * B.ϵ`.
If this condition is satisfied it just call the Middle-term t-strategy for decrements.
"""
function decrement_t(B::AbstractBundle,ts::soft_long_term_t_strategy)
    if B.vStar > B.params.tSPar2 * B.ϵ
        decrement_t(B,ts.middle_term_strategy)
    end
end    

"""
The Soft Long-term t-strategy just call the Middle-term t-strategy for increments.
"""
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