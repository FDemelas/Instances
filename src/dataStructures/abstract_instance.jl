"""
Abstract type for the instances.
"""
abstract type abstractInstance end

"""
Abstract type for the instance factories.
"""
abstract type abstractInstanceFactory end

"""
objective_coefficient_type(::abstractInstanceFactory)

Returns the type of the objective for an instance in the standard formulation, that is Int16. 
"""
objective_coefficient_type(::abstractInstanceFactory) = Int16