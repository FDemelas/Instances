"""
Abstract type for every instance of the Multi Commodity Network Design Problem.
"""
abstract type abstractInstanceMCND <: abstractInstance end

"""
function origin(ins::instance, indexk)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `indexk`: the index of the commodity.

	Returns the origin node of the demand k
"""
origin(ins::abstractInstanceMCND, indexk) = ins.K[indexk][1]

"""
function destination(ins::instance, indexk)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `indexk`: the index of the commodity.

	Returns the destination node of the demand k
"""
destination(ins::abstractInstanceMCND, indexk) = ins.K[indexk][2]

"""
function volume(ins::instance, indexk)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `indexk`: the index of the commodity.

	Returns the volume of the demand k
"""
volume(ins::abstractInstanceMCND, indexk) = ins.K[indexk][3]

"""
function routing_cost(ins::instance, e, k)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `e`: the index of the edge.
	- `k`: the index of the commodity.

	Returns the routing cost 
"""
routing_cost(ins::abstractInstanceMCND, e, k) = ins.r[k, e]

"""
function tail(ins::instance, indexe)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `indexe`: the index of the edge.
	
	Returns the tail of arc ij (that is i)
"""
tail(ins::abstractInstanceMCND, indexe::Int) = ins.edges[indexe][1]

"""
function head(ins::instance, indexe)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `indexe`: the index of the edge.

	Returns the head of arc ij (that is j)
"""
head(ins::abstractInstanceMCND, indexe::Int) = ins.edges[indexe][2]

"""
function capacity(ins::instance, e)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `e`: the index of the edge.

	Returns the capacity of arc ij (that is j)
"""
capacity(ins::abstractInstanceMCND, e) = ins.c[e]

"""
function fixed_cost(ins::instance, e)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `e`: the index of the edge.

	Returns the fixed cost of arc e
"""
fixed_cost(ins::abstractInstanceMCND, e) = ins.f[e]

"""
function sizeK(ins::instance)

	# Arguments:
	- `ins`: a normalized instance structure.
	
	Returns the number of demands
"""
sizeK(ins::abstractInstanceMCND) = length(ins.K)

"""
function sizeV(ins::instance)

	# Arguments:
	- `ins`: a normalized instance structure.

	Returns the number of nodes
"""
sizeV(ins::abstractInstanceMCND) = ins.n

"""
function sizeE(ins::instance)

	# Arguments:
	- `ins`: a normalized instance structure.

	Returns the number of arcs
"""
sizeE(ins::abstractInstanceMCND) = length(ins.edges)

"""
function b(ins::instance, i, k)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `i`: the index of the node.
	- `k`: the index of the commodity.

	Given a certain node and a certain commodity, this function
	returns the volume of the demand for the commodity, if the node 
	is the origin, the inverse of the volume if the node is the destination
	and zero otherwise.
"""
function b(ins::abstractInstanceMCND, i, k)
	if i == origin(ins, k)
		return volume(ins, k)
	elseif i == destination(ins, k)
		return -volume(ins, k)
	else
		return 0
	end
end

"""
function isInKij(ins::instance, k, e)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `e`: the index of the edge.
	- `k`: the index of the commodity.

	Returns True if k is in Kij
"""
isInKij(ins::abstractInstanceMCND, k, e) = origin(ins, k) != head(ins, e) && destination(ins, k) != tail(ins, e)

"""
function outdegree(ins::instance, i)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `i`: the index of the node.

	Returns the outdegree of i
"""
outdegree(ins::abstractInstanceMCND, i::Integer) = count(ij -> tail(ins, ij) == i, collect(1:size(ins.edges, 1)))

"""
function indegree(ins::instance, i)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `i`: the index of the node.

	Returns the indegree of i
"""
indegree(ins::abstractInstanceMCND, i::Integer) = count(ij -> head(ins, ij) == i, collect(1:size(ins.edges, 1)))

"""
function outdegree_k(ins::instance, i, k)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `i`: the index of the node.
	- `k`: the index of the commodity.

	Returns the outdegree of i for a given commodity k
"""
outdegree_k(ins::abstractInstanceMCND, k::Integer, i::Integer) = destination(ins, k) == i ? 0 : count(ij -> tail(ins, ij) == i && isInKij(ins, k, ij), collect(1:size(ins.edges, 1)))

"""
function indegree_k(ins::instance, i, k)

	# Arguments:
	- `ins`: a normalized instance structure.
	- `i`: the index of the node.
	- `k`: the index of the commodity.

	Returns the indegree of i for a given commodity k
"""
indegree_k(ins::abstractInstanceMCND, k::Integer, i::Integer) = origin(ins, k) == i ? 0 : count(ij -> head(ins, ij) == i && isInKij(ins, k, ij), collect(1:size(ins.edges, 1)))

"""
function lengthLM(ins::instanceGA)
	
# Arguments:
- `ins`: instance object, should be a sub-type of instanceGA

returns the length of the Lagrangian multipliers for the instance `ins`.
"""
function lengthLM(ins::abstractInstanceMCND)
	return sizeK(ins) * sizeV(ins)
end

"""
function sizeLM(ins::instanceGA)
	
# Arguments:
- `ins`: instance object, should be a sub-type of instanceGA

returns the size of the Lagrangian multipliers for the instance `ins`.
"""
function sizeLM(ins::abstractInstanceMCND)
	return (Int64(sizeK(ins)), Int64(sizeV(ins)))
end
