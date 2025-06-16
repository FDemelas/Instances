"""
struct gpuMCNDinstance <: abstractInstanceMCND

# Fields:
- `n` : nombre de sommets
- `edges`` : table of arcs: [(tₑ,hₑ)]ₑ
- `K` : table of commodities: [(oᵏ, dᵏ, qᵏ)]ₖ
- `f` : fixed costs
- `r` : routing costs
- `c` : capacities of arcs 
- `...`: other technical parameters that are used for the GPU-computations of the Lagrangian Sub-Problem and are computed just once.
		It is not mandatory to provide these parameters to the constructor as they can be automatically constructed starting from the  ones shared with the cpu version of the instance. 
"""
struct gpuMCNDinstance <: abstractInstanceMCND
	n::Int16
	edges::Vector{Tuple{Int16, Int16}}
	K::Vector{Tuple{Int16, Int16, Int64}}
	f::Vector{Int64}
	r::Matrix{Int64}
	c::Vector{Int64}
	MHT::Any
	M01::Any
	q::Any
	eIDX::Any
	E::Any
	B::Any
	gpuMCNDinstance(n, edges, K, f, r, c) = (
		ins = cpuInstanceMCND(n, edges, K, f, r, c);
		#compute the adjacency matrix |nodes| x |arcs|
		MHT = Float32[(tail(ins, ij) == i) ? 1 : (head(ins, ij) == i ? -1 : 0) for i in 1:sizeV(ins), ij in 1:sizeE(ins)] |> gpu;
		#compute the boolean matrix |commodities| x |arcs| that say if one arcs can be used by a certain commodity
		M01 = Float32[isInKij(ins, k, ij) for k in 1:sizeK(ins), ij in 1:sizeE(ins)] |> gpu;
		#compute the potentials of the demands
		q = Float32[ins.K[i][3] for i in 1:sizeK(ins)] |> gpu;
		# compute a matrix of size |commodities| x |nodes| that say if a node is the origin/destination
		# for a certain commodity and in that case give (respectively) the value plus or minus the volume
		# of the commodity
		B = zeros(Float32, sizeK(ins), sizeV(ins));
		for k in 1:sizeK(ins)
			B[k, origin(ins, k)] = volume(ins, k)
			B[k, destination(ins, k)] = -volume(ins, k)
		end;
		B = B |> gpu;
		E = zeros(Float32, sizeV(ins), sizeE(ins));
		for e in 1:sizeE(ins)
			E[tail(ins, e), e] = 1
			E[head(ins, e), e] = -1
		end;
		E = E' |> gpu;
		# construct the edge position
		# will be used after to re-sort the solution in the correct way
		eIDX = gpu(repeat(Vector(0:sizeK(ins):sizeK(ins)*(sizeE(ins)-1))', outer = [sizeK(ins), 1]));
		new(n, edges, K, f, r, c, MHT, M01, q, eIDX, E, B)
	)
end

struct gpuMCNDinstanceFactory <: MCNDinstanceFactory end

"""
create_data_object(::gpuMCNDinstanceFactory, n, edges, K, f, r, c)

# Arguments:
- ::gpuMCNDinstanceFactory: The standard instance factory formulation.
- `n`: The number of node in the graph that defines the instance.
- `edges`: An array of tuble (size ℤ²ˣᴱ), for each edge 'e' we have a tuple defined as (start(e),end(e))
		with the starting end ending node for that arc.
- `K`: An array of triplet (with size ℤᵏˣ³), for each commodity k the associated component is (oᵏ,dᵏ,qᵏ), i.e. the origin, destination and volume of the commodity.
- `f`: The vector of fixed costs ( ∈ ℝᴱ)
- `r`: The vector of routing costs ( ∈ ℝᴷˣᴱ)
- `c`: The vector of capacities ( ∈ ℝᴱ)

It create an instance structure ( for an instance in the standard formulation)
with the provided inputs data. 
"""
create_data_object(::gpuMCNDinstanceFactory, n, edges, K, f, r, c) = gpuMCNDinstance(n, edges, K, f, r, c)
