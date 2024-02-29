"""
struct cpuInstanceMCND <: abstractInstance

# Fields:
- `n` : nombre de sommets
- `edges`` : table of arcs: [(tₑ,hₑ)]ₑ
- `K` : table of commodities: [(oᵏ, dᵏ, qᵏ)]ₖ
- `f` : fixed costs
- `r` : routing costs
- `c` : capacities of arcs 
"""
struct cpuInstanceMCND <: abstractInstanceMCND
	n::Int64
	edges::Vector{Tuple{Int16, Int16}}
	K::Vector{Tuple{Int16, Int16, Int64}}
	f::Vector{Int64}
	r::Matrix{Int64}
	c::Vector{Int64}
end

"""
Abstract type for the Multi Commodity Network Design Problem.
"""
abstract type MCNDinstanceFactory <: abstractInstanceFactory end

"""
Structure for the cpu instance of the Multi Commodity Network Design Problem.
"""
struct cpuMCNDinstanceFactory <: MCNDinstanceFactory end


"""
function create_data_object(::cpuMCNDinstanceFactory, n, edges, K, f, r, c)

# Arguments:
- ::cpuMCNDinstanceFactory: The standard instance factory formulation.
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
create_data_object(::cpuMCNDinstanceFactory, n, edges, K, f, r, c) = cpuInstanceMCND(n, edges, K, f, r, c)


"""
function read_dat(path::String,factory::cpuMCNDinstanceFactory)

#Arguments:
-`path`: a String that contains the path to the data file of the instance
-`factory`: a cpuMCNDinstanceFactory used only to construct an instance of the proper type (cpuInstanceMCND)

Given the path and the factory of the proper type, this function reads the instance information from the file in the provided path and construct an instance of the Bin Packing problem.
"""
function read_dat(path::String, factory::cpuMCNDinstanceFactory)
	f = open(path, "r")
	sep = " "#"\t"
	line = split(readline(f), sep; keepempty = false)

	n = parse(Int64, line[1])
	E = parse(Int64, line[2])
	K = parse(Int64, line[3])
	commodities = Tuple{Int16, Int16, Int64}[]
	edges = Tuple{Int16, Int16}[]

	fc = zeros(E)
	c = zeros(E)
	r = zeros(K, E)

	for e in 1:E
		line = split(readline(f), sep; keepempty = false)
		push!(edges, (parse(Int16, line[2]), parse(Int16, line[1])))
		fc[e] = parse(Int64, line[3])
		c[e] = parse(Int64, line[4])
		for k in 1:K
			line = split(readline(f), sep; keepempty = false)
			r[k, e] = parse(Int64, line[2])
		end
	end
	for k in 1:(K)
		lineD = split(readline(f), sep; keepempty = false)
		lineO = split(readline(f), sep; keepempty = false)
		if parse(Int64, lineD[3]) >= 0
			push!(commodities, (parse(Int64, lineO[2]), parse(Int64, lineD[2]), parse(Int64, lineD[3])))
		else
			push!(commodities, (parse(Int64, lineD[2]), parse(Int64, lineO[2]), parse(Int64, lineO[3])))
		end
	end
	close(f)
	return LearningPi.cpuInstanceMCND(n, edges, commodities, fc, r, c)
end

"""
function modify_instance(ins::cpuInstanceMCND, seed::Int64, x::Real)

#Arguments: 
-`ins`: an instance object cpuInstanceMCND, 
-`seed`: the random generation seed
-`x`: a number that allows to increase/decrease the number of demands/commodities in the instance as size(new_demands)=round(x*size(new_demands))

return an instance generated straing from `ins` and perturbing the routing costs, the volumes, the origins and the destinations of the demands.
Furthermore it is possible to change the number of demands with the parameter `x`.
"""
function modify_instance(ins::cpuInstanceMCND, seed::Int64, x::Real)
	rng = Random.MersenneTwister(seed)

	K = round(Int64,x * sizeK(ins))
	o = floor.(rand(rng, K) * sizeV(ins)) .+ 1
	d = floor.(rand(rng, K) * sizeV(ins)) .+ 1
	for i in 1:(size(o, 1))
		while o[i] == d[i]
			o[i] = floor.(rand(rng, 1) * sizeV(ins))[1] % sizeV(ins) + 1
		end
	end

	q = [ins.K[i][3] for i in 1:sizeK(ins)]
	q = max.(1, floor.(mean(q) .+ sqrt(var(q)) .* (randn(rng, x * size(q, 1)))))
	q = Int64.(floor.(q ./ x) .+ 1)
	#q=shuffle(rng,q).+floor.(0.1*mean(q).*(rand(rng,size(q,1)).-0.5))

	commodities = [(o[k], d[k], q[k]) for k in 1:K]

	#    f = shuffle(rng,ins.f) .+ floor.(0.1*mean(ins.f).*(rand(rng,size(ins.f)[1]).-0.5))
	r = max.(1, floor.(mean(ins.r) .+ sqrt(var(ins.r)) .* (randn(rng, (K, sizeE(ins))))))

	return LearningPi.cpuInstanceMCND(ins.n, ins.edges, commodities, ins.f, r, ins.c)
end


"""
function print_dat(path::String, ins::cpuInstanceMCND)

	# Arguments:
		- `path`: the path to the file where we want print the data
		- `ins`: the instance object that we want print in a file, should be a cpuInstanceMCND
"""
function print_dat(path::String, ins::cpuInstanceMCND)
	f = open(path, "w")
	write(f, "\t" * string(sizeV(ins)) * "\t" * string(sizeE(ins)) * "\t" * string(sizeK(ins)) * "\n")
	for e in 1:sizeE(ins)
		write(f, "\t" * string(head(ins, e)) * "\t" * string(tail(ins, e)) * "\t" * string(fixed_cost(ins, e)) * "\t" * string(capacity(ins, e)) * "\t" * string(sizeK(ins)) * "\n")
		for k in 1:sizeK(ins)
			write(f, "\t" * string(k) * "\t" * string(routing_cost(ins, e, k)) * "\t" * string(min(volume(ins, k), capacity(ins, e))) * "\n")
		end
	end
	for k in 1:sizeK(ins)
		write(f, "\t" * string(k) * "\t" * string(origin(ins, k)) * "\t" * string(-volume(ins, k)) * "\n")
		write(f, "\t" * string(k) * "\t" * string(destination(ins, k)) * "\t" * string(volume(ins, k)) * "\n")
	end
	close(f)
end
