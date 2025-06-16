"""
Abstract type for every instance of the Generalized Assignment Problem.
"""
abstract type instanceGA <: abstractInstance end

"""
Structure that describe an instance of the Generalized Assignment Problem.
#Fields:
- `I`: number of items
- `J`: number of bins
- `p`: profit matrix
- `w`: weights matrix
- `c`: capacities of the bins
"""
struct cpuInstanceGA <: instanceGA
	I::Int64 # n objects
	J::Int64 # m bins
	p::Matrix{Float32}
	w::Matrix{Float32}
	c::Vector{Int64}
end

"""
Abstract Factory for every instance of the Generalized Assignment Problem.
"""
abstract type GAinstanceFactory <: abstractInstanceFactory end

"""
Factory structure used to construct an instance of the Bin Packing Problem.
"""
struct cpuGAinstanceFactory <: GAinstanceFactory end

"""
function create_data_object(::cpuGAinstanceFactory, I, J, c, f, q, d)

#Arguments:
- `_`: factory of type cpuGAinstanceFactory
- `I`: number of items
- `J`: number of bins
- `p`: profit matrix
- `w`: weights matrix
- `c`: capacities of the bins

Given the data, creates an instance of the Bin Packing Problem.
"""
create_data_object(::cpuGAinstanceFactory, I, J, p, w, c) = cpuInstanceGA(I, J, p, w, c)

"""
function lengthLM(ins::instanceGA)
	
# Arguments:
- `ins`: instance object, should be a sub-type of instanceGA

returns the length of the Lagrangian multipliers for the instance `ins`.
"""
function lengthLM(ins::instanceGA)
	return ins.I
end

"""
function sizeLM(ins::instanceGA)
	
# Arguments:
- `ins`: instance object, should be a sub-type of instanceGA

returns the size of the Lagrangian multipliers for the instance `ins`.
"""
function sizeLM(ins::instanceGA)
	return (1, ins.I)
end

"""
function read_dat(path::String, _::cpuGAinstanceFactory)

#Arguments:
-`path`: a String that contains the path to the data file of the instance
-`_`: a cpuGAinstanceFactory used only to construct an instance of the proper type ( . <: instanceGA)

Given the path and the factory of the proper type, this function reads the instance information from the file in the provided path and construct an instance of the Bin Packing problem.
"""
function read_dat(path::String, _::cpuGAinstanceFactory)
	f = open(path, "r")
	sep = " "

	line = split(readline(f), sep; keepempty = false)
	J = parse(Int64, line[1])
	I = parse(Int64, line[2])

	p = zeros(Float32, (I, J))
	w = zeros(Float32, (I, J))

	for j in 1:J
		idx = 1
		line = split(readline(f), sep; keepempty = false)
		readed = length(line)
		for i in 1:I
			if i > readed
				idx = 1
				line = split(readline(f), sep; keepempty = false)
				readed += length(line)
			end
			p[i, j] = parse(Float32, line[idx])
			idx += 1
		end
	end

	for j in 1:J
		idx = 1
		line = split(readline(f), sep; keepempty = false)
		readed = length(line)
		for i in 1:I
			if i > readed
				idx = 1
				line = split(readline(f), sep; keepempty = false)
				readed += length(line)
			end
			w[i, j] = parse(Float32, line[idx])
			idx += 1
		end
	end

	c = zeros(Int64, J)

#	line = split(readline(f), sep; keepempty = false)
#	readed = length(line)
	idx=1
    while idx <= J
		idxL=1
		line = split(readline(f), sep; keepempty = false)
		readed = length(line)
		for _ in 1:readed
			c[idx] = parse(Float32, line[idxL])
			idx += 1
		end
		if idx == J+1
			break
		end
	end

	close(f)
	return cpuInstanceGA(I, J, p, w, c)
end

"""
function print_dat(path::String, ins::instanceGA)

	# Arguments:
		- `path`: the path to the file where we want print the data
		- `ins`: the instance object that we want print in a file, should be a <: instanceGA

print the instance in a .dat file.
"""
function print_dat(path::String, ins::instanceGA)
	f = open(path, "w")
	write(f, string(ins.J) * " " * string(ins.I) * "\n")

	for j in 1:ins.J
		line = " "
		for i in 1:ins.I
			line *= string(Int64(ins.p[i, j])) * " "
		end
		write(f, line * "\n")
	end

	for j in 1:ins.J
		line = " "
		for i in 1:ins.I
			line *= string(Int64(ins.w[i, j])) * " "
		end
		write(f, line * " \n ")
	end

	for j in 1:ins.J
		write(f, string(Int64(ins.c[j])) * "\n")
	end

	close(f)
end

"""
function read_modify_dat(path::String,factory::cpuGAinstanceFactory,seed=1,α=0.75,newJ=100)

#Arguments:
  - `path`: the path to the file .dat containing the information about the instance that we want modify
  - `factory`: the instance factory, supports cpuGAinstanceFactory
  - `seed`: the random generation seed
  - `α` : a parameter that controll the ratio of demands and capacities
  - `newJ`: new number of custumers
  
  returns an instance obtained modifying the original one.
"""
function read_modify_dat(path::String, factory::cpuGAinstanceFactory, seed = 1, α = 0.75)
	ins = read_dat(path, factory)

	I, J, c, p, w = ins.I, ins.J, ins.c, ins.p, ins.w

	rng = Random.MersenneTwister(seed)

	#min_c,max_c = 0.9*minimum(c), 1.1*maximum(c)
	min_w, max_w = 0.8 * minimum(w), 1.2 * maximum(w)
	min_p, max_p = 0.8 * minimum(p), 1.2 * maximum(p)

	#c = mean(c) .+ 2 * sqrt(var(c)) .* rand(rng, J)
	p = mean(p) .+ 2 * sqrt(var(p)) .* rand(rng, I, J)
	w = mean(w) .+ 2 * sqrt(var(w)) .* rand(rng, I, J)

	#c = round.(Int64,min.(max.(c,min_c),max_c))
	p = round.(Int64, min.(max.(p, min_p), max_p))
	w = round.(Int64, min.(max.(w, min_w), max_w))

	return cpuInstanceGA(I, J, p, w, c)
end

"""
function generate_GA(seed::Int,I::Int,J::Int)

#Arguments:
- `seed`: random generator seed.
- `I`: number of items.
- `J`: number of bins.

return an instance with `I` items and `J` bins generating the weights as:

wᵢⱼ = 1 - 10⋅ϵᵢⱼ 

the profits as:

pᵢⱼ = wᵢⱼ/1000 - 10 ⋅ δᵢⱼ

where ϵᵢⱼ, δᵢⱼ are random values in (0,1).

The capacities are genrated as:
"""
function generate_GA(seed::Int, I::Int, J::Int)
	rng = MersenneTwister(seed)

	w = 1 .- 10 * log.(rand(rng, I, J))
	p =   1000 ./ w - 10 * rand(rng, I, J)
	
	# w = 1 .- 10 * log.(rand(rng, I, J))
	# p = w .* 1 / 100 - 10 * rand(rng, I, J)
	
	c = [min(0.8 / J * sum(w[:, j]), round(maximum(w[:, j]))) for j in 1:J]

	w = round.(w)
	p = round.(p)
	c = round.(c)

	return cpuInstanceGA(I, J, p, w, c)
end


"""
function generate_GA2(seed::Int,I::Int,J::Int)

#Arguments:
- `seed`: random generator seed.
- `I`: number of items.
- `J`: number of bins.

return an instance with `I` items and `J` bins generating an instance as the `generate_GA` function.
The main difference is that ... 
"""
function generate_GA2(seed::Int, I::Int, J::Int)
	rng = MersenneTwister(seed)
	d=Beta(0.5,0.5)

	w = 1 .- 10 * log.(rand(rng,d, I, J))
	p = w .* 1 / 1000 - 10 * rand(rng,d, I, J)

	min_w,max_w=minimum(w),maximum(w)
	min_p,max_p=minimum(p),maximum(p)

	w=min_w .+ rand(rng,d, I, J).*(max_w-min_w)
	p=min_p .+ rand(rng,d, I, J).*(max_p-min_p)

	c = [max(0.8 / J * sum(w[:, j]), maximum(w[:, j])) for j in 1:J]
	w = max.(1, round.(w))
	p = max.(1, round.(-p .* 10))
	c = ceil.(c)
	
	return cpuInstanceGA(I, J, p, w, c)
end
