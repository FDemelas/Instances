"""
Abstract type for every instance of the Capacitated Warehose Locaation Problem.
"""
abstract type instanceCWL <: abstractInstance end

"""
Structure that describe a cpu instance of the Capacitated Warehose Locaation Problem.
#Fields:
-`I`: facilities number
-`J`: custumers number
-`c`: costs to satisfy the demand of a costumer with a certain facility 
-`f`: costs for open facilities
-`q`: capacity of the facilities
-`d`: demands of the costumers
"""
struct cpuInstanceCWL <: instanceCWL
	I::Int64 #n facilities
	J::Int64 #n custumers
	c::Matrix{Float32}
	f::Vector{Float32}
	q::Vector{Int64}
	d::Vector{Int64}
end

"""
Abstract Factory for every instance of the Capacitated Warehose Locaation Problem.
"""
abstract type CWLinstanceFactory <: abstractInstanceFactory end

"""
Factory structure used to construct an instance of the Bin Packing Problem.
"""
struct cpuCWLinstanceFactory <: CWLinstanceFactory end

"""
Given the data, creates an instance of the Bin Packing Problem.
"""
create_data_object(::cpuCWLinstanceFactory, I, J, c, f, q, d) = cpuInstanceCWL(I, J, c, f, q, d)

"""
function lengthLM(ins::instanceCWL)
	
# Arguments:
- `ins`: instance object, should be a sub-type of instanceCWL

returns the length of the Lagrangian multipliers for the instance `ins`.
"""
function lengthLM(ins::instanceCWL)
	return ins.J
end

"""
function sizeLM(ins::instanceCWL)

# Arguments:
- `ins`: instance object, should be a sub-type of instanceCWL

returns the size of the Lagrangian multipliers for the instance `ins`.
"""
function sizeLM(ins::instanceCWL)
	return (1, ins.J)
end

"""
function read_dat(path::String,factory::cpuCWLinstanceFactory)

#Arguments:
- `path` : a String that contains the path to the data file of the instance
- `factory` : a cpuCWLinstanceFactory used only to construct an instance of the proper type ( . <: instanceCWL)

Given the `path` and the `factory` of the proper type, this function reads the instance information 
from the file in the provided path and construct an instance of the Capacitated Warehouse Location problem.
"""
function read_dat(path::String, factory::cpuCWLinstanceFactory)
	f = open(path, "r")
	sep = " "

	line = split(readline(f), sep; keepempty = false)
	I = parse(Int64, line[1])
	J = parse(Int64, line[2])

	q, fc = zeros(Int64, I), zeros(Float32, I)

	for i in 1:I
		line = split(readline(f), sep; keepempty = false)
		q[i] = parse(Int64, line[1])
		fc[i] = parse(Float32, line[2])
	end

	c = zeros(Float32, (I, J))
	demands = zeros(Int64, J)
	for j in 1:J
		line = readline(f)
		demands[j] = parse(Int64, line)
		line = split(readline(f), sep; keepempty = false)
		readed = length(line)
		idx = 1
		for i in 1:I
			if i > readed
				idx = 1
				line = split(readline(f), sep; keepempty = false)
				readed += length(line)
			end
			c[i, j] = parse(Float32, line[idx])
			idx += 1
		end
	end
	close(f)
	return LearningPi.cpuInstanceCWL(I, J, c, fc, q, demands)
end

"""
function print_dat(path::String, ins::CWLinstance)

	# Arguments:
		- `path`: the path to the file where we want print the data
		- `ins`: the instance object that we want print in a file, should be a sub-type of instanceCWL
"""
function print_dat(path::String, ins::instanceCWL)
	f = open(path, "w")
	write(f, string(ins.I) * " " * string(ins.J) * "\n")

	for i in 1:ins.I
		write(f, string(ins.q[i]) * " " * string(ins.f[i]) * "\n")
	end

	for j in 1:ins.J
		write(f, string(ins.d[j]) * "\n")
		line = " "
		for i in 1:ins.I
			line *= string(ins.c[i, j]) * " "
		end
		write(f, line * "\n")
	end
	close(f)
end

"""
function read_modify_dat(path::String,factory::cpuCWLinstanceFactory,seed=1,α=0.75,newJ=100)

#Arguments:
  - `path`: the path to the file .dat containing the information about the instance that we want modify
  - `factory`: the instance factory, supports cpuCWLinstanceFactory
  - `seed`: the random generation seed
  - `α` : a parameter that controll the ratio of demands and capacities
  - `newJ`: new number of custumers
  
  returns an instance obtained modifying the original one.
"""
function read_modify_dat(path::String, factory::cpuCWLinstanceFactory, seed = 1, α = 0.75, newJ = 100)
	f = open(path, "r")
	sep = " "
	0.75
	line = split(readline(f), sep; keepempty = false)
	I = parse(Int64, line[1]) # facility location
	J = parse(Int64, line[2]) # customer

	q, fc = zeros(Int64, I), zeros(Float32, I) #

	for i in 1:I
		line = split(readline(f), sep; keepempty = false)
		q[i] = parse(Int64, line[1])
		fc[i] = parse(Float32, line[2])
	end

	c = zeros(Float32, (I, J))
	demands = zeros(Int64, J)
	for j in 1:J
		line = readline(f)
		demands[j] = parse(Int64, line)
		line = split(readline(f), sep; keepempty = false)
		readed = length(line)
		idx = 1
		for i in 1:I
			if i > readed
				idx = 1
				line = split(readline(f), sep; keepempty = false)
				readed += length(line)
			end
			c[i, j] = parse(Float32, line[idx])
			idx += 1
		end
	end
	close(f)

	ĉ = c ./ demands'

	rng = Random.MersenneTwister(seed)

	ĉ = mean(ĉ) .+ 2 * sqrt(var(ĉ)) .* rand(rng, I, J)

	demands = mean(demands) .+ 2 * sqrt(var(demands)) .* rand(rng, J)
	demands = round.(Int64, min.(demands, maximum(q)))

	#if α*sum(q) < sum(demands[1:newJ])
	demands[1:newJ] *= round.(Int64, α * sum(q) / sum(demands[1:newJ]))
	#end

	c = ĉ .* demands'

	return LearningPi.cpuInstanceCWL(I, newJ, c[:, 1:newJ], fc / 100, q, demands[1:newJ])
end
