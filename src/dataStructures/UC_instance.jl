struct cpuUCinstanceFactory <:abstractInstanceFactory end

<<<<<<< HEAD
=======
"""
Structure to describe  a UC instance in such a way that the sub-problem can be solved using CPU.

# Fields:
- `G` : number of generators
- `T` : time horizon
- `C_no_load` : no load cost
- `C_marginal` : marginal cost
- `C_startup` : startup cost
- `P_max_gen` : maximum power generation 
- `P_min_gen` : minimum power generation  
- `P_ramp_up` : up-ramp factor when working
- `P_ramp_down` : down-ramp factor  when working
- `P_startup_ramp` : up-ramp factor when startup
- `P_shutdown_ramp`: down-ramp factor when shutdown
- `T_startup_time` : minimum time required to be on when the generator is startup
- `T_shutdown_time`: minimum time required to be off when the generator is shutdown
- `Pd_power_demend`: power demand
- `Pr_reserve_requirement: power requirement
- `model`: the JuMP model that describe Lagrangian relaxation of the instance
"""
>>>>>>> master
mutable struct UC_instance <: abstractInstance
	G::Int64 # number of generators
	T::Int64 # time horizon
	### generators
	#costs	
	C_no_load::Vector{Float32} #C^{nl}
	C_marginal::Vector{Float32} #C^{mr}
	C_startup::Vector{Float32} #C^{up}
	#limits
	P_max_gen::Vector{Float32} #P^{MAX/MIN}
	P_min_gen::Vector{Float32} 
	P_ramp_up::Vector{Float32} #P^{ru/rd}
	P_ramp_down::Vector{Float32}
	P_startup_ramp::Vector{Float32} #P^{su/sd}
	P_shutdown_ramp::Vector{Float32}
	T_startup_time::Vector{Float32} #T^{u/d}
	T_shutdown_time::Vector{Float32}
	###demands
	Pd_power_demend::Vector{Float32}
	Pr_reserve_requirement::Vector{Float32}
    model::Any
end

<<<<<<< HEAD
=======
"""
Structure that describe the Lagrangian Relaxation of an UC instance in such a way that the sub-problems are solved in a decomposed formulation.
In other world the Lagrangian Sub-Problem are saved as independent JuMP models that are then solved sequentially, but independently.
"""
>>>>>>> master
mutable struct decomposed_model
    decomposed::Vector{Pair{Int64,JuMP.Model}}
    constant_term::Float32
    is_decomposable::Bool
    decomposed_model(decomposed,constant_term)=new(decomposed,constant_term,true)
end


<<<<<<< HEAD
=======
"""
Structure that describe the Lagrangian Relaxation of an UC instance in such a way that the sub-problems are written in a unique JuMP model.
This should be used only if the solver associated to the model is automatically capable to exploit the decomposable structure. 
"""
>>>>>>> master
mutable struct compact_model
    formulation::JuMP.Model
    is_decomposable::Bool
    compact_model(formulation)=new(formulation,false)
end

<<<<<<< HEAD

=======
"""

# Arguments:

- `G` : number of generators
- `T` : time horizon
- `C_no_load` : no load cost
- `C_marginal` : marginal cost
- `C_startup` : startup cost
- `P_max_gen` : maximum power generation 
- `P_min_gen` : minimum power generation  
- `P_ramp_up` : up-ramp factor when working
- `P_ramp_down` : down-ramp factor  when working
- `P_startup_ramp` : up-ramp factor when startup
- `P_shutdown_ramp`: down-ramp factor when shutdown
- `T_startup_time` : minimum time required to be on when the generator is startup
- `T_shutdown_time`: minimum time required to be off when the generator is shutdown
- `Pd_power_demend`: power demand
- `Pr_reserve_requirement: power requirement
- `model`: the JuMP model that describe the instance
"""
>>>>>>> master
function create_data_object(G::Int64,T::Int64,C_no_load::Vector{Float32},C_marginal::Vector{Float32} ,C_startup::Vector{Float32},P_max_gen::Vector{Float32},P_min_gen::Vector{Float32} ,P_ramp_up::Vector{Float32},P_ramp_down::Vector{Float32},P_startup_ramp::Vector{Float32},P_shutdown_ramp::Vector{Float32},T_startup_time::Vector{Float32},T_shutdown_time::Vector{Float32},Pd_power_demend::Vector{Float32},Pr_reserve_requirement::Vector{Float32},model::Any) 
    return UC_instance(G,T,C_no_load,C_marginal ,C_startup,P_max_gen,P_min_gen,P_ramp_up,P_ramp_down,P_startup_ramp,P_shutdown_ramp,T_startup_time,T_shutdown_time,Pd_power_demend,Pr_reserve_requirement,model) 
end

<<<<<<< HEAD

vector_read_line(file) =  split(readline(file))

=======
"""
# Arguments:
- `path`: a string specifying the path to the file
- `factory`: a factory that allows to return the correct instance type, for this function should be `cpuUCinstanceFactory`  
- `decomposable`: a boolean that say if memorize the problems as decomposed sub-problems.

read a data from the file in `path` and return an instance assocated to the `factory`.
"""
>>>>>>> master
function read_dat(path::String,factory::cpuUCinstanceFactory,decomposable=true)
    G=0
    T=0
    ### generators
    #costs	
	file=open(path,"r")
<<<<<<< HEAD
	line = vector_read_line(file)
	#@assert(line[1]=="ProblemNum", "Error in file format on GENERATOR INDEX")
	seed = parse(Float32,line[2])
	
	line = vector_read_line(file)
	#@assert(line[1]=="HorizonLen", "Error in file format on HORIZON LENGTH")
	T = parse(Int64,line[2])
	
	line = vector_read_line(file)
=======
	line = split(readline(file))
	#@assert(line[1]=="ProblemNum", "Error in file format on GENERATOR INDEX")
	seed = parse(Float32,line[2])
	
	line = split(readline(file))
	#@assert(line[1]=="HorizonLen", "Error in file format on HORIZON LENGTH")
	T = parse(Int64,line[2])
	
	line = split(readline(file))
>>>>>>> master
	#@assert(line[1]=="NumTermal", "Error in file format on TERMAL NUMBER")
	G = parse(Int64,line[2])

    C_no_load=zeros(Float32, G)
    C_marginal=zeros(Float32,G)
    C_startup=zeros(Float32,G)
    #limits
    P_max_gen=zeros(Float32,G)
    P_min_gen=zeros(Float32,G) 
    P_ramp_up=zeros(Float32,G)
    P_ramp_down=zeros(Float32,G)
    P_startup_ramp=zeros(Float32,G)
    P_shutdown_ramp=zeros(Float32,G)
    T_startup_time=zeros(Float32,G)
    T_shutdown_time=zeros(Float32,G)
    ###demands
    Pd_power_demend=zeros(Float32,T)
    Pr_reserve_requirement=zeros(Float32,T)

<<<<<<< HEAD
	line = vector_read_line(file)
	#@assert(line[1]=="NumHydro", "Error in file format on HYDRO NUMBER")
	#@assert(line[2]==0, "Error: this version only support termal units")

	line = vector_read_line(file)
=======
	line = split(readline(file))
	#@assert(line[1]=="NumHydro", "Error in file format on HYDRO NUMBER")
	#@assert(line[2]==0, "Error: this version only support termal units")

	line = split(readline(file))
>>>>>>> master
	#@assert(line[1]=="NumCascade", "Error in file format on CASCADE NUMBER")
	#@assert(line[2]==0, "Error: this version only support termal units")


<<<<<<< HEAD
	line = vector_read_line(file)
	#@assert(line[1]=="Load Curve", "Error in file format")
	
	
	line = vector_read_line(file)
=======
	line = split(readline(file))
	#@assert(line[1]=="Load Curve", "Error in file format")
	
	
	line = split(readline(file))
>>>>>>> master
	#@assert(line[1]=="MinSystemCapacity", "Error in file format on MINIMUM SYSTEM CAPACITY")
	MinSystemCapacity = parse(Float32,line[2])


<<<<<<< HEAD
	line = vector_read_line(file)
=======
	line = split(readline(file))
>>>>>>> master
	#@assert(line[1]=="MaxSystemCapacity", "Error in file format on MAXIMUM SYSTEM CAPACITY")
	MaxSystemCapacity = parse(Float32,line[2])


<<<<<<< HEAD
	line = vector_read_line(file)
=======
	line = split(readline(file))
>>>>>>> master
	#@assert(line[1]=="MaxThermalCapacity", "Error in file format on MAXIMUM TERMAL CAPACITY")
	MaxThermalCapacity = parse(Float32,line[2])
	#@assert(MaxThermalCapacity==MaxSystemCapacity,"Error: this version only support termal units")



<<<<<<< HEAD
	line = vector_read_line(file)
	#@assert(line[1]=="Loads", "Error in file format on LOADS")
	#@assert(line[2]==1, "Error in file format on LOADS")
	#@assert(line[3]==T, "Error in file format on LOADS")
	line = vector_read_line(file)
	Pd_power_demend = parse.(Float32,line)
	

	line = vector_read_line(file)
	#@assert(line[1]=="SpinningReserve", "Error in file format on SPINNING RESERVE")
	#@assert(line[2]==1, "Error in file format on SPINNING RESERVE")
	#@assert(line[3]==T, "Error in file format on SPINNING RESERVE")
	line = vector_read_line(file)
=======
	line = split(readline(file))
	#@assert(line[1]=="Loads", "Error in file format on LOADS")
	#@assert(line[2]==1, "Error in file format on LOADS")
	#@assert(line[3]==T, "Error in file format on LOADS")
	line = split(readline(file))
	Pd_power_demend = parse.(Float32,line)
	

	line = split(readline(file))
	#@assert(line[1]=="SpinningReserve", "Error in file format on SPINNING RESERVE")
	#@assert(line[2]==1, "Error in file format on SPINNING RESERVE")
	#@assert(line[3]==T, "Error in file format on SPINNING RESERVE")
	line = split(readline(file))
>>>>>>> master
    Pr_reserve_requirement = parse.(Float32,line)

    ###demands
    
<<<<<<< HEAD
	line = vector_read_line(file)
	#@assert(line[1]=="ThermalSection", "Error in file format on TERMAL SECTION")
	for g in 1:G
			generator_index, quadratic_cost, linear_cost, constant_cost, min_out, max_out, init_status, min_up, min_down, cool_and_fuel_cost, hot_and_fuel_cost, tau, tau_max, fixed_cost, succ, p0= parse.(Float32,vector_read_line(file))
			line = vector_read_line(file)
=======
	line = split(readline(file))
	#@assert(line[1]=="ThermalSection", "Error in file format on TERMAL SECTION")
	for g in 1:G
			generator_index, quadratic_cost, linear_cost, constant_cost, min_out, max_out, init_status, min_up, min_down, cool_and_fuel_cost, hot_and_fuel_cost, tau, tau_max, fixed_cost, succ, p0= parse.(Float32,vector_read_line(file))
			line = split(readline(file))
>>>>>>> master
			#@assert(line[1]=="RampingConstraint", "Error in file format on RAMPING CONSTRAINTS")
			ramp_up,ramp_down = parse.(Float32,line[2:end])
            
            min_cost = (constant_cost + linear_cost * min_out + quadratic_cost * (min_out^2) )
            max_cost = (constant_cost + linear_cost * max_out + quadratic_cost * (max_out^2) )
            

            
            C_marginal[g] =  (max_cost - min_cost) / (max_out - min_out)
            C_no_load[g] = min_cost - C_marginal[g] * min_out
            # It seems the description of the data file is not correct,
            # since `hot_and_fuel_cost` seems to be 0 anywhere.
            # Instead, reading values in the following way seems more realistic.
            # - fixed_cost -> cool_and_fuel_cost
            # - succ -> hot_and_fuel_cost
            # - p0 -> fixed_cost
            C_startup[g] =  succ * 12.0 + p0
   
            #limits
            P_max_gen[g] = max_out
            P_min_gen[g] = min_out
            P_ramp_up[g] = ramp_up
            P_ramp_down[g] = ramp_down
            
            P_startup_ramp[g] = max(
                ramp_up, min_out
            )
            P_shutdown_ramp[g] = max(
                ramp_down, min_out
            )
            T_startup_time[g] = min_up
            T_shutdown_time[g] = min_down
	end

	close(file)
    
    if decomposable 
        model = decomposed_model( Vector{Pair{Int64,JuMP.Model}}[ ] , 0.0f0 )
        ins = UC_instance(G,T,C_no_load,C_marginal,C_startup,P_max_gen,P_min_gen,P_ramp_up,P_ramp_down, P_startup_ramp, P_shutdown_ramp, T_startup_time,T_shutdown_time,Pd_power_demend,Pr_reserve_requirement,model)         
        for g in 1:G
             push!(model.decomposed,Pair(g,create_LR_component(ins,g)))
        end
        model.constant_term = 0.0f0
        return ins
    end
    model = compact_model(JuMP.Model())
    ins = UC_instance(G,T,C_no_load,C_marginal,C_startup,P_max_gen,P_min_gen,P_ramp_up,P_ramp_down, P_startup_ramp, P_shutdown_ramp, T_startup_time,T_shutdown_time,Pd_power_demend,Pr_reserve_requirement,model)
    ins.model.formulation = create_LR(ins)
    return ins
<<<<<<< HEAD
end






#function create_instance(ins::UnitCommitmentInstance)
#	G = length(ins.units)
#	T = ins.time
#	### generators
#	#costs	
#	C_no_load = [ins.units[g].min_power_cost[1] for g in 1:G]
#	C_marginal::Vector{Float32} #C^{mr}
#	C_startup =  [ins.units[g].startup_categories[1].cost for g in 1:G]
#
#	#limits
#	P_max_gen = [ins.units[g].max_power for g in 1:G]
#	P_min_gen = [ins.units[g].min_power for g in 1:G]
#	P_ramp_up =  [ins.units[g].ramp_up_limit for g in 1:G]
#	P_ramp_down =  [ins.units[g].ramp_down_limit for g in 1:G]
#	P_startup_ramp =  [ins.units[g].startup_limit for g in 1:G]
#    P_shutdown_ramp = [ins.units[g].shutdown_ramp for g in 1:G]
#    
#	T_startup_time = [ins.units[g].min_uptime for g in 1:G]]
#	T_shutdown_time = [ins.units[g].min_downtime for g in 1:G]]
#	###demands
#	Pd_power_demend::Vector{Float32}
#	Pr_reserve_requirement::Vector{Float32}
#    model::JuMP.Model
#end
=======
end
>>>>>>> master
