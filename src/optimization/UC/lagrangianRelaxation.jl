
"""
<<<<<<< HEAD
CR(ins)

# Arguments:
- `ins`: an instance structure.

Solve the linear relaxation of the provided instance and then, it returns the objective value of the linear relaxation, the dual variables associated to flow constraints, the dual variables associated to capacity constraints, the primal flow variables and the primal design variables.
=======
# Arguments:
- `ins`: an instance structure of type UC_instance.

Construct the Lagrangian Relaxation Sub-Problem for non-decomposed sub-problems.
>>>>>>> master
"""
function create_LR(ins::UC_instance)
	T = ins.T
	G = ins.G 

	#y1 ≥ 0
    model = Model(Gurobi.Optimizer)

    #@variable(model, 1 >= α[g = 1:G, t = 1:T] >= 0)
    #@variable(model, 1 >= γ[g = 1:G, t = 1:T] >= 0)
    #@variable(model, 1 >= η[g = 1:G, t = 1:T] >= 0)
	
	@variable(model, α[g = 1:G, t = 1:T],Bin)
	@variable(model, γ[g = 1:G, t = 1:T],Bin)
	@variable(model, η[g = 1:G, t = 1:T],Bin)
	@variable(model, Inf >= ρ[g = 1:G, t = 1:T] >= 0)

	@constraint(model, power_output_lower_bounds[g = 1:G, t = 1:T], ins.P_min_gen[g]*α[g,t]- ρ[g,t] <= 0)
	@constraint(model, power_output_upper_bounds[g = 1:G, t = 1:T], -ins.P_max_gen[g]*α[g,t]+ ρ[g,t] <= 0)

	@constraint(model, ramp_up_bounds[g = 1:G, t = 2:T], - ins.P_startup_ramp[g] * γ[g,t]-ins.P_ramp_up[g]*α[g,t-1] + ρ[g,t] - ρ[g,t-1] <= 0)
	@constraint(model, ramp_down_bounds[g = 1:G, t = 2:T], - ins.P_shutdown_ramp[g] *  η[g,t] -ins.P_ramp_down[g]*α[g,t] + ρ[g,t-1] - ρ[g,t] <= 0)

	@constraint(model, minimum_uptime[g = 1:G, t = 1:T], sum([γ[g,u] for u in  Int64(max(1,t-ins.T_startup_time[g]+1)):t]) - α[g,t] <= 0)
	@constraint(model, minimum_downtime[g = 1:G, t = 1:T], sum([η[g,u] for u in Int64(max(1,t-ins.T_startup_time[g]+1)):t]) + α[g,t] -1 <= 0)

	@constraint(model, logistical_constraints_1[g = 1:G, t = 2:T], + α[g,t] - α[g,t-1] - γ[g,t] + η[g,t] == 0)
	
	@constraint(model, logistical_constraints_2[g = 1:G, t = 1:T], γ[g,t] + η[g,t] <= 1)
	
    @objective(model, Min, LinearAlgebra.dot(repeat(ins.C_no_load',T) , α)+ LinearAlgebra.dot(repeat(ins.C_marginal',T),  ρ) + LinearAlgebra.dot(repeat(ins.C_startup',T),  γ ) )
    return model
end


<<<<<<< HEAD
"""
CR(ins)

# Arguments:
- `ins`: an instance structure.

Solve the linear relaxation of the provided instance and then, it returns the objective value of the linear relaxation, the dual variables associated to flow constraints, the dual variables associated to capacity constraints, the primal flow variables and the primal design variables.
=======

"""
# Arguments:
- `ins`: an instance structure of type UC_instance.
- `g`: the index of the generator associated to the sub-problem.

Construct the Lagrangian Relaxation Sub-Problem associated to a given generator `g` in the case
of decomposable sub-problems.
>>>>>>> master
"""
function create_LR_component(ins::UC_instance,g::Int64)
	T = ins.T
	
	#y1 ≥ 0
    model = Model(LR_Optimiser)

    #@variable(model, 1 >= α[g = 1:G, t = 1:T] >= 0)
    #@variable(model, 1 >= γ[g = 1:G, t = 1:T] >= 0)
    #@variable(model, 1 >= η[g = 1:G, t = 1:T] >= 0)
	
	@variable(model, α[t = 1:T],Bin)
	@variable(model, γ[t = 1:T],Bin)
	@variable(model, η[t = 1:T],Bin)
	@variable(model, Inf >= ρ[ t = 1:T] >= 0)

	@constraint(model, power_output_lower_bounds[t = 1:T], ins.P_min_gen[g]*α[t]- ρ[t] <= 0)
	@constraint(model, power_output_upper_bounds[t = 1:T], -ins.P_max_gen[g]*α[t]+ ρ[t] <= 0)

	@constraint(model, ramp_up_bounds[t = 2:T], - ins.P_startup_ramp[g] * γ[t]-ins.P_ramp_up[g]*α[t-1] + ρ[t] - ρ[t-1] <= 0)
	@constraint(model, ramp_down_bounds[t = 2:T], - ins.P_shutdown_ramp[g] *  η[t] -ins.P_ramp_down[g]*α[t] + ρ[t-1] - ρ[t] <= 0)

	@constraint(model, minimum_uptime[t = 1:T], sum([γ[u] for u in  Int64(max(1,t-ins.T_startup_time[g]+1)):t]) - α[t] <= 0)
	@constraint(model, minimum_downtime[t = 1:T], sum([η[u] for u in Int64(max(1,t-ins.T_startup_time[g]+1)):t]) + α[t] -1 <= 0)

	@constraint(model, logistical_constraints_1[t = 2:T], + α[t] - α[t-1] - γ[t] + η[t] == 0)
	
	@constraint(model, logistical_constraints_2[t = 1:T], γ[t] + η[t] <= 1)
	
    @objective(model, Min, LinearAlgebra.dot(ins.C_no_load[g]*ones(T) , α)+ LinearAlgebra.dot(ins.C_marginal[g]*ones(T),  ρ) + LinearAlgebra.dot(ins.C_startup[g]*ones(T),  γ ) )
    return model
end

<<<<<<< HEAD
function modify_objective(ins,y1,y2)
=======
"""
# Arguments:
- `ins`: an instance structure of type `UC_instance`
- `y1`: the dual variables vector associated to Power Demand Constraints
- `y2`: the dual variables vector associated to Reserve Constraints

For Undecomposed sub-problem.
Modify the objective function of the Lagrangian Sub-Problem in the model constained in `ins` considering `y1` and `y2` as Lagrangian Mutlipliers vectors.
"""
function modify_objective(ins::UC_instance,y1::AbstractVector,y2::AbstractVector)
>>>>>>> master
    if ins.model.is_decomposable
        modify_objectives(ins,y1,y2)
    else 
    @objective(ins.model.formulation, Min, 
      LinearAlgebra.dot( repeat(ins.C_no_load',ins.T) , ins.model.formulation[:α] )
    + LinearAlgebra.dot( repeat(ins.C_marginal',ins.T), ins.model.formulation[:ρ] ) 
    + LinearAlgebra.dot( repeat(ins.C_startup',ins.T) , ins.model.formulation[:γ] )    
    + LinearAlgebra.dot( y2 , ins.Pr_reserve_requirement )
    - LinearAlgebra.dot(ins.model.formulation[:α]' * ins.P_max_gen, y2)
    + LinearAlgebra.dot(ins.model.formulation[:ρ]'*ones(ins.G), y2)
    + LinearAlgebra.dot(y1, ins.Pd_power_demend )
    - LinearAlgebra.dot(y1 , ins.model.formulation[:ρ]'*ones(ins.G)) 
    )
    end
end



<<<<<<< HEAD
function modify_objectives(ins,y1,y2)
=======
"""
# Arguments:
- `ins`: an instance structure of type `UC_instance`
- `y1`: the dual variables vector associated to Power Demand Constraints
- `y2`: the dual variables vector associated to Reserve Constraints

For Decomposed sub-problem.
Modify the objectives functions of the Lagrangian Sub-Problems in the model constained in `ins` considering `y1` and `y2` as Lagrangian Mutlipliers vectors.
"""
function modify_objectives(ins::UC_instance,y1::AbstractVector,y2::AbstractVector)
>>>>>>> master
    for (g,component) in ins.model.decomposed
    @objective(component, Min, 
      LinearAlgebra.dot( ins.C_no_load[g] * ones(ins.T) , component[:α] )
    + LinearAlgebra.dot( ins.C_marginal[g] * ones(ins.T), component[:ρ] ) 
    + LinearAlgebra.dot( ins.C_startup[g] * ones(ins.T) , component[:γ] )    
    - LinearAlgebra.dot( component[:α] * ins.P_max_gen[g], y2)
    + LinearAlgebra.dot( component[:ρ], y2)
    - LinearAlgebra.dot(y1 , component[:ρ]) 
    )
    end
    ins.model.constant_term =  LinearAlgebra.dot( y2 , ins.Pr_reserve_requirement ) + LinearAlgebra.dot(y1, ins.Pd_power_demend )
end

function optimize_model!(model)
    if model.is_decomposable
        for (g,component) in model.decomposed
            set_silent(component)
            optimize!(component)
        end    
    else
        set_silent(model.formulation)
        optimize!(model.formulation)
    end
end



#@constraint(model, load_balance[t = 1:T], sum(ρ[:,t]) >= ins.Pd_power_demend[t])
#	@constraint(model, reserve[t = 1:T], sum([ ins.P_max_gen[g] * α[g,t] - ρ[g,t] for g in 1:G]) >= ins.Pr_reserve_requirement[t])
function value_objective(model)
    if model.is_decomposable
        obj=model.constant_term
        for (g,comp) in model.decomposed
            obj+=JuMP.objective_value(comp)
        end
        return obj
    else
        return JuMP.objective_value(model.formulation)
    end
end	


function LR(ins,λ)
    model=ins.model
    
    modify_objective(ins,λ[1:Int64(end/2)],λ[Int64(end/2+1):end])    


    optimize_model!(model)

    αV  = zeros(ins.G,ins.T) 
    γV  = zeros(ins.G,ins.T)
    ηV  = zeros(ins.G,ins.T)
    ρV  = zeros(ins.G,ins.T)

    for g in 1:ins.G
        for t in 1:ins.T
            αV[g , t ] = model.is_decomposable ? value(ins.model.decomposed[g][2][:α][t]) : value(ins.model.formulation[:α][g , t ])
            γV[g , t ] = model.is_decomposable ? value(ins.model.decomposed[g][2][:γ][t]) : value(ins.model.formulation[:γ][g , t ])
            ηV[g , t ] = model.is_decomposable ? value(ins.model.decomposed[g][2][:η][t]) : value(ins.model.formulation[:η][g , t ])
            ρV[g , t ] = model.is_decomposable ? value(ins.model.decomposed[g][2][:ρ][t]) : value(ins.model.formulation[:ρ][g , t ])
        end
    end

    return value_objective(model),(ρV, αV),(γV,ηV)
end