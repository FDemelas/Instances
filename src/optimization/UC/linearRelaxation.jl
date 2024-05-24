
"""
CR(ins)

# Arguments:
- `ins`: an instance structure.

Solve the linear relaxation of the provided instance and then, it returns the objective value of the linear relaxation, the dual variables associated to flow constraints, the dual variables associated to capacity constraints, the primal flow variables and the primal design variables.
"""
function CR(ins::UC_instance)
	T = ins.T
	G = ins.G 
# αγηρ
	model = Model(CR_Optimiser)

	@variable(model, 1 >= α[g = 1:G, t = 1:T] >= 0)
	@variable(model, 1 >= γ[g = 1:G, t = 1:T] >= 0)
	@variable(model, 1 >= η[g = 1:G, t = 1:T] >= 0)
	@variable(model, Inf >= ρ[g = 1:G, t = 1:T] >= 0)

	@constraint(model, load_balance[t = 1:T], sum(ρ[:,t]) >= ins.Pd_power_demend[t])
	@constraint(model, reserve[t = 1:T], sum([ ins.P_max_gen[g] * α[g,t] - ρ[g,t] for g in 1:G]) >= ins.Pr_reserve_requirement[t])
	
    @constraint(model, power_output_lower_bounds[g = 1:G, t = 1:T], ins.P_min_gen[g]*α[g,t]- ρ[g,t] <= 0)
	@constraint(model, power_output_upper_bounds[g = 1:G, t = 1:T], -ins.P_max_gen[g]*α[g,t]+ ρ[g,t] <= 0)


	@constraint(model, ramp_up_bounds[g = 1:G, t = 2:T], - ins.P_startup_ramp[g] * γ[g,t]-ins.P_ramp_up[g]*α[g,t-1] + ρ[g,t] - ρ[g,t-1] <= 0)
	@constraint(model, ramp_down_bounds[g = 1:G, t = 2:T], - ins.P_shutdown_ramp[g] *  η[g,t] -ins.P_ramp_down[g]*α[g,t] + ρ[g,t-1] - ρ[g,t] <= 0)

	@constraint(model, minimum_uptime[g = 1:G, t = 1:T], sum([γ[g,u] for u in  Int64(max(1,t-ins.T_startup_time[g]+1)):t]) - α[g,t] <= 0)
	@constraint(model, minimum_downtime[g = 1:G, t = 1:T], sum([η[g,u] for u in Int64(max(1,t-ins.T_startup_time[g]+1)):t]) + α[g,t] -1 <= 0)

	@constraint(model, logistical_constraints_1[g = 1:G, t = 2:T], + α[g,t] - α[g,t-1] - γ[g,t] + η[g,t] == 0)
		
	@constraint(model, logistical_constraints_2[g = 1:G, t = 1:T], γ[g,t] + η[g,t] <= 1)
	
	@objective(model, Min, LinearAlgebra.dot(repeat(ins.C_no_load',T) , α)+ LinearAlgebra.dot(repeat(ins.C_marginal',T),  ρ) + LinearAlgebra.dot(repeat(ins.C_startup',T),  γ ) )
  
    
	
	set_silent(model)
	optimize!(model)

	if dual_status(model) == NO_SOLUTION
		return 0
	end

	αV  = zeros(G,T) 
	γV  = zeros(G,T)
	ηV  = zeros(G,T)
	ρV  = zeros(G,T)
	αRC = zeros(G,T) 
	γRC = zeros(G,T)
	ηRC = zeros(G,T)
	ρRC = zeros(G,T)

    load_balance_v = dual.(load_balance)
    reserve_v = dual.(reserve)
    power_output_lower_bounds_v = dual.(power_output_lower_bounds)
    power_output_upper_bounds_v = dual.(power_output_upper_bounds)
    ramp_up_bounds_v = dual.(ramp_up_bounds)
    ramp_down_bounds_v = dual.(ramp_down_bounds)
    minimum_uptime_v = dual.(minimum_uptime)
    minimum_downtime_v = dual.(minimum_downtime)
    logistical_constraints_1_v = dual.(logistical_constraints_1)
    logistical_constraints_2_v = dual.(logistical_constraints_2)

    load_balance_sp = shadow_price.(load_balance)
    reserve_sp = shadow_price.(reserve)
    power_output_lower_bounds_sp = shadow_price.(power_output_lower_bounds)
    power_output_upper_bounds_sp = shadow_price.(power_output_upper_bounds)
    ramp_up_bounds_sp = shadow_price.(ramp_up_bounds)
    ramp_down_bounds_sp = shadow_price.(ramp_down_bounds)
    minimum_uptime_sp = shadow_price.(minimum_uptime)
    minimum_downtime_sp = shadow_price.(minimum_downtime)
    logistical_constraints_1_sp = shadow_price.(logistical_constraints_1)
    logistical_constraints_2_sp = shadow_price.(logistical_constraints_2)
	
	for g in 1:G
		for t in 1:T
			αV[g , t ] = value(α[g , t ])
			γV[g , t ] = value(γ[g , t ])
			ηV[g , t ] = value(η[g , t ])
			ρV[g , t ] = value(ρ[g , t ])
			αRC[g , t ]  = reduced_cost(α[g , t ])
			γRC[g , t ] = reduced_cost(γ[g , t ])
			ηRC[g , t ] = reduced_cost(η[g , t ])
			ρRC[g , t ] = reduced_cost(ρ[g , t ])
		end
	end

	return JuMP.objective_value(model), [load_balance_v..., reserve_v...],αV,  γV, ηV, ρV, αRC, γRC,ηRC,ρRC, power_output_lower_bounds_v, power_output_upper_bounds_v, ramp_up_bounds_v, ramp_down_bounds_v, minimum_uptime_v, minimum_downtime_v, logistical_constraints_1_v, logistical_constraints_2_v, load_balance_sp, reserve_sp, power_output_lower_bounds_sp, power_output_upper_bounds_sp, ramp_up_bounds_sp, ramp_down_bounds_sp, minimum_uptime_sp, minimum_downtime_sp, logistical_constraints_1_sp, logistical_constraints_2_sp
end