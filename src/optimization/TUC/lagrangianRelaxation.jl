function solve_SP(uc::TUC,λD;dinamic_programming::Bool=false)
    p=zeros(uc.I, nT(uc))
    u=zeros(uc.I, nT(uc))
    v = sum(uc.D .* λD) 
    for i in 1:uc.I
        v_i,p_i,u_i = dinamic_programming ? construct_i_graph(uc,i,λD) : solve_SP(uc,i,λD)
        v -= v_i
        p[i,:] = p_i
        u[i,:] = u_i
   end
   return v,u,p
end


function solve_SP(uc::TUC, i::Int64, λD)
	ρd, ρu = uc.MinPower[i], uc.MaxPower[i]
	Δu, Δd = uc.DeltaRampUp[i], uc.DeltaRampDown[i]
	a, b, c = uc.QuadTerm[i,:], uc.LinearTerm[i, :], uc.ConstTerm[i, :]
	ρ0, τ0 = uc.InitialPower[i], uc.InitUpDownTime[i]
	τu, τd = uc.MinUpTime[i], uc.MinDownTime[i]
	s = uc.StartUpCost[i]

	T = nT(uc)
	model = Model(LR_Optimiser)
	
	set_attribute(model, "Method", 0)

	set_silent(model)
	@variable(model, ρu >= p[1:T] >= 0.0)
	@variable(model, 1 >= u[1:T] >= 0, Bin)
	@variable(model, 1 >= v[1:T] >= 0, Bin)
	@variable(model, 1 >= w[1:T] >= 0, Bin)


	@constraint(model, link_uvw[t = 2:T], u[t] - u[t-1] == v[t] - w[t])
	@constraint(model, link_uvw0, u[1] - (τ0 > 0 ? 1 : 0) == v[1] - w[1])
	@constraint(model, rampU[t = 1:T-1], p[t+1] - p[t] <= u[t] * Δu + (1 - u[t]) * ρu)
	@constraint(model, rampD[t = 1:T-1], p[t] - p[t+1] <= u[t+1] * Δd + (1 - u[t+1]) * ρd)
	@constraint(model, rampU0, p[1] - ρ0 <= (τ0 > 0 ? Δu : ρu))
	@constraint(model, rampD0, ρ0 - p[1] <= u[1] * Δd + (1 - u[1]) * ρd)
	@constraint(model, link_up_up[t = 1:T], ρu * u[t] >= p[t])
	@constraint(model, link_up_down[t = 1:T], p[t] >= ρd * u[t])
	@constraint(model, timeUp[t = Int64.(τu:T)], sum(v[r] for r in Int64.((t-τu+1):(t))) <= u[t])
	@constraint(model, timeDown[t = Int64.(τd:T)], sum(w[r] for r in Int64.((t-τd+1):(t))) <= 1 - u[t])
	if τu > τ0 > 0
		@constraint(model, initial[t = Int64.(1:(τu-τ0))], u[t] == 1)
	else
		if 0<-τ0 < τd
			@constraint(model, initial[t = Int64.(1:(τd+τ0))], u[t] == 0)
		end
	end


	@objective(model, Min, sum(p .* p .* a) + LinearAlgebra.dot(p, b .- λD) + sum(c[t] * u[t] for t in 1:T) + sum(s * v[t] for t in 1:T))
	optimize!(model)
	return objective_value(model), value.(u), value.(p)
end

#function solve_SP(uc::TUC,)
#	value=solve_SP(uc::TUC,i::Int64,λD)
#	return

function get_v_w(uc, u, i=1)
	ρ0, τ0 = uc.InitialPower[i], uc.InitUpDownTime[i]
	v, w = zeros(nT(uc)), zeros(nT(uc))
	if τ0 > 0 && u[1] == 0
		w[1] = 1
	end
	if τ0 < 0 && u[1] == 1
		v[1] = 1
	end
	for i in 2:nT(uc)
		if u[i] < u[i-1]
			w[i] = 1
		end
		if u[i] > u[i-1]
			v[i] = 1
		end
	end
	return v, w
end






function check_feasibility(uc, i, u, p)
    v,w=get_v_w(uc,u)
	ρd, ρu = uc.MinPower[i], uc.MaxPower[i]
	Δu, Δd = uc.DeltaRampUp[i], uc.DeltaRampDown[i]
	a, b, c = uc.QuadTerm[i,:], uc.LinearTerm[i, :], uc.ConstTerm[i, :]
	ρ0, τ0 = uc.InitialPower[i], uc.InitUpDownTime[i]
	τu, τd = uc.MinUpTime[idx], uc.MinDownTime[idx]
	s = uc.StartUpCost[idx]

	T = nT(uc)

	println("Check Variable Bounds: ")
	for t in 1:T
		if !(ρu >= p[t] >= 0.0)
			println("Error in bounds for p at t=", t)
		end
		if !(1 >= u[t] >= 0.0)
			println("Error in bounds for u at t=", t)
		end
		if !(1 >= v[t] >= 0.0)
			println("Error in bounds for v at t=", t)
		end
		if !(1 >= w[t] >= 0.0)
			println("Error in bounds for w at t=", t)
		end
	end

	println("Check Linking Constraints: ")
	if !(u[1] - (τ0 > 0 ? 1 : 0) <= v[1] - w[1])
		println("Error in linking constraints at t=1")
	end
	for t in 2:T
		if !(u[t] - u[t-1] <= v[t] - w[t])
			println("Error in linking constraints at t=", t)
		end
	end



	println("Check Ramping Constraints: ")
	if !(p[1] - ρ0 <= (τ0 > 0 ? Δu : ρu))
		println("Error in ramping-up constraints at t=0")
	end
	if !(ρ0 - p[1] <= u[1] * Δd + (1 - u[1]) * ρd)
		println("Error in ramping-down constraints at t=0")
	end
	for t in 1:T-1
		if !(p[t+1] - p[t] <= u[t] * Δu + (1 - u[t]) * ρu)
			println("Error in ramping-up constraints at t=", t)
		end
		if !(p[t] - p[t+1] <= u[t+1] * Δd + (1 - u[t+1]) * ρd)
			println("Error in ramping-down constraints at t=", t)
		end
	end

	println("Check Link-Up-Down Constraints: ")
	for t in 1:T
		if !(ρu * u[t] >= p[t])
			println("Error in link-up constraints at t=", t)
		end
		if !(p[t] >= ρd * u[t])
			println("Error in link-down constraints at t=", t)
		end
	end


	println("Check Time-Up Constraints: ")
	for t in Int64.(τu:T)
		if !(sum(v[r] for r in Int64.((t-τu+1):(t))) <= u[t])
			println("Error in time-up constraints at t=", t)
		end
	end
	println("Check Time-Down Constraints: ")
	for t in Int64.(τd:T)
		if !(sum(w[r] for r in Int64.((t-τd+1):(t))) <= 1 - u[t])
			println("Error in time-down constraints at t=", t)
		end
	end



	println("Check Initialization Constraints: ")
	if τu > τ0 > 0
		for t in Int64.(1:(τu-τ0))
			if !(u[t] == 1)
				println("Error, u[", t, "] must be equal to one")
			end
		end
	else
		if -τ0 < τd
			for t in Int64.(1:(τd-τ0))
				if !(u[t] == 0)
					println("Error, u[", t, "] must be equal to zero")
				end
			end
		end
	end
end





function objective_SP(uc::TUC, i::Int64, λD, u,p)
    v,w=get_v_w(uc,u)
    ρd, ρu = uc.MinPower[i], uc.MaxPower[i]
	Δu, Δd = uc.DeltaRampUp[i], uc.DeltaRampDown[i]
	a, b, c = uc.QuadTerm[i,:], uc.LinearTerm[i, :], uc.ConstTerm[i, :]
	ρ0, τ0 = uc.InitialPower[i], uc.InitUpDownTime[i]
	τu, τd = uc.MinUpTime[idx], uc.MinDownTime[idx]
    return sum(p .* p .* a) + LinearAlgebra.dot(p, b - λD) + sum(c[t] * u[t] for t in 1:T) + sum(s * v[t] for t in 1:T)
end