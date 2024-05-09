"""
CR(ins)

# Arguments:
- `ins`: an instance structure.

Solve the linear relaxation of the provided instance and then, it returns the objective value of the linear relaxation, the dual variables associated to flow constraints, the dual variables associated to capacity constraints, the primal flow variables and the primal design variables.
"""
function CR(ins::abstractInstanceMCND)
	nE = sizeE(ins)
	nK = sizeK(ins)
	nV = sizeV(ins)

	G = zeros(nK, nV, nE)
	C = zeros(nK, nE)
	F = zeros(nE)

	weight = ones(nE, nK)
	for ij in 1:nE
		F[ij] = fixed_cost(ins, ij)
		for k in 1:nK
			G[k, head(ins, ij), ij] = -1 * isInKij(ins, k, ij)
			G[k, tail(ins, ij), ij] = 1 * isInKij(ins, k, ij)
			C[k, ij] = routing_cost(ins, ij, k)
		end
		for k in 1:nK
			weight[ij, k] = isInKij(ins, k, ij)
		end
	end


	model = Model(CR_Optimiser)

	@variable(model, min(ins.K[k][3], capacity(ins, ij)) >= x[k = 1:nK, ij = 1:nE] >= 0)

	@variable(model, 1 >= y[1:nE] >= 0)#,Bin)

	@constraint(model, flow[i = 1:nV, k = 1:nK], G[k, i, :]' * x[k, :] == b(ins, i, k))

	@constraint(model, cap_constr[ij = 1:nE], weight[ij, :]' * x[:, ij] <= capacity(ins, ij) * y[ij])

	@objective(model, Min, LinearAlgebra.dot(C, x) + LinearAlgebra.dot(F, y))


	set_silent(model)
	optimize!(model)

	if dual_status(model) == NO_SOLUTION
		return 0
	end

	πV = zeros(nK, nV)
	xV = zeros(nK, nE)
	yV = zeros(nE)
	μV = zeros(nE)
	xRC = zeros(nK, nE)
	yRC = zeros(nE)
	πSP = zeros(nK, nV)
	μSP = zeros(nE)

	for i in 1:nV
		for k in 1:nK
			πV[k, i] = dual(flow[i, k])
			πSP[k, i] = shadow_price(flow[i, k])
		end
	end

	for e in 1:nE
		μV[e] = dual(cap_constr[e])
		μSP[e] = shadow_price(cap_constr[e])
	end

	for i in 1:nE
		for k in 1:nK
			xV[k, i] = value(x[k, i])
			xRC[k, i] = reduced_cost(x[k, i])
		end
		yV[i] = value(y[i])
		yRC[i] = reduced_cost(y[i])
	end
	return JuMP.objective_value(model), -πV, μV, xV, yV, -πSP, μSP, xRC, yRC
end
