

"""
CR(ins::cpuInstanceCWL)

# Arguments:
- `ins`: a CWL instance structure.

Solve the linear relaxation of the provided instance and then, it returns the objective value of the linear relaxation, the dual variables associated to packing constraints, the dual variables associated to capacity constraints, the primal assigment variables and the primal pack activation variables.
"""
function CR(ins::cpuInstanceCWL)

	I = ins.I
	J = ins.J

	model = Model(CR_Optimiser)

	@variable(model, 1 >= x[1:I, 1:J] >= 0)
	@variable(model, 1 >= y[1:I] >= 0)

	@constraint(model, packing[j = 1:J], sum(x[:, j]) == 1)
	@constraint(model, capacity[i = 1:I], LinearAlgebra.dot(ins.d, x[i, :]) <= ins.q[i] * y[i])

	@objective(model, Min, LinearAlgebra.dot(ins.f, y) + LinearAlgebra.dot(ins.c, x))

	optimize!(model)

	π = zeros(J)
	xV = zeros(I, J)
	yV = zeros(I)
	μ = zeros(I)
	xRC = zeros(I, J)
	yRC = zeros(I)
	πSP = zeros(J)
	μSP = zeros(I)

	for j in 1:J
		π[j] = dual(packing[j])
	end

	for i in 1:I
		μ[i] = dual(capacity[i])
	end

	for i in 1:I
		for j in 1:J
			xV[i, j] = value(x[i, j])
		end
		yV[i] = value(y[i])
	end

	for j in 1:J
		πSP[j] = shadow_price(packing[j])
	end

	for i in 1:I
		μSP[i] = shadow_price(capacity[i])
	end

	for i in 1:I
		for j in 1:J
			xRC[i, j] = reduced_cost(x[i, j])
		end
		yRC[i] = reduced_cost(y[i])
	end

	return objective_value(model), -π, μ, xV, yV, -πSP, μSP, xRC, yRC
end
