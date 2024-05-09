"""
CR(ins::cpuInstanceGA)

# Arguments:
- `ins`: a GA instance structure.

Solve the linear relaxation of the provided instance and then, it returns the objective value of the linear relaxation, the dual variables associated to packing constraints, the dual variables associated to capacity constraints, the primal assigment variables and the primal pack activation variables.
"""
function CR(ins::cpuInstanceGA)
	I = ins.I
	J = ins.J

	model = Model(CR_Optimiser)

	@variable(model, 1 >= x[1:I, 1:J] >= 0)

	@constraint(model, packing[i = 1:I], sum(x[i, :]) <= 1)
	@constraint(model, capacity[j = 1:J], LinearAlgebra.dot(ins.w[:,j], x[:, j]) <= ins.c[j])

	@objective(model, Max, LinearAlgebra.dot(ins.p, x))

	optimize!(model)

	π = zeros(I)
	xV = zeros(I, J)
	μ = zeros(J)
	xRC = zeros(I, J)
	πSP = zeros(I)
	μSP = zeros(J)

	for i in 1:I
		π[i] = dual(packing[i])
        πSP[i] = shadow_price(packing[i])
	end

	for i in 1:I
		for j in 1:J
			xV[i, j] = value(x[i, j])
            xRC[i, j] = reduced_cost(x[i, j])
		end
	end

	for j in 1:J
		μSP[j] = shadow_price(capacity[j])
        μ[j] = dual(capacity[j])
	end

	return objective_value(model), -π, μ, xV, πSP, μSP, xRC
end
