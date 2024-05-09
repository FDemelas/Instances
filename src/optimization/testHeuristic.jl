using JuMP,HiGHS, Instances, LinearAlgebra


"""
CR(ins)

# Arguments:
- `ins`: an instance structure.

Solve the linear relaxation of the provided instance and then, it returns the objective value of the linear relaxation, the dual variables associated to flow constraints, the dual variables associated to capacity constraints, the primal flow variables and the primal design variables.
"""
function CR(ins::abstractInstanceMCND,y_fixed::AbstractVector)
	nE = sizeE(ins)
	nK = sizeK(ins)
	nV = sizeV(ins)

	G = zeros(nK, nV, nE)
	C = zeros(nK, nE)
	
	M = 10 * sizeK(ins) * sizeE(ins) * (maximum(ins.r)+maximum(ins.f))
	weight = ones(nE, nK)
	for ij in 1:nE
		for k in 1:nK
			G[k, head(ins, ij), ij] = -1 * isInKij(ins, k, ij)
			G[k, tail(ins, ij), ij] = 1 * isInKij(ins, k, ij)
			C[k, ij] = (routing_cost(ins, ij, k) + (y_fixed[ij]  ? 0  : fixed_cost(ins,ij))/capacity(ins, ij) )* (y_fixed[ij]  ? 1  : M )
		end
		for k in 1:nK
			weight[ij, k] = isInKij(ins, k, ij)
		end
	end

	model = Model(HiGHS.Optimizer)

	@variable(model, min(ins.K[k][3], capacity(ins, ij)) >= x[k = 1:nK, ij = 1:nE] >= 0)
	@constraint(model, cap_constr[ij = 1:nE], weight[ij, :]' * x[:, ij] <= capacity(ins, ij))
	
	@constraint(model, flow[i = 1:nV, k = 1:nK], G[k, i, :]' * x[k, :] == b(ins, i, k))

	@objective(model, Min, LinearAlgebra.dot(C, x) )
	
	set_silent(model)
	optimize!(model)

	if dual_status(model) == NO_SOLUTION
		return 0
	end

	πV = zeros(nK, nV)
	xV = zeros(nK, nE)

	for i in 1:nV
		for k in 1:nK
			πV[k, i] = dual(flow[i, k])
		end
	end

	for i in 1:nE
		for k in 1:nK
			xV[k, i] = value(x[k, i])
		end
	end
	return JuMP.objective_value(model), -πV, xV
end

function get_heuristic_value(ins,z)
	objLR, _, y = LR(ins,z)
	obj, _,x = CR(ins, Bool.(y))    
	y=ceil.(sum(x,dims=1) .> 0)
    ub = objective_value(ins,x,y)
	println(obj," ",ub)
    return objLR,ub
end

function objective_value(ins,x,y)
	return sum(ins.r .* x) + sum(ins.f .*y)
end


ins_folder="/media/francesco/Kali/ESPERIMENTI_IMPORTANTI_ICML/WarmstartingSMS++/test_dat/"
z_folder="/media/francesco/Kali/ESPERIMENTI_IMPORTANTI_ICML/WarmstartingSMS++/test_labels/"
gold_folder="/media/francesco/Kali/ESPERIMENTI_IMPORTANTI_ICML/WarmstartingSMS++/test_goldV/"


directory=readdir(ins_folder)
pred_v,ceil_cr,cr_v,zr_v,golds=[],[],[],[],[],[],[],[]
for f in directory
	ins=read_dat(ins_folder*f,Instances.cpuMCNDinstanceFactory())
	file=open(z_folder*f)
	lines=readlines(file)
	close(file)
	z_vs = map(x->parse.(Float32,split(x," ";keepempty=false)),lines)
	z = [z_vs[i][j] for i in 1:sizeK(ins), j in 1:sizeV(ins)]
	push!(pred_v, get_heuristic_value(ins,z))

	obj,z,_,x,y,_ = Instances.CR(ins)
	push!( ceil_cr , objective_value(ins,x,y))
	push!( cr_v , get_heuristic_value(ins,z))

	push!(zr_v , get_heuristic_value(ins,zeros(size(z))))

	file = open(gold_folder*f)
	v = parse(Float32,readlines(file)[1])
	close(file)
	push!(golds,v)
end


g=zeros(sizeV(ins),sizeV(ins))
for e in 1:sizeE(ins)
	g[tail(ins,e),head(ins,e)]=1
end
colors=[1/2*mean( [ y*Int(origin(ins,k) == i)[1] for k in 1:sizeK(ins)])+ 1/2*mean( [ r*Int(destination(ins,k) == i)[1] for k in 1:sizeK(ins)]) for i in 1:sizeV(ins)]

graphplot(g,
names=1:sizeV(ins),
markercolor=colors,
curvature_scalar=0.1)

y=colorant"yellow"
r=colorant"red"
b=colorant"blue"
