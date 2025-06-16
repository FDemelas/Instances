function on_index(t::Int64)
	return 1 + t * 2
end

function off_index(t::Int64)
	return 1 + t * 2 - 1
end

function is_off_index(i::Int64)
	return (i % 2 == 0) ? true : false
end

function is_on_index(i::Int64)
	return !is_off_index(i)
end

function dag_shortest_path(g::DiGraph, weights, source::Int)
	n = nv(g)
	top_order = collect(minimum([e.src for e in collect(edges(g))]):maximum([e.src for e in collect(edges(g))]))#topological_sort(g)
	dist = fill(Inf, n)
	pred = fill(-1, n)
	dist[source] = 0.0

	for u in top_order
		for v in outneighbors(g, u)
			w = weights[(u, v)]
			if dist[v] > dist[u] + w
				dist[v] = dist[u] + w
				pred[v] = u
			end
		end
	end

	return dist, pred
end

function construct_i_graph(uc::TUC, idx::Int64, λD)
	ρd, ρu = uc.MinPower[idx], uc.MaxPower[idx]
	Δu, Δd = uc.DeltaRampUp[idx], uc.DeltaRampDown[idx]
	a, b, c = uc.QuadTerm[idx, :], uc.LinearTerm[idx, :], uc.ConstTerm[idx, :]
	s = uc.StartUpCost[idx]
	ρ0, τ0 = uc.InitialPower[idx], uc.InitUpDownTime[idx]
	τu, τd = uc.MinUpTime[idx], uc.MinDownTime[idx]

	# the first node is the source
	ss = 1
	# the following 2*nT(uc) nodes are the one for OFF_t and ON_t
	# in such a way that OFF_t is in the position 1+2*t-1
	# and ON_t in the position 1+2*t
	nodes = collect(1:(2*nT(uc)+2))
	graph = DiGraph(2 * nT(uc) + 2)
	# the last node is the destination node
	d = 2 * nT(uc) + 2
	# the array containing the arcs indeces (couples of source and destination)
	arcs = Tuple{Int64, Int64}[]
	# the array containing the weight of the correspondent arc
	weights = Dict()
	sols = Dict()

	# creating ON_i -> OFF_j arcs
	# indicating that the unit is up from i to j-1 and down in j
	for i in 1:(nT(uc)-1)
		# get the node index for ON_i 
		idx_i = on_index(i)
		for j in Int64(i + τu):nT(uc)
			# get the node index for OFF_j
			idx_j = off_index(j)
			# create the arc only if i<j and if
			# the unit is active for more than τu time periods
			if j - i >= τu
				# add the arc
				arc = (idx_i, idx_j)
				add_edge!(graph, arc)
				# add the weight, in this case it is 
				# the cost of keeping the unit open 
				# plus the objective ue of the
				# Economic Dispatch problem 
				val, p = ED(uc, idx, i, j - 1, λD)

				weights[arc] = sum(c[i:(j-1)]) + val
				sols[arc] = p
			end
		end
	end

	#creating the OFF_i -> ON_j arcs
	# down between i and j-1 and open in j
	for i in 1:(nT(uc)-1)
		# get the node index for OFF_i
		idx_i = off_index(i)
		for j in Int64(i + τd):nT(uc)
			# get the node index for ON_j
			idx_j = on_index(j)
			# create the arc only if the new position index is
			# greater than the old position one and is
			# the unit is active for more than τd time periods
			if j - i >= τd
				# add the arc
				arc = (idx_i, idx_j)
				add_edge!(graph, arc)
				# in this case the weight is simply 
				# the start-up cost for the unit
				weights[arc] = s
				sols[arc] = zeros(j - i - 1)
			end
		end
	end


	#creating the ON_i -> d arcs
	for i in 1:(nT(uc))
		# get the node index of ON_i
		idx_i = on_index(i)
		# add the arc
		arc = (idx_i, d)
		add_edge!(graph, arc)
		# in this case the weight is the cost of keeping the unit
		# working for the period and the optimal value of the Economic Dispatch problem
		val, p = ED(uc, idx, i, nT(uc), λD)
		weights[arc] = sum(c[i:end]) + val
		sols[arc] = p
	end


	#creating the OFF_i -> d arcs
	for i in 1:nT(uc)
		# get the node index of OFF_i
		idx_i = off_index(i)
		# add the arc
		arc = (idx_i, d)
		add_edge!(graph, arc)
		# in this case the weight is zero
		weights[arc] = 0
		sols[arc] = zeros(nT(uc) - i + 1)
	end



	#creating the arc starting to s accordin to the initial conditions
	if τ0 > 0
		#  minimum number of time instants that are necessary to bring the unit to the power level required to stop (t_ramp_min could be 0 if initial_power happens to be exactly the right power level)
		t_ramp_min = minimum([j for j in 0:nT(uc) if ρ0 -j * Δd <= ρd])
		# minimum index of the node from which it is possible to construct an arc
		min_node = (τ0 >= τu ? t_ramp_min : max(t_ramp_min, τu - τ0))
		if min_node == 0
			idx_i = off_index(1)
			arc = (ss, idx_i)
			add_edge!(graph, arc)
			weights[arc] = 0.0
			sols[arc] = []
			idx_i = on_index(1)
			arc = (ss, idx_i)
			add_edge!(graph, arc)
			weights[arc] = s
			sols[arc] = []
		end


		for i in Int64.(max(2,min_node+1):nT(uc))
			idx_i = off_index(i)
			val, p = ED(uc, idx, 1, i - 1, λD)
			arc = (ss, idx_i)
			add_edge!(graph, arc)
			weights[arc] = sum(c[1:(i-1)]) + val
			sols[arc] = p
		end
		#creating the s -> d arc
		val, p = ED(uc, idx, 1, nT(uc), λD)
		arc = (ss, d)
		add_edge!(graph, arc)
		weights[arc] = sum(c) + val
		sols[arc] = p

	else
		# minimum index of the node from which it is possible to construct an arc
		min_node = Int64(max(τd + τ0, 0))
		if min_node == 0
		    idx_i= off_index(1)
			arc = (ss, idx_i)
			add_edge!(graph,arc)
			weights[arc] = 0.0
			sols[arc] = []
		end

		for i in (min_node+1):nT(uc)
			idx_i = on_index(i)
			arc = (ss, idx_i)
			add_edge!(graph, arc)
			weights[arc] = s
			sols[arc] = zeros(i - 1)
		end
		#creating the s -> d arc
		arc = (ss, d)
		add_edge!(graph, arc)
		weights[arc] = 0.0
		sols[arc] = zeros(nT(uc))
	end

	dists, predecessors = dag_shortest_path(graph, weights, 1)
	path_dsp = []
	j = d
	while (j > 1)
		append!(path_dsp, j)
		j = predecessors[j]
	end
	append!(path_dsp, j)
	path_dsp = reverse(path_dsp)

	u_sol = zeros(nT(uc))
	p_sol = zeros(nT(uc))
	for i in eachindex(path_dsp[1:end-1])
		# and the ending node
		idx_ip1 = floor(Int64, (path_dsp[i+1]) / 2)
		# obtain the time index of the associated nodes composing the arcs
		if path_dsp[i] == ss
			# if we are considering an arcs of type source-> destination
			if path_dsp[i+1] == d
				# the solution will be all one if the unit was alreasy up
				if τ0 > 0
					u_sol .= 1
				else
					# otherwise the solution remains all zero
					u_sol .= 0
				end
			else
				# if we go from the source to an off node, then all the values are open and the last close
				if path_dsp[i+1] % 2 == 0
					u_sol[1:(idx_ip1-1)] .= 1
					u_sol[idx_ip1] = 0
				else
					# otherwise we go from the source to an on node, we open only the last
					u_sol[1:(idx_ip1-1)] .= 0
					u_sol[idx_ip1] = 1
				end
			end
		else
			# if we are not considering the source as starting node
			# if the head of the arc is the destination, then
			if path_dsp[i+1] == d
				# if we go from an off node to the destination, then all the values are 0
				if is_off_index(path_dsp[i])
					idx_i = floor(Int64, (path_dsp[i]) / 2)
					u_sol[idx_i:end] .= 0
				else
					idx_i = floor(Int64, (path_dsp[i]) / 2)
					u_sol[idx_i:end] .= 1
				end

				#otherwise we do nothing as the u_sol is already zero 
			else
				#otherwise we are considering an arc going through two nodes that are not source and destination
				idx_i = floor(Int64, (path_dsp[i]) / 2)
				if is_off_index(path_dsp[i+1])
					u_sol[idx_i:idx_ip1-1] .= 1
					u_sol[idx_ip1] = 0
				else
					u_sol[idx_i:idx_ip1-1] .= 0
					u_sol[idx_ip1] = 1
				end
			end
		end
	end

	ps = [Vector{Float32}(sols[(path_dsp[j], path_dsp[j+1])][:]) for j in eachindex(path_dsp[1:end-1])]
	p_sol = vcat(ps...)
	return dists[end], u_sol, p_sol
end

function ED(uc::TUC, i::Int64, h::Int64, k::Int64, λD)
	ρd, ρu = uc.MinPower[i], uc.MaxPower[i]
	Δu, Δd = uc.DeltaRampUp[i], uc.DeltaRampDown[i]
	a, b, c = uc.QuadTerm[i, h:k], uc.LinearTerm[i, h:k], uc.ConstTerm[i, h:k]
	ρ0, τ0 = uc.InitialPower[i], uc.InitUpDownTime[i]

	model = Model(LR_Optimiser)

	@variable(model, ρu >= p[h:k] >= ρd)
	@constraint(model, rampU[t = h:(k-1)], p[t+1] - p[t] <= Δu)
	@constraint(model, rampD[t = h:(k-1)], p[t] - p[t+1] <= Δd)

	if h == 1
		@constraint(model, rampU0, p[1] - ρ0 <= (τ0 > 0 ? Δu : ρu))
		@constraint(model, rampD0, ρ0 - p[1] <= Δd)
	else
		@constraint(model, rampU0, p[h] - 0.0 <= ρu)
		#@constraint(model, rampD0, 0.0 - p[h] <=  Δd)
	end
	if k != nT(uc)
		@constraint(model, rampDk, 0 - p[k] <= Δu)
		@constraint(model, rampUk, p[k] - 0 <= ρu)
	end
	@objective(model, Min, sum(p .* p .* a) + LinearAlgebra.dot(p, b - λD[h:k]))
	set_silent(model)
	optimize!(model)
	return objective_value(model), value.(p)
end
