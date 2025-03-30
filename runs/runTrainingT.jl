using BundleNetworks, Instances, Statistics, Flux, LinearAlgebra, JuMP, JSON
using ArgParse

using BundleNetworks
using Instances
using Flux
using Random
using JSON
using Statistics
using BSON: @save, @load
using TensorBoardLogger, Logging
#using CUDA

include("../runs/readingFunctions.jl")

function ep_train_and_val(
	folder,
	directory,
	dataset,
	idxs_train,
	idxs_val,
	opt;
	maxEP = 10,
	maxIT = 50,
	lt = BundleNetworks.RnnTModelfactory(),
	t_strat = BundleNetworks.constant_t_strategy(),
	t = 0.000001,
	nn = create_NN(lt),
	test_mode = false,
	retrain_mode = false,
	location = "./results/",
	cr_init::Bool = false,
	exactGrad = true,
	telescopic = false,
	γ = 0.1,
	use_gold = true,
	instance_features = false,
	seed = 1,
	single_prediction::Bool = false,
)


	BundleNetworks.device = cpu
	device = cpu

	format = split(directory[1], ".")[end]


	lt = BundleNetworks.RnnTModelfactory()


	f = open(location * "dataset.json", "w")
	JSON.print(f, Dict("train" => directory[idxs_train], "validation" => directory[idxs_val], "test" => [directory[i] for i in eachindex(directory) if !(i in idxs_val) && !(i in idxs_train)]))
	close(f)



	res_values_train, res_times_train = [], []
	res_values_val, res_times_val = [], []
	res = Dict()
	res_val = Dict()
	for epoch in 1:maxEP
		res[epoch] = Dict()
		for ins_path in directory[idxs_train]
			res[epoch][ins_path] = Dict()
		end
	end

	for epoch in 1:maxEP
		res_val[epoch] = Dict()
		for ins_path in directory[idxs_val]
			f = split(split(ins_path, "/")[end], ".")[1]
			res_val[epoch][ins_path] = Dict()
		end
	end
	maxEP = test_mode ? 1 : maxEP
	test_mode = retrain_mode ? true : test_mode
	nn_copy = nn
	nn_best = deepcopy(nn)
	obj_best = 0.0

	crs = Dict()
	cr_duals = Dict()
	for (idx, ϕ) in dataset
		inst = ϕ.inst
		if !(cr_init)
			cr_duals[idx] = zeros((sizeK(inst), sizeV(inst)))
			crs[idx] = 1.0
		else
			crs[idx], cr_duals[idx] = CR(inst)[1:2]

		end
		GC.gc()
	end
	rng = Random.MersenneTwister(seed)

	state = Flux.setup(opt, nn)

	lg = TBLogger(location, min_level = Logging.Info)
	with_logger(lg) do
		for epoch in 1:maxEP
			values, times = Float64[], Float64[]
			shuffle!(rng, idxs_train)
			shuffle!(rng, idxs_val)
			for idx_t in idxs_train
				ins_path, ϕ = dataset[idx_t]
				B = initializeBundle(tLearningBundleFactory(), ϕ, t, cr_duals[ins_path], lt, nn, maxIT; exactGrad, instance_features)
				B.params.maxIt = maxIT

				t0 = time()
				co = BundleNetworks.train!(B, ϕ, state; γ, samples = 1, normalization_factor = 1.0, telescopic, single_prediction)

				append!(times, time() - t0)
				append!(values, ϕ(reshape(BundleNetworks.zS(B), (sizeK(ϕ.inst), sizeV(ϕ.inst)))) * ϕ.rescaling_factor)

				nn = B.nn

				res[epoch][ins_path]["time"] = time() - t0
				res[epoch][ins_path]["obj"] = values[end]
				res[epoch][ins_path]["optimality"] = co
				GC.gc()
			end
			push!(res_values_train, [mean(values), std(values), quantile(values)...])
			push!(res_times_train, [mean(times), std(times), quantile(times)...])

			values, times = Float64[], Float64[]
			for idx_i in idxs_val
				println("Validation Instance")
				ins_path, ϕ = dataset[idx_i]
				B = initializeBundle(tLearningBundleFactory(), ϕ, t, cr_duals[ins_path], lt, nn, 5 * maxIT + 1; exactGrad, instance_features)

				B.params.maxIt = 5 * maxIT

				t0 = time()
				#co = solve!(B, ϕ; t_strat)
				co = BundleNetworks.Bundle_value_gradient!(B, ϕ, false, single_prediction)
				println()
				append!(times, time() - t0)
				append!(values, maximum(B.all_objs[1:maxIT+1]))
				nn = nn_copy

				f = ins_path
				res_val[epoch][f]["time"] = time() - t0
				res_val[epoch][f]["obj maxIT"] = maximum(B.all_objs[1:maxIT+1]) * ϕ.rescaling_factor
				res_val[epoch][f]["obj 2*maxIT"] = maximum(B.all_objs[1:2*maxIT+1]) * ϕ.rescaling_factor
				res_val[epoch][f]["obj 5*maxIT"] = maximum(B.all_objs[1:end]) * ϕ.rescaling_factor
				res_val[epoch][f]["optimality"] = co
				GC.gc()
			end
			push!(res_values_val, [mean(values), std(values), quantile(values)...])
			push!(res_times_val, [mean(times), std(times), quantile(times)...])

			objV = mean([res_val[epoch][f]["obj maxIT"] for f in keys(res_val[epoch])])
			objVx2 = mean([res_val[epoch][f]["obj 2*maxIT"] for f in keys(res_val[epoch])])
			objVx5 = mean([res_val[epoch][f]["obj 5*maxIT"] for f in keys(res_val[epoch])])



			if use_gold
				GAP_t = mean([abs(res[epoch][f]["obj"] - gold[f]) / gold[f] for f in keys(res[epoch])]) * 100
				GAP_v = mean([abs(res_val[epoch][f]["obj maxIT"] - gold[f]) / gold[f] for f in keys(res_val[epoch])]) * 100
				GAP_vx5 = mean([abs(res_val[epoch][f]["obj 5*maxIT"] - gold[f]) / gold[f] for f in keys(res_val[epoch])]) * 100
				GAP_vx2 = mean([abs(res_val[epoch][f]["obj 2*maxIT"] - gold[f]) / gold[f] for f in keys(res_val[epoch])]) * 100

				@info " Train " GAP_percentage = GAP_t log_step_increment = 0
				@info " Validation " GAP_percentage = GAP_v log_step_increment = 0
				@info " Validation x5" GAP_percentage = GAP_vx5 log_step_increment = 0
				@info " Validation x2" GAP_percentage = GAP_vx2 log_step_increment = 0
			end
			if obj_best < objV
				obj_best = objV
				nn_best = deepcopy(nn)
			end

			@info " Train " LSP = res_values_train[end][1] log_step_increment = 0
			@info " Validation x5 " LSP = objVx5 log_step_increment = 0
			@info " Validation " LSP = objV log_step_increment = 1

		end
	end
	@save (location * "nn.bson") nn
	@save (location * "nn_bestLV.bson") nn_best


	saveJSON(location * "res_train.json", res)


	f = open(location * "res_values_train.json", "w")
	JSON.print(f, res_values_train)
	close(f)

	f = open(location * "res_times_train.json", "w")
	JSON.print(f, res_times_train)
	close(f)

	saveJSON(location * "res_val.json", res_val)

	f = open(location * "res_values_val.json", "w")
	JSON.print(f, res_values_val)
	close(f)

	f = open(location * "res_times_val.json", "w")
	JSON.print(f, res_times_val)
	close(f)
	return #nn, res, res_values_train, res_times_train,res_val,res_values_val,res_times_val
end


function saveJSON(name, res)
	f = open(name, "w")
	JSON.print(f, res)
	close(f)
end




function main(args)
	s = ArgParseSettings("Training an unrolling model" *
						 "version info, default values, " *
						 "options with types, variable " *
						 "number of arguments.",
		version = "Version 1.0", # version info
		add_version = true)      # audo-add version option

	@add_arg_table! s begin
		"--data"
		arg_type = String
		default = "ADAM"
		help = "optimizer for the training"
		"--lr"
		required = true
		arg_type = Float32
		help = "learning rate"
		"--cn"
		default = -1
		arg_type = Int64
		help = "Clip Norm"
		"--mti"
		required = true
		arg_type = Int64
		help = "Maximum number of training instances"
		"--mvi"
		required = true
		arg_type = Int64
		help = "Maximum number of validation instances"
		"--seed"
		required = true
		arg_type = Int64
		help = "Random seed"
		"--maxIT"
		required = true
		arg_type = Int64
		help = "Maximum number of Unrolled iteration."
		"--maxEP"
		required = true
		arg_type = Int64
		help = "Maximum number of training epochs."
		"--cr_init"
		arg_type = Bool
		default = false
		help = "If true we start from the dual variables of the continuous relaxation otherwhise from zero"
		"--exactGrad"
		arg_type = Bool
		default = true
		help = "If true we use a more accurate formula for the gradient of the solution of the MP w.r.t. the step size t otherwhise zero"
		"--telescopic"
		arg_type = Bool
		default = false
		help = "When true, the loss will consider the Lagrangian Sub-Problem value of all the point visited during the execution. If false only the final point."
		"--gamma"
		arg_type = Float32
		default = 0.1
		help = "When is used the telescopic sum, this is the parameter used to weight the different points as gamma^{iteration}."
		"--use_gold"
		arg_type = Bool
		default = true
		help = "When true the gold labels should be provided to print the GAP in the Tensorboard Plots."
		"--gold_location"
		arg_type = String
		default = "../golds/MCNDforTest/gold.json"
		help = "Location of the gold labels."
		"--instance_features"
		arg_type = Bool
		default = true
		help = "When truestatic features related to LR are added as input to the neural network model."
		"--single_prediction"
		arg_type = Bool
		default = false
		help = "Use constant t parameter, provided at the beginning of the training."
		"--dataset_location"
		arg_type = String
		default = "-1"
		help = "Location of the gold labels."
	end

	# take the input parameters and construct a Dictionary
	parsed_args = parse_args(args, s)

	folder = parsed_args["data"]
	lr = parsed_args["lr"]
	cn = parsed_args["cn"]
	mti = parsed_args["mti"]
	mvi = parsed_args["mvi"]
	seed = parsed_args["seed"]
	maxIT = parsed_args["maxIT"]
	maxEP = parsed_args["maxEP"]
	cr_init = parsed_args["cr_init"]
	exactGrad = parsed_args["exactGrad"]
	telescopic = parsed_args["telescopic"]
	use_gold = parsed_args["use_gold"]
	gold_location = parsed_args["gold_location"]
	γ = parsed_args["gamma"]
	instance_features = parsed_args["instance_features"]
	single_prediction = parsed_args["single_prediction"]
	dataset_location = parsed_args["dataset_location"]
	telescopic = γ == 0.0 ? false : true

	directory = readdir(folder)

	if cn > 0
		opt = Flux.OptimiserChain(Flux.Optimise.Adam(lr), ClipNorm(cn))
	else
		opt = Flux.OptimiserChain(Flux.Optimise.Adam(lr))
	end


	rng = Random.MersenneTwister(seed)
	shuffle!(rng, directory)


	idxs_train = collect(1:mti)
	idxs_val = collect((mti+1):(mti+mvi))
	datasets = Dict()
	if !(dataset_location == "-1")
		f = JSON.open(dataset_location, "r")
		datasets = JSON.parse(f)
		close(f)
	else
		datasets["training"] = directory[1:(mti)]
		datasets["validation"] = directory[(mti+1):(mti+mvi)]
	end
	dataset = []
	gold = Dict()
	 format = split(directory[1], ".")[end]


	if format == "dat"
		tmp_idx = 0
		for set in [datasets["training"], datasets["validation"]]
			for f in set
				ins = my_read_dat(folder * f)
				ϕ = BundleNetworks.constructFunction(ins, 1.0)
				_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeK(ins), sizeV(ins)))
				ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
				push!(dataset, (f, ϕ))
				tmp_idx += 1
				if tmp_idx % 100 == 0
					GC.gc()
				end
			end
		end
		gold_location = "./golds/" * split(folder, "/")[end-1] * "/gold.json"
		f = JSON.open(gold_location, "r")
		gold = JSON.parse(f)
		close(f)

	else
		tmp_idx = 0
		for set in [datasets["training"], datasets["validation"]]
			for f in set
				ins, Ld = my_read_dat_json(folder * f)
				ϕ = BundleNetworks.constructFunction(ins, 1.0)
				_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeK(ins), sizeV(ins)))
				ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
				push!(dataset, (f, ϕ))
				gold[f] = Ld
				tmp_idx += 1
				if tmp_idx % 100 == 0
					GC.gc()
					# CUDA.reclaim()
					# CUDA.free_memory()
				end
			end
		end
	end
	mti,mvi=length(datasets["training"]),length(datasets["validation"])

	res_folder =
		"res_goldLossWeights_" * (instance_features ? "with" : "without") * "InstFeat_init" * (cr_init ? "CR" : "Zero") * "_lr" * string(lr) * "_cn" * string(cn) * "_maxIT" * string(maxIT) * "_maxEP" * string(maxEP) * "_data" *
		string(split(folder, "/")[end-1]) * "_exactGrad" * string(exactGrad) * "_gamma" * string(γ) * "_seed" * string(seed) * "_single_prediction" * string(single_prediction)
	sN = sum([1 for j in readdir("res") if contains(j, res_folder)]; init = 0.0)
	location = "res/" * res_folder * "_" * string(sN) * "_" * "/"
	mkdir(location)
	a = ep_train_and_val(folder, directory, dataset, idxs_train, idxs_val, opt; maxIT, maxEP, location, cr_init, exactGrad, telescopic, γ, use_gold, instance_features, seed, single_prediction)
end


main(ARGS)
