using BundleNetworks, Instances, Statistics, Flux, LinearAlgebra, JuMP, JSON
using ArgParse
using BSON: @load

include("./readingFunctions.jl")



function test_model(
	folder,
	directory,
	nn,
	test_idxs,
	maxIT = 100,
	t = 0.000001,
	lt = BundleNetworks.RnnTModelfactory(),
)


	BundleNetworks.device = cpu
	device = cpu

	format = split(directory[1], ".")[end]
	single_prediction = false

	res = Dict()
	for ins_path in test_idxs
		res[ins_path] = Dict()
	end

	dataset = Dict()
	gold = Dict()
	println(format)
	if format == "dat"
		tmp_idx = 0
		for f in test_idxs
			ins = my_read_dat(folder * f)
			ϕ = BundleNetworks.constructFunction(ins, 1.0)
			_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeK(ins), sizeV(ins)))
			ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
			dataset[f] = (f, ϕ)
			tmp_idx += 1
			if tmp_idx % 100 == 0
				GC.gc()
				#CUDA.reclaim()
				#CUDA.free_memory()
			end
		end
		gold_location = "./golds/" * split(folder, "/")[end-1] * "/gold.json"
		f = JSON.open(gold_location, "r")
		gold = JSON.parse(f)
		close(f)

	else
		tmp_idx = 0
		for f in test_idxs
			ins, Ld = my_read_dat_json(folder * f)
			ϕ = BundleNetworks.constructFunction(ins, 1.0)
			_, g = BundleNetworks.value_gradient(ϕ, zeros(sizeK(ins), sizeV(ins)))
			ϕ = BundleNetworks.constructFunction(ins, sqrt(sum(g .* g)))
			dataset[f] = (f, ϕ)
			gold[f] = Ld
			tmp_idx += 1
			if tmp_idx % 100 == 0
				GC.gc()
				# CUDA.reclaim()
				# CUDA.free_memory()
			end
		end

	end

	cr_init = false
	crs = Dict()
	cr_duals = Dict()
	for i in keys(dataset)
		idx, ϕ = dataset[i]
		inst = ϕ.inst
		if !(cr_init)
			cr_duals[idx] = zeros((sizeK(inst), sizeV(inst)))
			crs[idx] = 1.0
		else
			crs[idx], cr_duals[idx] = CR(inst)[1:2]

		end
		GC.gc()
	end


	for idx_i in test_idxs

		ins_path, ϕ = dataset[idx_i]
		B = BundleNetworks.initializeBundle(BundleNetworks.tLearningBundleFactory(), ϕ, t, cr_duals[ins_path], lt, deepcopy(nn), maxIT + 1; exactGrad = true, instance_features = true)

		B.params.maxIt = maxIT

		t0 = time()
		#co = solve!(B, ϕ; t_strat)
		co, timesD = solve!(B, ϕ; t_strat = BundleNetworks.nn_t_strategy(), unstable = false)
		#co = BundleNetworks.Bundle_value_gradient!(B, ϕ, false,single_prediction)
		println()
		#append!(times, time() - t0)
		#append!(values, maximum(B.all_objs[1:maxIT+1]))
		f = ins_path
		res[f]["time"] = time() - t0
		res[f]["objs"] = B.all_objs[1:end] * ϕ.rescaling_factor
		res[f]["times"] = timesD
		res[f]["gaps"] = [gap(i, gold[idx_i]) for i in res[f]["objs"]]
		GC.gc()
	end
	f = open("res_test_$(split(folder,"/")[end-1]).json", "w")
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
		required = true
		help = "path to the instance folder"
		"--model"
		arg_type = String
		required = true
		help = "path to the model"
		"--dataset"
		arg_type = String
		required = true
		help = "path to the dataset file fixing train, validation and test sets"
	end

	parsed_args = parse_args(args, s)

	folder = parsed_args["data"]
	model_path = parsed_args["model"]
	dataset_path = parsed_args["dataset"]

	directory = readdir(folder)

	@load "$(model_path)/nn_bestLV.bson" nn_best

	global nn = (nn_best)

	f = open(dataset_path * "dataset.json", "r")
	dataset = JSON.parse(f)["test"]
	close(f)

	a = test_model(folder, directory, nn, dataset)
end


main(ARGS)
