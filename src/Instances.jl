module Instances

const VERSION = "0.1.0"

using JuMP 
using HiGHS, Gurobi
using LinearAlgebra
using SparseArrays
using Random
using Statistics


global CR_Optimiser = HiGHS.Optimizer

global LR_Optimiser = Gurobi.Optimizer

#assure the good CUDA version, for the cluster
#CUDA.set_runtime_version!(v"11.4")

include("dataStructures/abstract_instance.jl")

include("dataStructures/ga_instance.jl")

include("dataStructures/cwl_instance.jl")
include("dataStructures/MCND_dataInstance.jl")
include("dataStructures/mcnd_instance.jl")
include("dataStructures/mcnd_instance_gpu.jl")
include("dataStructures/UC_instance.jl")

include("optimization/KnapsackSolver.jl")
include("optimization/CWL/linearRelaxation.jl")
include("optimization/CWL/lagrangianSubProblem.jl")
include("optimization/MCND/linearRelaxation.jl")
include("optimization/MCND/lagrangianSubProblem.jl")
include("optimization/GA/linearRelaxation.jl")
include("optimization/GA/lagrangianSubProblem.jl")
include("optimization/gpuMCND/lagrangianSubProblem.jl")
include("optimization/UC/linearRelaxation.jl")
include("optimization/UC/lagrangianRelaxation.jl")

export abstractInstance,abstractInstanceFactory
export instanceCWL,cpuInstanceCWL ,CWLinstanceFactory,cpuCWLinstanceFactory
export instanceGA, cpuInstanceGA, GAinstanceFactory, cpuGAinstanceFactory
export abstractInstanceMCND
export gpuMCNDinstance, gpuMCNDinstanceFactory
export cpuInstanceMCND, MCNDinstanceFactory, cpuMCNDinstanceFactory, modify_instance

export objective_coefficient_type
export create_data_object, lengthLM, sizeLM, read_dat, print_dat,read_modify_dat
export read_modify_dat, generate_GA
export origin,destination, volume, routing_cost, tail, head, capacity, fixed_cost, sizeK, sizeV, sizeE, b, isInKij, outdegree, indegree, outdegree_k, indegree_k
export LR, CR
export cijk, constantLagrangianBound, value_LR, value_LR_a
export solve_knapsack_continuous, solve_knapsack

end
