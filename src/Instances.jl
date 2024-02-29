module Instances

const VERSION = "0.1.0"

using JuMP 
using HiGHS
using LinearAlgebra
using SparseArrays
using Random
using Statistics

#assure the good CUDA version, for the cluster
#CUDA.set_runtime_version!(v"11.4")

include("dataStructures/abstract_instance.jl")

include("dataStructures/ga_instance.jl")

include("dataStructures/cwl_instance.jl")
include("dataStructures/MCND_dataInstance.jl")
include("dataStructures/mcnd_instance.jl")
include("dataStructures/mcnd_instance_gpu.jl")

include("optimization/KnapsackSolver.jl")
include("optimization/CWL/linearRelaxation.jl")
include("optimization/CWL/lagrangianSubProblem.jl")
include("optimization/MCND/linearRelaxation.jl")
include("optimization/MCND/lagrangianSubProblem.jl")
include("optimization/GA/linearRelaxation.jl")
include("optimization/GA/lagrangianSubProblem.jl")
include("optimization/gpuMCND/lagrangianSubProblem.jl")

export abstractInstance, abstractInstanceMCND, cpuInstanceMCND, gpuInstanceMCND
export labelsExtraction, featuresExtraction
export prediction, target
export inputSize
export createLoss
export ComputeGAP, ComputeGAPset
export printOutResults
export abstractInstanceFactory, MCNDinstanceFactory,cpuMCNDinstanceFactory,gpuMCNDinstanceFactory, CWLinstanceFactory,cpuCWLinstanceFactory, cpuInstanceCWL, instanceCWL
export createExamplesFromInstance
export createDataSet, dataLoader
export createCorpus, createKfold
export dictValuesToVector
export featuresLoader
export tail, head, capacity, fixed_cost, routing_cost, origin, destination, volume
export sizeK, sizeV, sizeE, b, isInKij
export outdegree, indegree
export load_data
export labelsLoader
export cijk
export checkDims
export LR, value_LR_a, value_LR, constantLagrangianBound
export create_model
export pea_cor
export objective_coefficient_type
export dataLoader
export create_data_object
export CR

end
