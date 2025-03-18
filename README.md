# BundleNetworks.jl

To install the package use this command inside the project main directory:
```
using Pkg
Pkg.activate(".")
Pkg.add(url="https://github.com/FDemelas/Instances")
Pkg.instantiate();
using BundleNetworks
```

## Testing the Soft Bundle

Some example of runs to test the code.
Note: in all the examples we consider a small dataset contained in `./data/MCNDforTest/`, and so this should be substituted to larger dataset for "real" experiences.

The dataset can be found in:
...(ADD_LINK)...

Running the Baselines:
```julia
julia --project=. ./runs/runBaselines.jl --folder ./data/MCNDforTest/ --maxIterDescentType 1000 --maxIterBundle 100 --TS 0.1 1 10 100 1000 10000 
```
It runs the baselines consisting on Adam, Descent and the (Vanilla) Bundle method (with 4 different long-term t-strategies) considering different initialization for the t-parameter (`0.1,1,10,100,1000,10000`) and save the results in a json dictionary, solving all the instances contained in the folder `./data/MCNDforTest/`.
For the Bundle variants we consider `100` iterations and for the Descent variants `1000`.


1) Train a `BundleNetwork` on the dataset using a batch size of `2`:
```julia
julia --project=. ./runs/runTraining.jl --data ./data/MCNDforTest/ --lr 1.0e-4 --decay 0.9  --cn 5 --mti 4 --mvi 4 --seed 1 --maxIt 10 --maxEP 10 --soft_updates true --h_representation 32 --use_softmax false --gamma 0.999 --lambda 0.0 --delta 0.0 --use_graph false --maxItBack -1 --maxItVal 20 --batch_size 2 --always_batch true --h_act softplus --sampling_gamma false
```

2) Train a `BundleNetwork` on the dataset using a batch size of `1`, but still using the batch implementation:
```julia
julia --project=. ./runs/runTraining.jl --data ./data/MCNDforTest/ --lr 1.0e-5 --decay 0.9  --cn 5 --mti 4 --mvi 4 --seed 1 --maxIt 10 --maxEP 10 --soft_updates true --h_representation 32  --use_softmax false --gamma 0.999 --lambda 0.0 --delta 0.0 --use_graph false --maxItBack -1 --maxItVal 20 --batch_size 1 --always_batch true --h_act softplus --sampling_gamma false
```

3) Train a `BundleNetwork` on the dataset using a batch size of `1`, but without the batch implementation:
```julia
julia --project=. ./runs/runTraining.jl --data ./data/MCNDforTest/ --lr 1.0e-4 --decay 0.9  --cn 5 --mti 2 --mvi 2 --seed 1 --maxIt 10 --maxEP 10 --soft_updates true --h_representation 32 --use_softmax false --gamma 0.999 --lambda 0.0 --delta 0.0 --use_graph false --maxItBack -1 --maxItVal 20 --batch_size 1 --always_batch false --h_act softplus --sampling_gamma false
```

4) Train a model that predicts only T, but solve the 'true' Dual Master Problem:
```julia
julia --project=. ./runs/runTrainingT.jl --data ./data/MCNDforTest/ --lr 1.0e-4 --cn 5 --mti 4 --mvi 4 --seed 1 --maxIT 10 --maxEP 10 --cr_init false --telescopic true --instance_features true --gamma 0.9 --single_prediction false
```

Perform tests for the model trained in (2):
```julia
julia --project=. ./runs/runTest.jl  --folder ./data/MCNDforTest/  --model_folder BatchVersion_bs_1_true_MCNDforTest_1.0e-5_0.9_5_4_4_1_10_10_true_32_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_1.0
```

Perform tests for the model trained in (4):
```julia
julia --project=. ./runs/runTestT.jl  --data  ./data/MCNDforTest/ --model ./res/res_goldLossWeights_withInstFeat_initZero_lr0.0001_cn5_maxIT10_maxEP10_dataMCNDforTest_exactGradtrue_gamma0.9_seed1_single_predictionfalse_18.0_ --dataset ./res/BatchVersion_bs_1_true_MCNDforTest_1.0e-5_0.9_5_4_4_1_10_10_true_32_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_1.0/
```