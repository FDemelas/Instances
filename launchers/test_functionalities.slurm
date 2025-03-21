#!/usr/bin/env bash

#SBATCH --job-name=full_25
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1    
#SBATCH --cpus-per-task=1     
#SBATCH --ntasks=1 
#SBATCH --gres=gpu:1       
#SBATCH --qos=qos_gpu_t4        
#SBATCH --output=./results/output_full_100.txt    
#SBATCH --error=./results/error_full_100.txt 

cd ..
export JULIA_NUM_THREADS=1

julia --project=. ./runs/runBaselines.jl --folder ./data/MCNDforTest/ --maxIterDescentType 10 --maxIterBundle 10 --TS 100 

julia --project=. ./runs/runTraining.jl --data ./data/MCNDforTest/ --lr 1.0e-4 --decay 0.9 --cn 5 --mti 4 --mvi 4 --seed 1 --maxIt 10 --maxEP 10 --soft_updates true --h_representation 32 --use_softmax false --gamma 0.999 --lambda 0.0 --delta 0.0 --use_graph false --maxItBack -1 --maxItVal 20 --batch_size 2 --always_batch true --h_act softplus --sampling_gamma false

julia --project=. ./runs/runTraining.jl --data ./data/MCNDforTest/ --lr 1.0e-4 --decay 0.9  --cn 5 --mti 4 --mvi 4 --seed 1 --maxIt 10 --maxEP 10 --soft_updates true --h_representation 32  --use_softmax false --gamma 0.999 --lambda 0.0 --delta 0.0 --use_graph false --maxItBack -1 --maxItVal 20 --batch_size 1 --always_batch true --h_act softplus --sampling_gamma false

julia --project=. ./runs/runTrainingT.jl --data ./data/MCNDforTest/ --lr 1.0e-4 --cn 5 --mti 4 --mvi 4 --seed 1 --maxIT 10 --maxEP 10 --cr_init false --telescopic true --instance_features true --gamma 0.9 --single_prediction false

julia --project=. ./runs/runTraining.jl --data ./data/MCNDforTest/ --lr 1.0e-4 --decay 0.9  --cn 5 --mti 2 --mvi 2 --seed 1 --maxIt 10 --maxEP 10 --soft_updates true --h_representation 32 --use_softmax false --gamma 0.999 --lambda 0.0 --delta 0.0 --use_graph false --maxItBack -1 --maxItVal 20 --batch_size 1 --always_batch false --h_act softplus --sampling_gamma false

julia --project=. ./runs/runTest.jl  --folder ./data/MCNDforTest/  --model_folder BatchVersion_bs_1_true_MCNDforTest_0.0001_0.9_5_4_4_1_10_10_true_32_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_1.0

julia --project=. ./runs/runTestT.jl  --data  ./data/MCNDforTest/ --model ./res/res_goldLossWeights_withInstFeat_initZero_lr0.0001_cn5_maxIT10_maxEP10_dataMCNDforTest_exactGradtrue_gamma0.9_seed1_single_predictionfalse_0.0_ --dataset ./res/BatchVersion_bs_1_true_MCNDforTest_0.0001_0.9_5_4_4_1_10_10_true_32_false_softplus_false_0.999_0.0_0.0_sparsemax_false_false_1.0/
