#!/bin/bash
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu_a100
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 10
#SBATCH --mem-per-cpu 1000

#SBATCH --mail-user=y5jing@uwaterloo.ca
#SBATCH --mail-type=end,fail   
#SBATCH --output=./outputs/%x-%j.out

module load anaconda3
CUBLAS_WORKSPACE_CONFIG=:16:8 python3 mini_gen.py















