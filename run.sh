#!/bin/bash
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu_a100
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 30
#SBATCH --mem-per-cpu 4000

#SBATCH --mail-user=y5jing@uwaterloo.ca
#SBATCH --mail-type=end,fail   

module load anaconda3
python3 mini_gen.py













