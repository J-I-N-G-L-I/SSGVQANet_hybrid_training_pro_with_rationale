#!/bin/bash
#SBATCH --job-name=build_vocabs
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
# #SBATCH --partition=standard

module load miniforge
conda activate SSGVQA_comp_exp

export PYTHONIOENCODING=UTF-8
export LANG=en_GB.UTF-8
export LC_ALL=en_GB.UTF-8


cd /users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/utils/

python -u save_train_vocabs.py
