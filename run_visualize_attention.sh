#!/bin/bash
#SBATCH --job-name=vis_attn
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=visual_logs/vis_attn_%j.out
#SBATCH --error=visual_logs/vis_attn_%j.err


mkdir -p visual_logs

module load miniforge
conda activate SSGVQA_comp_exp

cd /users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale

# TO + rationale
# --checkpoint "/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/temporal_only_rationale_on_20251109_203135_2202405/Best.pth.tar" \


# HY + no rationale
# --checkpoint "/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_off_20251109_203217_2202417/Best.pth.tar" \

# HY + rationale
# --checkpoint "/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251109_203237_2202418/Best.pth.tar" \


python visualization_attention.py \
  --stsg_qa_root "/mnt/scratch/sc232jl/datasets/SSGVQA/data/STSG_QA_Pro_8_Classes" \
  --feature_root "/mnt/scratch/sc232jl/datasets/SSGVQA/data" \
  --raw_image_root "/mnt/scratch/sc232jl/datasets/CholecT45/data" \
  --checkpoint "/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251109_203237_2202418/Best.pth.tar" \
  --temporal_videos VID22 VID74 VID60 VID02 VID43 \
  --vocabs_json "/users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/utils/vocabs_train.json" \
  --output_dir "/mnt/scratch/sc232jl/vis_output_test" \
  --vis_count 20 \
  --target_category "ordering" \
  --vis_mode "global"