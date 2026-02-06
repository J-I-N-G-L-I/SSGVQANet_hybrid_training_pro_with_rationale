#!/bin/bash
#SBATCH --job-name=vis_attn_static
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=visual_logs/vis_attn_static_%j.out
#SBATCH --error=visual_logs/vis_attn_static_%j.err

mkdir -p visual_logs

module load miniforge
conda activate SSGVQA_comp_exp

cd /users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale

# STSG-VQA-Net but only use multi-frame + static qa pairs: 
# --checkpoint "/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251127_103223_2317142/Best.pth.tar" \

# STSG-VQA-Net hybrid training, but for the answer cls head, only use one frame for each static QA: 
# --checkpoint "/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251109_203237_2202418/Best.pth.tar" \

# python visualization_attention.py \
#   --mode static \
#   --ssg_qa_root "/mnt/scratch/sc232jl/datasets/SSGVQA/data/qa_txt/ssg-qa" \
#   --feature_root "/mnt/scratch/sc232jl/datasets/SSGVQA/data" \
#   --raw_image_root "/mnt/scratch/sc232jl/datasets/CholecT45/data" \
#   --checkpoint "/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251109_203237_2202418/Best.pth.tar" \
#   --temporal_videos VID22 VID74 VID60 VID02 VID43 \
#   --output_dir "/mnt/scratch/sc232jl/vis_output_static_answer_hybrid" \
#   --static_frames_per_sample 12 \
#   --vis_count 500 \
#   --target_category "answer" \
#   --vis_mode "global"


# 单帧训练 ckpt（2202418）
# 注意：这里 不写 --static_use_multi_frames
# python visualization_attention.py \
#   --mode static \
#   --ssg_qa_root "/mnt/scratch/sc232jl/datasets/SSGVQA/data/qa_txt/ssg-qa" \
#   --feature_root "/mnt/scratch/sc232jl/datasets/SSGVQA/data" \
#   --raw_image_root "/mnt/scratch/sc232jl/datasets/CholecT45/data" \
#   --checkpoint "/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251109_203237_2202418/Best.pth.tar" \
#   --temporal_videos VID22 VID74 VID60 VID02 VID43 \
#   --output_dir "/mnt/scratch/sc232jl/vis_output_static_answer_single" \
#   --static_frames_per_sample 12 \
#   --vis_count 1000 \
#   --target_category "answer" \
#   --vis_mode "global"



# 多帧训练 ckpt（2317142）
python visualization_attention.py \
  --mode static \
  --ssg_qa_root "/mnt/scratch/sc232jl/datasets/SSGVQA/data/qa_txt/ssg-qa" \
  --feature_root "/mnt/scratch/sc232jl/datasets/SSGVQA/data" \
  --raw_image_root "/mnt/scratch/sc232jl/datasets/CholecT45/data" \
  --checkpoint "/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251127_103223_2317142/Best.pth.tar" \
  --temporal_videos VID22 VID74 VID60 VID02 VID43 \
  --output_dir "/mnt/scratch/sc232jl/vis_output_static_answer_multi" \
  --static_frames_per_sample 12 \
  --static_use_multi_frames \
  --vis_count 1000 \
  --target_category "answer" \
  --vis_mode "global"