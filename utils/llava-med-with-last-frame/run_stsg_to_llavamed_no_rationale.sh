#!/bin/bash
#SBATCH --job-name=export_stsg_llava_last_frame_no_rationale
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --open-mode=append

module load miniforge
conda activate SSGVQA_comp_exp

export LANG=en_GB.UTF-8
export LC_ALL=en_GB.UTF-8
export PYTHONIOENCODING=UTF-8
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONHASHSEED=42

PROJECT_DIR="${PROJECT_DIR:-/users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale}"

# STSG temporal QA root
STSG_QA_ROOT="${STSG_QA_ROOT:-/mnt/scratch/sc232jl/datasets/SSGVQA/data/STSG_QA_Pro_8_Classes}"

# Raw frames root from CholecT45
VIDEO_ROOT="${VIDEO_ROOT:-/mnt/scratch/sc232jl/datasets/CholecT45/data}"

VOCAB_JSON="${VOCAB_JSON:-/users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/utils/vocabs_train.json}"

# Output root for LLaVA-Med export (JSONL + single frames)
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/scratch/sc232jl/LLaVA-Med/stsg_temporal_eval_last_frame_no_rationale}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

TEMPORAL_VIDEOS=(${TEMPORAL_VIDEOS:-VID22 VID74 VID60 VID02 VID43})

# Frame export parameters
TILE_WIDTH=${TILE_WIDTH:-320}
TILE_HEIGHT=${TILE_HEIGHT:-180}
FRAME_EXT="${FRAME_EXT:-.png}"

echo "== Runtime summary =="
echo "PROJECT_DIR    : ${PROJECT_DIR}"
echo "STSG_QA_ROOT   : ${STSG_QA_ROOT}"
echo "VIDEO_ROOT     : ${VIDEO_ROOT}"
echo "OUTPUT_ROOT    : ${OUTPUT_ROOT}"
echo "TEMPORAL_VIDEOS: ${TEMPORAL_VIDEOS[*]}"
echo "TILE_WIDTH     : ${TILE_WIDTH}"
echo "TILE_HEIGHT    : ${TILE_HEIGHT}"
echo "FRAME_EXT      : ${FRAME_EXT}"

cd "${PROJECT_DIR}" || exit 1

echo "== START export STSG QA to LLaVA-Med format (last frame, no rationale) =="

srun -u python utils/llava-med-with-last-frame/export_stsg_to_llavamed.py \
  --stsg_qa_root "${STSG_QA_ROOT}" \
  --video_root "${VIDEO_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --videos "${TEMPORAL_VIDEOS[@]}" \
  --vocabs_json  "${VOCAB_JSON}" \
  --tile_width "${TILE_WIDTH}" \
  --tile_height "${TILE_HEIGHT}" \
  --output_questions stsg_temporal_eval_questions_last_frame_no_rationale.jsonl \
  --frame_ext "${FRAME_EXT}"

echo "== DONE export STSG QA to LLaVA-Med format (last frame, no rationale) =="
