#!/bin/bash
#SBATCH --job-name=llavamed_eval_last_frame
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=02:00:00
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

PROJECT_DIR="${PROJECT_DIR:-/users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale}"
ANSWERS_FILE="${ANSWERS_FILE:-/mnt/scratch/sc232jl/LLaVA-Med/stsg_temporal_eval_last_frame/answers_llavamed_v15_mistral7b_temp0_last_frame.jsonl}"

OUTPUT_FILE="${OUTPUT_FILE:-/mnt/scratch/sc232jl/LLaVA-Med/stsg_temporal_eval_last_frame/llavamed_v15_mistral7b_temp0_eval_last_frame.json}"

VOCAB_JSON="${VOCAB_JSON:-/users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/utils/vocabs_train.json}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "== Runtime summary =="
echo "PROJECT_DIR : ${PROJECT_DIR}"
echo "ANSWERS_FILE: ${ANSWERS_FILE}"
echo "OUTPUT_FILE : ${OUTPUT_FILE}"

cd "${PROJECT_DIR}" || exit 1

srun -u python utils/llava-med-with-last-frame/eval_llavamed_stsg.py \
    --answers-file "${ANSWERS_FILE}" \
    --output-file "${OUTPUT_FILE}" \
    --vocabs-json "${VOCAB_JSON}"

echo "== DONE LLaVA-Med STSG eval (last frame) =="
