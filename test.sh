#!/bin/bash
#SBATCH --job-name=ssgvqa_hybrid_testing
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=16G
#SBATCH --open-mode=append

module load miniforge
conda activate SSGVQA_comp_exp

export LANG=en_GB.UTF-8
export LC_ALL=en_GB.UTF-8
export PYTHONIOENCODING=UTF-8
export PYTHONUTF8=1
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONHASHSEED=42
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/SSGVQANet_hybrid_training_pro_with_rationale}"
DATA_DIR="${DATA_DIR:-/mnt/scratch/${USER}/datasets/SSGVQA/data}"
STSG_QA_ROOT="${STSG_QA_ROOT:-${DATA_DIR}/STSG_QA_Pro_8_Classes}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_DIR}}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/results_archive/test_temporal_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}}"
mkdir -p "${CHECKPOINT_DIR}"

# 1. temporal-only: without rationale
# old
# CKPT_FILE="${CKPT_FILE:-/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/temporal_only_rationale_off_20251107_123840_2163769/Best.pth.tar}"

# new
# CKPT_FILE="${CKPT_FILE:-/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/temporal_only_rationale_off_20251109_203039_2202391/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-False}

# new features
# CKPT_FILE="${CKPT_FILE:-/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/temporal_only_rationale_off_20251122_235616_2284589/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-False}


# 2. temporal-only: with rationale
# old
# CKPT_FILE="${CKPT_FILE:-/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/temporal_only_rationale_on_20251107_130512_2163770/Best.pth.tar}"

# new
# CKPT_FILE="${CKPT_FILE:-/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/temporal_only_rationale_on_20251109_203135_2202405/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-True}

# new features
# CKPT_FILE="${CKPT_FILE:-/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/temporal_only_rationale_on_20251122_235640_2284590/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-True}


# 3. hybrid: without rationale
# old
# CKPT_FILE="${CKPT_FILE:-/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_off_20251107_123316_2163767/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-False}

# new
# CKPT_FILE="${CKPT_FILE:-/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_off_20251109_203217_2202417/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-False}


# # new features
# CKPT_FILE="${CKPT_FILE:-/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_off_20251122_235724_2284592/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-False}


# 4. hybrid: with rationale; the TS_ratio is 2:1
# old
# CKPT_FILE="${CKPT_FILE:-/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251107_123407_2163768/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-True}

# new
# CKPT_FILE="${CKPT_FILE:-/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251109_203237_2202418/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-True}

# # new features
CKPT_FILE="${CKPT_FILE:-/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251122_235833_2284593/Best.pth.tar}"
USE_RATIONALE=${USE_RATIONALE:-True}




# ---------different ratio comparison (all hybrid + rationale)
# 1. 1:1
# CKPT_FILE="${CKPT_FILE:-/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251110_085909_2204954/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-True}

# 2. 4:1
# CKPT_FILE="${CKPT_FILE:-/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251110_090018_2204956/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-True}

# 3. 8:1
# CKPT_FILE="${CKPT_FILE:-/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints/hybrid_rationale_on_20251110_090035_2204957/Best.pth.tar}"
# USE_RATIONALE=${USE_RATIONALE:-True}

if [[ ! -f "$CKPT_FILE" ]]; then
  echo "ERROR: checkpoint not found at $CKPT_FILE"
  exit 2
fi
echo "[Use checkpoint] $CKPT_FILE"
ls -lh "$CKPT_FILE" || true

COUNT_BINS=${COUNT_BINS:-20}
MAX_FRAMES=${MAX_FRAMES:-16}
FPS=${FPS:-1.0}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-6}}

TOKENIZER_VER="${TOKENIZER_VER:-v2}"  # v4 -> bert-base-uncased; but 2025/11/08 training used v2

TEXT_MAX_LEN=${TEXT_MAX_LEN:-320}

# test sets (fixed)
TEMPORAL_VIDEOS=(${TEMPORAL_VIDEOS:-VID22 VID74 VID60 VID02 VID43})

VOCABS_JSON="${VOCABS_JSON:-${PROJECT_DIR}/utils/vocabs_train.json}"

echo "== Runtime summary =="
echo "PROJECT_DIR    : ${PROJECT_DIR}"
echo "STSG_QA_ROOT   : ${STSG_QA_ROOT}"
echo "FEATURE_ROOT   : ${FEATURE_ROOT}"
echo "CHECKPOINT_DIR : ${CHECKPOINT_DIR}"
echo "CKPT_FILE      : ${CKPT_FILE}"
echo "COUNT_BINS     : ${COUNT_BINS}"
echo "MAX_FRAMES     : ${MAX_FRAMES}"
echo "FPS            : ${FPS}"
echo "BATCH_SIZE     : ${BATCH_SIZE}"
echo "NUM_WORKERS    : ${NUM_WORKERS}"
echo "TOKENIZER_VER  : ${TOKENIZER_VER}"
echo "TEXT_MAX_LEN   : ${TEXT_MAX_LEN}"
echo "USE_RATIONALE  : ${USE_RATIONALE}"
echo "VIDEOS         : ${TEMPORAL_VIDEOS[*]}"
echo "VOCABS_JSON    : ${VOCABS_JSON}"

nvidia-smi || true
cd "${PROJECT_DIR}"

echo "== START temporal evaluation (use_rationale=${USE_RATIONALE}) =="

srun -u python test_hybrid.py \
  --stsg_qa_root "${STSG_QA_ROOT}" \
  --feature_root "${FEATURE_ROOT}" \
  --checkpoint "${CKPT_FILE}" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --count_bins "${COUNT_BINS}" \
  --max_frames "${MAX_FRAMES}" \
  --fps "${FPS}" \
  --tokenizer_ver "${TOKENIZER_VER}" \
  --text_max_len "${TEXT_MAX_LEN}" \
  --use_rationale "${USE_RATIONALE}" \
  --vocabs_json "${VOCABS_JSON}" \
  --temporal_videos "${TEMPORAL_VIDEOS[@]}"

echo "== DONE =="


# #!/bin/bash
# #SBATCH --job-name=ssgvqa_hybrid_testing_temporal
# #SBATCH --output=%x_%j.out
# #SBATCH --error=%x_%j.err
# #SBATCH --time=12:00:00
# #SBATCH --partition=gpu
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=6
# #SBATCH --mem-per-cpu=16G
# #SBATCH --open-mode=append

# module load miniforge
# conda activate SSGVQA_comp_exp

# export LANG=en_GB.UTF-8
# export LC_ALL=en_GB.UTF-8
# export PYTHONIOENCODING=UTF-8
# export PYTHONUTF8=1
# export PYTHONUNBUFFERED=1

# export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
# export PYTHONHASHSEED=42
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

# PROJECT_DIR="${PROJECT_DIR:-${HOME}/SSGVQANet_hybrid_training_pro_with_rationale}"
# DATA_DIR="${DATA_DIR:-/mnt/scratch/${USER}/datasets/SSGVQA/data}"
# STSG_QA_ROOT="${STSG_QA_ROOT:-${DATA_DIR}/STSG_QA_Pro_8_Classes}"
# FEATURE_ROOT="${FEATURE_ROOT:-${DATA_DIR}}"
# export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

# CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_DIR}/results_archive/test_temporal_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}}"
# mkdir -p "${CHECKPOINT_DIR}"

# CKPT_FILE="${CKPT_FILE:-/users/${USER}/SSGVQANet_hybrid_training_pro_with_rationale/checkpoints_dir/Best.pth.tar}"
# if [[ ! -f "$CKPT_FILE" ]]; then
#   echo "ERROR: checkpoint not found at $CKPT_FILE"
#   exit 2
# fi
# echo "[Use checkpoint] $CKPT_FILE"
# ls -lh "$CKPT_FILE" || true

# COUNT_BINS=${COUNT_BINS:-20}
# MAX_FRAMES=${MAX_FRAMES:-16}
# FPS=${FPS:-1.0}
# BATCH_SIZE=${BATCH_SIZE:-32}
# NUM_WORKERS=${NUM_WORKERS:-${SLURM_CPUS_PER_TASK:-6}}

# TOKENIZER_VER="${TOKENIZER_VER:-v4}"  # default v4 -> bert-base-uncased
# LAYERS=${LAYERS:-6}
# N_HEADS=${N_HEADS:-8}
# HIDDEN_SIZE=${HIDDEN_SIZE:-1024}
# TEXT_MAX_LEN=${TEXT_MAX_LEN:-320}

# TEMPORAL_VIDEOS=(${TEMPORAL_VIDEOS:-VID22 VID74 VID60 VID02 VID43})

# echo "== Runtime summary =="
# echo "PROJECT_DIR    : ${PROJECT_DIR}"
# echo "STSG_QA_ROOT   : ${STSG_QA_ROOT}"
# echo "FEATURE_ROOT   : ${FEATURE_ROOT}"
# echo "CHECKPOINT_DIR : ${CHECKPOINT_DIR}"
# echo "CKPT_FILE      : ${CKPT_FILE}"
# echo "COUNT_BINS     : ${COUNT_BINS}"
# echo "MAX_FRAMES     : ${MAX_FRAMES}"
# echo "FPS            : ${FPS}"
# echo "BATCH_SIZE     : ${BATCH_SIZE}"
# echo "NUM_WORKERS    : ${NUM_WORKERS}"
# echo "TOKENIZER_VER  : ${TOKENIZER_VER}"
# echo "LAYERS         : ${LAYERS}"
# echo "N_HEADS        : ${N_HEADS}"
# echo "HIDDEN_SIZE    : ${HIDDEN_SIZE}"
# echo "TEXT_MAX_LEN   : ${TEXT_MAX_LEN}"
# echo "VIDEOS         : ${TEMPORAL_VIDEOS[*]}"

# nvidia-smi || true

# cd "${PROJECT_DIR}"
# echo "== START temporal evaluation (rationale-aware) =="

# srun -u python test_hybrid_trained_with_rationale_new.py \
#   --stsg_qa_root "${STSG_QA_ROOT}" \
#   --feature_root "${FEATURE_ROOT}" \
#   --checkpoint "${CKPT_FILE}" \
#   --checkpoint_dir "${CHECKPOINT_DIR}" \
#   --batch_size "${BATCH_SIZE}" \
#   --count_bins "${COUNT_BINS}" \
#   --max_frames "${MAX_FRAMES}" \
#   --fps "${FPS}" \
#   --tokenizer_ver "${TOKENIZER_VER}" \
#   --layers "${LAYERS}" \
#   --n_heads "${N_HEADS}" \
#   --hidden_size "${HIDDEN_SIZE}" \
#   --text_max_len "${TEXT_MAX_LEN}" \
#   --temporal_videos "${TEMPORAL_VIDEOS[@]}"

# echo "== DONE =="


