#!/bin/bash
#SBATCH --job-name=ssgvqa_hybrid_training
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --output=/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/logs/%x_%j.out
#SBATCH --error=/mnt/scratch/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/logs/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=8G
#SBATCH --open-mode=append

module load miniforge
conda activate SSGVQA_comp_exp

export LANG=en_GB.UTF-8
export LC_ALL=en_GB.UTF-8
export PYTHONIOENCODING=UTF-8

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export PYTHONHASHSEED=42
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"

export PROJ=SSGVQANet_hybrid_training_pro_with_rationale
mkdir -p /scratch/$USER/$PROJ/{logs,checkpoints}


if [ -L ~/$PROJ/checkpoints ]; then
  echo "[skip] ~/$PROJ/checkpoints is a symlink -> skip rsync"
elif [ -d ~/$PROJ/checkpoints ]; then
  rsync -a --remove-source-files ~/$PROJ/checkpoints/ /scratch/$USER/$PROJ/checkpoints/ || true
  rm -rf ~/$PROJ/checkpoints
fi
ln -sfn /scratch/$USER/$PROJ/checkpoints ~/$PROJ/checkpoints

if [ -L ~/$PROJ/logs ]; then
  echo "[skip] ~/$PROJ/logs is a symlink -> skip rsync"
elif [ -d ~/$PROJ/logs ]; then
  rsync -a --remove-source-files ~/$PROJ/logs/ /scratch/$USER/$PROJ/logs/ || true
  rm -rf ~/$PROJ/logs
fi
ln -sfn /scratch/$USER/$PROJ/logs ~/$PROJ/logs

# caches to /scratch
export CONDA_ENVS_PATH=/scratch/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch/$USER/.conda_pkgs
export HF_HOME=/scratch/$USER/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/scratch/$USER/.cache/torch
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export TMPDIR=/scratch/$USER/tmp
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TMPDIR"

LOGDIR=/scratch/$USER/$PROJ/logs
CKPTDIR=/scratch/$USER/$PROJ/checkpoints
mkdir -p "$LOGDIR" "$CKPTDIR"

PROJECT_DIR="${PROJECT_DIR:-${HOME}/SSGVQANet_hybrid_training_pro_with_rationale}"
DATA_DIR="${DATA_DIR:-/mnt/scratch/sc232jl/datasets/SSGVQA/data}"
SSG_QA_ROOT="${SSG_QA_ROOT:-${DATA_DIR}/qa_txt/ssg-qa}"
STSG_QA_ROOT="${STSG_QA_ROOT:-${DATA_DIR}/STSG_QA_Pro_8_Classes}"
FEATURE_ROOT="${FEATURE_ROOT:-${DATA_DIR}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_DIR}/checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_DIR}/results_archive}"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs}"

export PROJECT_DIR DATA_DIR SSG_QA_ROOT STSG_QA_ROOT FEATURE_ROOT CHECKPOINT_ROOT RESULTS_ROOT LOG_DIR

# vocabs_train.json (keys: extreme, phase_transition, boundary, ordering, motion)
FIXED_VOCABS_JSON="${FIXED_VOCABS_JSON:-${PROJECT_DIR}/utils/vocabs_train.json}"

mkdir -p "${CHECKPOINT_ROOT}" "${RESULTS_ROOT}" "${LOG_DIR}"

# DATASET_TYPE: mixed-temporal / ssg-qa-roi_coord  -> (ssgvqa+stsg / only ssgvqa)
DATASET_TYPE="${DATASET_TYPE:-mixed-temporal}"

EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-0.000005}"
QUESTION_LEN="${QUESTION_LEN:-77}"
N_LAYERS="${N_LAYERS:-6}"
N_HEADS="${N_HEADS:-8}"
TOKENIZER_VER="${TOKENIZER_VER:-v2}"
NUM_WORKERS="${NUM_WORKERS:-1}"

# temporal/multi-task
MAX_FRAMES="${MAX_FRAMES:-16}"
FPS="${FPS:-1.0}"
COUNT_BINS="${COUNT_BINS:-20}"        # 0..19
W_ANSWER="${W_ANSWER:-1.0}"
W_COUNT="${W_COUNT:-1.0}"
W_DURATION="${W_DURATION:-1.0}"
W_TEMPORAL_CLS="${W_TEMPORAL_CLS:-1.0}"

# related to multi-frames + static qa pairs
# STATIC_MULTIFRAME="${STATIC_MULTIFRAME:-False}"   # True/False
# STATIC_FRAMES="${STATIC_FRAMES:-12}"             # window length K

# # whether use temporal QA (for static-only)
# TEMPORAL_ENABLED="${TEMPORAL_ENABLED:-True}"     # True=有STSG, False=只有静态QA True=with temporal; False=only static

export WANDB_MODE="${WANDB_MODE:-online}"
# export WANDB_PROJECT="${WANDB_PROJECT:-SSGVQA_mixtraining_${SLURM_JOB_ID}}"


# 1. temporal only + no rationale
# INCLUDE_STATIC="${INCLUDE_STATIC:-False}" # if only want to use temporal qa (filter static), set to False 
# USE_RATIONALE="${USE_RATIONALE:-False}"  # new added for temporal + rationale

# 2. temporal only + with rationale
# INCLUDE_STATIC="${INCLUDE_STATIC:-False}"
# USE_RATIONALE="${USE_RATIONALE:-True}"  

# 3. hybrid + no rationale
# INCLUDE_STATIC="${INCLUDE_STATIC:-True}"
# USE_RATIONALE="${USE_RATIONALE:-False}"  

# 4. hybrid + with rationale
# INCLUDE_STATIC="${INCLUDE_STATIC:-True}"
# USE_RATIONALE="${USE_RATIONALE:-True}"  

# 5. static + multiframes
INCLUDE_STATIC="True"               
TEMPORAL_ENABLED="False"     # 关闭 STSG temporal QA → static-only
STATIC_MULTIFRAME="True"     # 开启静态多帧模式
STATIC_FRAMES="12"           # 每个 QA 样本包含 12 帧（默认即可）


# (ablation): different ratio for temporal:static data 
# # 1. 1:1
# TS_RATIO="${TS_RATIO:-1:1}"

# # 2. 2:1 (already tried)
TS_RATIO="${TS_RATIO:-2:1}"

# # 3. 4:1
# TS_RATIO="${TS_RATIO:-4:1}"

# 4. 8:1
# TS_RATIO="${TS_RATIO:-8:1}"

if [[ "${INCLUDE_STATIC:-True}" == "True" ]]; then
  EXP_GROUP="hybrid"            # use both ssgvqa and stsg data
else
  EXP_GROUP="temporal_only"     # only use stsg data
fi
if [[ "${USE_RATIONALE:-True}" == "True" ]]; then
  RAT_TAG="rationale_on"
else
  RAT_TAG="rationale_off"
fi

export WANDB_PROJECT="STSG_hybrid_rationale_new_features"
export WANDB_RUN_GROUP="${EXP_GROUP}"
export WANDB_TAGS="${EXP_GROUP}, ${RAT_TAG}"
RUN_NAME_DEFAULT="${EXP_GROUP}_${RAT_TAG}_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
export RUN_NAME="${RUN_NAME:-${RUN_NAME_DEFAULT}}"
export WANDB_NAME="${RUN_NAME}"

CKPT_DIR="${CHECKPOINT_ROOT}/${RUN_NAME}/"
mkdir -p "${CKPT_DIR}"
export WANDB_DIR="${CKPT_DIR}"

export WANDB_ENTITY="${WANDB_ENTITY:-}"



RATIONALE_MODE="${RATIONALE_MODE:-append}"   # append/prefix/pair
RATIONALE_DROPOUT="${RATIONALE_DROPOUT:-0.2}"
TEXT_MAX_LEN="${TEXT_MAX_LEN:-320}" # keep both(w/ w/o rationale) 320 for fair comparison
# TEXT_MAX_LEN="${TEXT_MAX_LEN:-192}" # reduce it when training without rationale (USE_RATIONALE:-False) to save computing

if [[ -n "${WANDB_API_KEY:-}" ]]; then
  python - <<'PY' || true
import os, subprocess
key=os.environ.get("WANDB_API_KEY")
subprocess.run(["wandb","login","--relogin",key], check=False)
PY
fi

echo "== Job ID: ${SLURM_JOB_ID}"
echo "== Host: $(hostname)"
echo "== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "== PROJECT_DIR=${PROJECT_DIR}"

cd "${PROJECT_DIR}"

{
  echo "===== CONDA ENV (export) ====="
  conda env export || true
  echo "===== PIP FREEZE ====="
  pip freeze || true
  echo "===== PYTORCH/CUDA ====="
  python - <<'PY'
import torch, sys
print("python:", sys.version.replace("\n"," "))
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version (torch):", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device name[0]:", torch.cuda.get_device_name(0))
PY
  echo "===== NVIDIA-SMI ====="
  nvidia-smi || true
  echo "===== GIT ====="
  git rev-parse HEAD 2>/dev/null || echo "not a git repo"
} | tee "${CKPT_DIR}/env_info.txt"

python - <<'PY' | tee "${CKPT_DIR}/dataset_snapshot.txt"
import os, glob, json

SSG_ROOT = os.environ.get("SSG_QA_ROOT", "")
STSG_ROOT = os.environ.get("STSG_QA_ROOT", "")

train_vids = ["VID73","VID40","VID62","VID42","VID29","VID56","VID50","VID78","VID66","VID13",
              "VID52","VID06","VID36","VID05","VID12","VID26","VID68","VID32","VID49","VID65",
              "VID47","VID04","VID23","VID79","VID51","VID10","VID57","VID75","VID25","VID14",
              "VID15","VID08","VID80","VID27","VID70"]
val_vids   = ["VID18", "VID48", "VID01", "VID35", "VID31"]
test_vids  = ["VID22", "VID74", "VID60", "VID02", "VID43"]

def count_ssg(root, vids):
    n_txt, n_q = 0, 0
    exist, missing = [], []
    for v in vids:
        d=os.path.join(root, v)
        if os.path.isdir(d):
            exist.append(v)
            fs=glob.glob(os.path.join(d,"*.txt"))
            n_txt += len(fs)
            for f in fs:
                try:
                    with open(f,"r",encoding="utf-8",errors="ignore") as r:
                        for ln in r:
                            if ln.strip(): n_q += 1
                except: pass
        else:
            missing.append(v)
    return exist, missing, n_txt, n_q

def count_stsg(root, vids):
    n_json, n_q = 0, 0
    exist, missing = [], []
    for v in vids:
        d=os.path.join(root, v)
        p=os.path.join(d,"temporal_qa.json")
        if os.path.isdir(d) and os.path.isfile(p):
            exist.append(v)
            n_json += 1
            try:
                import json
                with open(p,"r",encoding="utf-8",errors="ignore") as r:
                    data=json.load(r)
                    if isinstance(data,list):
                        n_q += sum(1 for _ in data)
            except: pass
        else:
            missing.append(v)
    return exist, missing, n_json, n_q

print("=== SSG single-frame (by split) ===")
for name, vids in [("train",train_vids),("val",val_vids),("test",test_vids)]:
    exist, missing, n_txt, n_q = count_ssg(SSG_ROOT, vids)
    print(f"{name}: want={len(vids)} exist={len(exist)} missing={len(missing)} txt_files={n_txt} questions={n_q}")
    if missing: print("  missing:", sorted(missing))

print("\n=== STSG temporal (by split) ===")
for name, vids in [("train",train_vids),("val",val_vids),("test",test_vids)]:
    exist, missing, n_json, n_q = count_stsg(STSG_ROOT, vids)
    print(f"{name}: want={len(vids)} exist={len(exist)} missing={len(missing)} json={n_json} questions={n_q}")
    if missing: print("  missing:", sorted(missing))

print("\n[Root overview]")
print("SSG root:", SSG_ROOT, "| VID dirs:", len([d for d in glob.glob(os.path.join(SSG_ROOT,'VID*')) if os.path.isdir(d)]))
print("STSG root:", STSG_ROOT, "| VID dirs:", len([d for d in glob.glob(os.path.join(STSG_ROOT,'VID*')) if os.path.isdir(d)]))
PY

echo "== Start training: ${DATASET_TYPE} =="
if [[ "${DATASET_TYPE}" == "mixed-temporal" ]]; then
  srun -u python train_hybrid.py \
    --dataset_type mixed-temporal \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --workers "${NUM_WORKERS}" \
    --question_len "${QUESTION_LEN}" \
    --encoder_layers "${N_LAYERS}" \
    --n_heads "${N_HEADS}" \
    --tokenizer_ver "${TOKENIZER_VER}" \
    --checkpoint_dir "${CKPT_DIR}" \
    --ssg_qa_root "${SSG_QA_ROOT}" \
    --stsg_qa_root "${STSG_QA_ROOT}" \
    --feature_root "${FEATURE_ROOT}" \
    --fixed_vocabs_json "${FIXED_VOCABS_JSON}" \
    --max_frames "${MAX_FRAMES}" \
    --fps "${FPS}" \
    --count_bins "${COUNT_BINS}" \
    --w_answer "${W_ANSWER}" \
    --w_count "${W_COUNT}" \
    --w_duration "${W_DURATION}" \
    --w_temporal_cls "${W_TEMPORAL_CLS}" \
    --use_wandb True \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_mode "${WANDB_MODE}" \
    --run_name "${RUN_NAME}" \
    --include_static "${INCLUDE_STATIC}" \
    --use_rationale "${USE_RATIONALE}" \
    --rationale_mode "${RATIONALE_MODE}" \
    --rationale_dropout "${RATIONALE_DROPOUT}" \
    --temporal_enabled "${TEMPORAL_ENABLED}" \
    --static_multiframe "${STATIC_MULTIFRAME}" \
    --static_frames "${STATIC_FRAMES}" \
    --text_max_len "${TEXT_MAX_LEN}" \
    --ts_ratio="${TS_RATIO}" \
    --loss_norm=per_sample \
    2>&1 | tee "${CKPT_DIR}/train_console.log"
else
  srun -u python train_hybrid.py \
    --dataset_type ssg-qa-roi_coord \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --workers "${NUM_WORKERS}" \
    --question_len "${QUESTION_LEN}" \
    --encoder_layers "${N_LAYERS}" \
    --n_heads "${N_HEADS}" \
    --tokenizer_ver "${TOKENIZER_VER}" \
    --patch_size 1 \
    --checkpoint_dir "${CKPT_DIR}" \
    --include_static "${INCLUDE_STATIC}" \
    --use_wandb True \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --wandb_mode "${WANDB_MODE}" \
    --run_name "${RUN_NAME}" \
    2>&1 | tee "${CKPT_DIR}/train_console.log"
fi

CKPT_FILE=$(ls -1t "${CKPT_DIR}"/*.pth* 2>/dev/null | head -n1 || true)
if [[ -z "${CKPT_FILE}" ]]; then
  echo "can't find checkpoint! skip validate-only stage"
else
  echo "== Validate with checkpoint: ${CKPT_FILE}"
  if [[ "${DATASET_TYPE}" == "mixed-temporal" ]]; then
    srun -u python train_hybrid.py \
      --dataset_type mixed-temporal \
      --epochs 1 \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --workers "${NUM_WORKERS}" \
      --question_len "${QUESTION_LEN}" \
      --encoder_layers "${N_LAYERS}" \
      --n_heads "${N_HEADS}" \
      --tokenizer_ver "${TOKENIZER_VER}" \
      --checkpoint_dir "${CKPT_DIR}" \
      --ssg_qa_root "${SSG_QA_ROOT}" \
      --stsg_qa_root "${STSG_QA_ROOT}" \
      --feature_root "${FEATURE_ROOT}" \
      --fixed_vocabs_json "${FIXED_VOCABS_JSON}" \
      --max_frames "${MAX_FRAMES}" \
      --fps "${FPS}" \
      --count_bins "${COUNT_BINS}" \
      --w_answer "${W_ANSWER}" \
      --w_count "${W_COUNT}" \
      --w_duration "${W_DURATION}" \
      --w_temporal_cls "${W_TEMPORAL_CLS}" \
      --checkpoint "${CKPT_FILE}" \
      --include_static "${INCLUDE_STATIC}" \
      --temporal_enabled "${TEMPORAL_ENABLED}" \
      --static_multiframe "${STATIC_MULTIFRAME}" \
      --static_frames "${STATIC_FRAMES}" \
      --validate True \
      --use_wandb True \
      --wandb_project "${WANDB_PROJECT}" \
      --use_rationale "${USE_RATIONALE}" \
      --rationale_mode "${RATIONALE_MODE}" \
      --rationale_dropout 0.0 \
      --text_max_len "${TEXT_MAX_LEN}"
  fi
fi
