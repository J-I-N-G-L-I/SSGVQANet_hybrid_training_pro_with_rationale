# SSGVQANet (Hybrid trained: TemporalQA + Multi-Task)

> 1. We extend **Original SSG-VQA (single-frame)** to a **mixed-temporal, multi-task** regime (counting, duration, boundary, concurrency, ordering, proximity, motion, extreme, phase transition), based on **VisualBERT** with early cross-modal fusion and **temporal position embeddings**.  
> 2. Original baseline (labeled **Baseline/Legacy**) is kept for faithful reproduction and fair comparison.
---

## Project Layout
```markdown
├── models/
│ ├── VisualBert_ssgqa.py # VisualBERT with Scene↔Text fusion + temporal pos. embeddings
│ ├── multitask_ssgvqa.py # MultiTaskSSGVQA: heads for answer/count/duration/...
│ ├── VisualBertClassification.py # [Baseline/Legacy] single-frame VisualBERT classifier
│ ├── resnets.py # [Legacy/Optional] CNN wrappers
│ └── ...
├── utils/
│ ├── mixed_datasets.py # single-frame + temporal datasets; fallback
│ ├── collate_temporal.py # collate for multi-frame/multi-ROI
│ ├── dataloaderClassification.py # [Baseline/Legacy] single-frame loader
│ └── utils.py # metrics, checkpoints, LR scheduler
├── train.py
├── test_answer_head.py
├── test.py / test_mix_trained.py
├── checkpoints/
├── data/
└── README.md
```

## Environment

- Python ≥ 3.8  
- PyTorch ≥ 1.12 (match CUDA & driver)  
- torchvision, transformers, h5py, numpy, wandb (optional)

**Example:**
```bash
conda create -n ssgvqa python=3.9 -y
conda activate ssgvqa
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.41.2 h5py numpy wandb
```
**Tokenizer**: emilyalsentzer/Bio_ClinicalBERT (downloaded on first run).

## Data & Feature Layout
```markdown
<feature_root>/
└── visual_feats/
    ├── cropped_images/<VIDxx>/vqa/img_features/1x1/<frame:06d>.hdf5
    └── roi_yolo_coord/<VIDxx>/labels/vqa/img_features/roi/<frame:06d>.hdf5
```
- Global (1x1): (512,)
- ROI (roi): (K,530) = [18 semantics+box | 512 visual]
- Replace ROI 512 with global 512 → [18 | 512]

## SSG-VQA QA files:
```markdown
<qa_root>/VIDxx/*.txt   # "question|answer"
```
**Fallback**: handled automatically by FrameFeatureStore.

## Fixed Splits
- **Train (35)**: VID73, VID40, VID62, ... VID70
- **Val (5)**:    VID18, VID48, VID01, VID35, VID31
- **Test (5)**:   VID02, VID22, VID43, VID60, VID74

## Train / Validate / Test
### Single-frame baseline
```bash
python test.py --validate=True --checkpoint checkpoints/ssg-qa-net.pth.tar
```

#### Result:
```markdown
Acc=0.6076 | mAP=0.5498 | mAR=0.4918 | mF1=0.5039 | wF1=0.6037
```
### Hybrid-temporal training
```bash
python train.py \
  --dataset_type mixed-temporal \
  --ssg_qa_root "E:/LJ/datasets/SSGVQA/ssg-qa/ssg-qa" \
  --stsg_qa_root "E:/LJ/datasets/STSG_QA" \
  --feature_root "E:/LJ/datasets/SSGVQA" \
  --tokenizer_ver v2 \
  --max_frames 16 --fps 1.0 \
  --count_bins 11 \
  --batch_size 64 --epochs 80 \
  --use_wandb False
 ```
### Evaluate answer head on single-frame SSG-VQA
```
python test_answer_head.py \
  --qa_root "E:/LJ/datasets/SSGVQA/ssg-qa/ssg-qa" \
  --feature_root "D:/Programming/SSG-VQA-re/data" \
  --checkpoint ./checkpoints/Best_fair_training.pth.tar \
  --checkpoint_dir ./test_run_results/ \
  --batch_size 32 \
  --tokenizer_ver v4
 ```
Saves results to:
```markdown
./test_run_results/answer_head_predictions.csv
```
### Results (%)
|Method| Acc      |mAP|mAR|mF1|wF1|
---|----------|---|---|---|---|
|Original single-frame baseline| 0.6076   |0.5498|0.4918|0.5039|0.6037|
|Mixed-temporal model (answer, single)| **0.7565** |**0.5759**|**0.5433**|**0.5375**|**0.7436**|

## Model Highlights
- **VisualBERT mods**
  - Scene↔Text fusion (ROI 18D semantics+box ↔ text)
  - Concatenate global 512D → enriched visual token 
  - Temporal pos embeddings (visual_frame_ids)
- **MultiTaskSSGVQA**
  - Shared encoder 
  - Heads: answer / count / duration / boundary / concurrency / ordering / motion / extreme / phase_transition

## Reproducibility Tips
- Legacy weights may use head_answer; new uses cls_answer (auto-mapped). 
- Windows: num_workers ≤ 2. 
- Fix seeds & cudnn flags for determinism.

## Citation & License
- Cite VisualBERT and related works when publishing.
- Datasets (SSG-VQA, Cholec80) follow their own licenses.