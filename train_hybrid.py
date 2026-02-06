# -*- coding: utf-8 -*-

import os
import json
import argparse
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoTokenizer, BertTokenizer

from utils.utils import (
    calc_acc, calc_classwise_acc, calc_precision_recall_fscore,
    eval_for_f1_et_all, adjust_learning_rate, save_clf_checkpoint
)
from utils.dataloaderClassification import (
    SSGVQAClassification_full_roi_coord,
    SSGVQAClassification_full_roi_analysis
)

from utils.mixed_datasets import (
    SSGVQAFrameDataset, STSGTemporalQADataset, build_logger
)
from utils.collate_temporal import make_collate_fn

from models.VisualBertClassification_ssgqa import VisualBertClassification
from models.multitask_ssgvqa import MultiTaskSSGVQA

import wandb

warnings.simplefilter(action="ignore", category=FutureWarning)


# utils
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_vid_dirs(root: str, held_out_bases: set) -> List[str]:
    if not os.path.isdir(root):
        return []
    all_dirs = [d for d in os.listdir(root) if d.startswith("VID") and os.path.isdir(os.path.join(root, d))]
    kept = [d for d in all_dirs if not any(d.startswith(b) for b in held_out_bases)]
    return sorted(kept)


def split_vids(vids: List[str], val_ratio=0.1, seed=42) -> Tuple[List[str], List[str]]:
    import random
    vids = list(sorted(vids))
    rnd = random.Random(seed)
    rnd.shuffle(vids)
    n_val = max(1, int(len(vids) * val_ratio))
    val = vids[:n_val]
    train = vids[n_val:]
    return train, val


# Single-frame training/val
def train_single_epoch(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    label_true = None
    label_pred = None

    for i, (_, visual_features, q, labels) in enumerate(train_dataloader, 0):
        questions = list(q)
        inputs = tokenizer(
            questions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.question_len,
        )

        visual_features = visual_features.to(device)
        labels = labels.to(device)

        if args.transformer_ver == "pure_language":
            outputs = model(inputs)
        else:
            outputs = model(inputs, visual_features)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)

        label_true = labels.detach().cpu() if label_true is None else torch.cat((label_true, labels.detach().cpu()), 0)
        label_pred = predicted.detach().cpu() if label_pred is None else torch.cat((label_pred, predicted.detach().cpu()), 0)

    # metrics
    acc, c_acc = calc_acc(label_true, label_pred), calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)
    print(f"Train(single): epoch: {epoch} loss: {total_loss:.6f} | Acc: {acc:.6f} | Precision: {precision:.6f} | "
          f"Recall: {recall:.6f} | FScore: {fscore:.6f}")

    return {
        "train/loss_sum": float(total_loss),
        "train/static/answer_acc": float(acc),
        "train/static/precision": float(precision),
        "train/static/recall": float(recall),
        "train/static/f1": float(fscore),
    }


@torch.no_grad()
def validate_single(args, val_loader, model, criterion, epoch, tokenizer, device, save_output=False) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    label_true = None
    label_pred = None

    criterion = nn.CrossEntropyLoss()

    for i, (file_name, visual_features, q, labels) in enumerate(val_loader, 0):
        questions = list(q)
        inputs = tokenizer(
            questions,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=args.question_len,
        )

        visual_features = visual_features.to(device)
        labels = labels.to(device)

        if args.transformer_ver == "pure_language":
            outputs = model(inputs)
        else:
            outputs = model(inputs, visual_features)

        loss = criterion(outputs, labels)
        total_loss += loss.item()

        _, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)
        label_true = labels.detach().cpu() if label_true is None else torch.cat((label_true, labels.detach().cpu()), 0)
        label_pred = predicted.detach().cpu() if label_pred is None else torch.cat((label_pred, predicted.detach().cpu()), 0)

    mAP, mAR, mAf1, wf1, acc = eval_for_f1_et_all(label_true, label_pred)
    print(f"Val(single): epoch: {epoch} loss: {total_loss:.6f} | Acc: {acc:.6f} | mAP: {mAP:.6f} | "
          f"mAR: {mAR:.6f} | mAf1: {mAf1:.6f} | wf1: {wf1:.6f}")

    return {
        "val/loss_sum": float(total_loss),
        "val/static/answer_acc": float(acc),
        "val/static/mAP": float(mAP),
        "val/static/mAR": float(mAR),
        "val/static/mF1": float(mAf1),
        "val/static/WF1": float(wf1),
    }


# -----------------------------
# Mixed-temporal: helpers & constants
# -----------------------------
ANSWER_TASK = "answer_cls"
COUNT_TASK  = "count_cls"

TEXT_HEAD_KEYS = ("boundary_text", "ordering_text", "extreme_text", "phase_text", "motion_text")

def _tensorize_long_from_list(labels, idx, device):
    out = []
    for j in idx:
        v = labels[j]
        if torch.is_tensor(v):
            out.append(int(v.detach().long().item()))
        else:
            out.append(int(v))
    return torch.tensor(out, dtype=torch.long, device=device)

def _tensorize_float_from_list(labels, idx, device):
    out = []
    for j in idx:
        v = labels[j]
        if torch.is_tensor(v):
            out.append(float(v.detach().cpu().item()))
        else:
            out.append(float(v))
    return torch.tensor(out, dtype=torch.float32, device=device)

def _accum_acc(pool: Dict[str, Dict[str, float]], key, pred, y):
    pool[key]["correct"] = pool.get(key, {}).get("correct", 0) + (pred == y).sum().item()
    pool[key]["n"] = pool.get(key, {}).get("n", 0) + y.size(0)

def _split_tasks(tasks: List[str]) -> Dict[str, List[int]]:
    mapping: Dict[str, List[int]] = {}
    for i, t in enumerate(tasks):
        mapping.setdefault(t, []).append(i)
    return mapping

def _task_kind(tname: str) -> str:
    if tname.endswith("_reg"):   return "reg"
    if tname.endswith("_bool"):  return "bin"
    if tname.endswith("_choice") or tname.endswith("_text") or tname.endswith("_cls"):
        return "multi"
    return "multi"

def _introspect_head_dims(model) -> Dict[str, int]:
    dims = {}
    if hasattr(model, "heads"):
        for k, v in model.heads.items():
            if isinstance(v, nn.Linear):
                dims[k] = int(v.out_features)
    return dims

def _assert_text_heads(model):
    dims = _introspect_head_dims(model)
    bad = []
    for k in TEXT_HEAD_KEYS:
        if k in dims and dims[k] <= 1:
            bad.append((k, dims[k]))
    if bad:
        msg = " | ".join([f"{k}={d}" for k, d in bad])
        raise RuntimeError(f"[FATAL] Degenerate text heads (out_features<=1): {msg}. "
                           f"Check your fixed_vocabs_json and args.*_vocab_size.")
    
def _parse_ts_ratio(r: str) -> Tuple[int, int]:
    """Parse 'a:b' string into (a, b) positive ints; fallback to (2,1)."""
    try:
        a, b = r.split(":")
        a = max(1, int(a.strip()))
        b = max(1, int(b.strip()))
        return a, b
    except Exception:
        return 2, 1

def _ratio_mix_iterator(loader_t: DataLoader,
                        loader_s: DataLoader,
                        t_per_cycle: int,
                        s_per_cycle: int):
    """
    Yield batches by ratio: per cycle, yield `t_per_cycle` temporal batches then `s_per_cycle` static batches.
    To avoid exhausting one stream earlier, we compute cycles = min(len(t)//t, len(s)//s) and drop the remainder.
    """
    if loader_s is None or s_per_cycle <= 0:
        # only temporal
        for b in loader_t:
            yield b
        return

    len_t = len(loader_t)
    len_s = len(loader_s)
    cycles = min(len_t // max(1, t_per_cycle), len_s // max(1, s_per_cycle))
    it_t = iter(loader_t)
    it_s = iter(loader_s)

    for _ in range(cycles):
        # temporal first
        for _ in range(t_per_cycle):
            yield next(it_t)
        # then static
        for _ in range(s_per_cycle):
            yield next(it_s)


def _finalize_epoch_stats(task_loss_sum, task_loss_n, bin_pool, multi_pool, cls_stats, reg_stats):
    """
    汇总：
      - 每任务 loss_avg
      - 每任务 acc / mae
      - 宏指标：bin_macro / multi_macro / cls_macro / duration_mae / overall_reg_mae
    """
    out = {}

    # loss avg & counts
    for k, s in task_loss_sum.items():
        n = max(1, task_loss_n.get(k, 0))
        out[f"loss/{k}_avg"] = float(s / n)
        out[f"data/{k}_n"]   = int(task_loss_n.get(k, 0))

    # per-task acc (bin+multi) & macro
    def _acc_from(pool):
        accs = []
        for k, v in pool.items():
            n = v.get("n", 0)
            if n > 0:
                out[f"temporal/{k}_acc"] = float(v["correct"] / n)
                out[f"data/{k}_n"] = int(n)
                accs.append(v["correct"] / n)
        return float(np.mean(accs)) if accs else 0.0

    bin_macro   = _acc_from(bin_pool)
    multi_macro = _acc_from(multi_pool)

    # answer / count
    answer_acc = (cls_stats[ANSWER_TASK]["correct"] / cls_stats[ANSWER_TASK]["n"]) if cls_stats[ANSWER_TASK]["n"] > 0 else 0.0
    count_acc  = (cls_stats[COUNT_TASK]["correct"]  / cls_stats[COUNT_TASK]["n"])  if cls_stats[COUNT_TASK]["n"]  > 0 else 0.0

    out["static/answer_acc"]       = float(answer_acc)
    out["temporal/counting_acc"]   = float(count_acc)
    out["temporal/bin_macro_acc"]  = float(bin_macro)
    out["temporal/multi_macro_acc"]= float(multi_macro)

    parts = []
    if cls_stats[COUNT_TASK]["n"] > 0: parts.append(count_acc)
    if bin_macro > 0:                parts.append(bin_macro)
    if multi_macro > 0:              parts.append(multi_macro)
    out["temporal/cls_macro_acc"] = float(np.mean(parts)) if parts else 0.0

    # regression
    dur_mae = (reg_stats["duration_reg"]["abs_err"] / reg_stats["duration_reg"]["n"]) if reg_stats["duration_reg"]["n"] > 0 else 0.0
    all_mae = (reg_stats["overall_reg"]["abs_err"]   / reg_stats["overall_reg"]["n"])   if reg_stats["overall_reg"]["n"]   > 0 else 0.0
    out["temporal/duration_mae"] = float(dur_mae)
    out["temporal/overall_reg_mae"] = float(all_mae)

    return out

def train_mixed(args,
                loader_t: DataLoader,
                loader_s: DataLoader,
                model,
                optimizer,
                device,
                epoch) -> Dict[str, float]:
    model.train()
    total_loss = 0.0

    # accumulators (unchanged)
    cls_stats = {
        ANSWER_TASK: {"n": 0, "correct": 0},
        COUNT_TASK:  {"n": 0, "correct": 0},
    }
    reg_stats = {
        "duration_reg": {"n": 0, "abs_err": 0.0},
        "overall_reg":  {"n": 0, "abs_err": 0.0},
    }
    bin_pool: Dict[str, Dict[str, float]] = {}
    multi_pool: Dict[str, Dict[str, float]] = {}

    task_loss_sum: Dict[str, float] = {}
    task_loss_n:   Dict[str, int]   = {}

    head_dims = _introspect_head_dims(model)

    # build ratio iterator
    t_per, s_per = _parse_ts_ratio(args.ts_ratio)
    batch_iter = _ratio_mix_iterator(loader_t, loader_s, t_per, s_per)

    for batch in batch_iter:
        text = {k: v.to(device) for k, v in batch["text"].items()}
        vis  = {k: v.to(device) for k, v in batch["visual"].items()}
        tasks:  List[str]        = batch["tasks"]
        labels                   = batch["labels"]
        metas:  List[Dict[str, Any]] = batch.get("meta", [None] * len(tasks))

        by_task = _split_tasks(tasks)

        # ----- loss normalization factors -----
        # count per task in this step
        task_sizes = {tname: len(idx) for tname, idx in by_task.items()}
        total_n_in_step = sum(task_sizes.values())
        num_tasks_present = len(task_sizes)

        def _norm_factor(tname: str) -> float:
            if args.loss_norm == "per_sample":
                return (task_sizes.get(tname, 0) / float(max(1, total_n_in_step)))
            elif args.loss_norm == "per_task":
                return (1.0 / float(max(1, num_tasks_present)))
            else:
                return 1.0  # "none"

        losses = []

        for tname, idx in by_task.items():
            if not idx:
                continue
            sub_text = {k: v[idx] for k, v in text.items()}
            sub_vis  = {k: v[idx] for k, v in vis.items()}
            sub_meta = [metas[i] for i in idx]

            kind = _task_kind(tname)

            if tname == ANSWER_TASK:
                keep, lids = [], []
                for j in idx:
                    lab = labels[j]
                    if torch.is_tensor(lab) and lab.dtype == torch.long:
                        keep.append(j); lids.append(int(lab.item()))
                    elif isinstance(lab, (int, np.integer)):
                        keep.append(j); lids.append(int(lab))
                if not keep:
                    continue
                take = [p for p, _ in enumerate(idx) if idx[p] in keep]
                sub_text = {k: v[take] for k, v in sub_text.items()}
                sub_vis  = {k: v[take] for k, v in sub_vis.items()}
                sub_meta = [sub_meta[p] for p in take]
                y = torch.tensor(lids, dtype=torch.long, device=device)

                out = model(sub_text, visual=sub_vis, task=ANSWER_TASK, labels=y, meta=sub_meta)
                base_w = args.w_answer
                nf = _norm_factor(tname)
                losses.append(base_w * nf * out["loss"])
                total_loss += float(out["loss"].item())

                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + float(out["loss"].item())
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                with torch.no_grad():
                    pred = out["logits"].argmax(dim=-1)
                    _accum_acc(cls_stats, ANSWER_TASK, pred, y)

            elif tname == COUNT_TASK:
                y = _tensorize_long_from_list(labels, idx, device)
                out = model(sub_text, visual=sub_vis, task=COUNT_TASK, labels=y, meta=sub_meta)
                base_w = args.w_count
                nf = _norm_factor(tname)
                losses.append(base_w * nf * out["loss"])
                total_loss += float(out["loss"].item())

                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + float(out["loss"].item())
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                with torch.no_grad():
                    pred = out["logits"].argmax(dim=-1)
                    _accum_acc(cls_stats, COUNT_TASK, pred, y)

            elif kind == "reg":
                y = _tensorize_float_from_list(labels, idx, device)
                out = model(sub_text, visual=sub_vis, task=tname, labels=y, meta=sub_meta)
                base_w = (args.w_duration if tname == "duration_reg" else args.w_temporal_cls)
                nf = _norm_factor(tname)
                losses.append(base_w * nf * out["loss"])
                total_loss += float(out["loss"].item())

                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + float(out["loss"].item())
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                with torch.no_grad():
                    pred = out["logits"].view(-1)
                    ae = torch.abs(pred - y).sum().item()
                    reg_stats["overall_reg"]["abs_err"] += ae
                    reg_stats["overall_reg"]["n"] += y.size(0)
                    if tname == "duration_reg":
                        reg_stats["duration_reg"]["abs_err"] += ae
                        reg_stats["duration_reg"]["n"] += y.size(0)

            elif kind == "bin":
                y = _tensorize_long_from_list(labels, idx, device)
                out = model(sub_text, visual=sub_vis, task=tname, labels=y, meta=sub_meta)
                base_w = args.w_temporal_cls
                nf = _norm_factor(tname)
                losses.append(base_w * nf * out["loss"])
                total_loss += float(out["loss"].item())

                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + float(out["loss"].item())
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                with torch.no_grad():
                    pred = out["logits"].argmax(dim=-1)
                    if tname not in bin_pool:
                        bin_pool[tname] = {"n": 0, "correct": 0}
                    _accum_acc(bin_pool, tname, pred, y)

            else:  # multi
                y = _tensorize_long_from_list(labels, idx, device)
                out = model(sub_text, visual=sub_vis, task=tname, labels=y, meta=sub_meta)
                base_w = args.w_temporal_cls
                nf = _norm_factor(tname)
                losses.append(base_w * nf * out["loss"])
                total_loss += float(out["loss"].item())

                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + float(out["loss"].item())
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                with torch.no_grad():
                    pred = out["logits"].argmax(dim=-1)
                    if tname not in multi_pool:
                        multi_pool[tname] = {"n": 0, "correct": 0}
                    _accum_acc(multi_pool, tname, pred, y)

        if not losses:
            continue
        loss = torch.stack(losses).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # summarize (unchanged)
    out = {"train/loss_sum": float(total_loss)}
    sums = _finalize_epoch_stats(task_loss_sum, task_loss_n, bin_pool, multi_pool, cls_stats, reg_stats)
    for k, v in sums.items():
        out[f"train/{k}"] = v

    for k, d in head_dims.items():
        out[f"train/model/{k}_classes"] = int(d)

    print(
        f"Train(mixed): epoch={epoch} | loss_sum={total_loss:.2f} | "
        f"static/answer_acc={out['train/static/answer_acc']:.4f} | "
        f"temporal/counting_acc={out['train/temporal/counting_acc']:.4f} | "
        f"temporal/bin_macro_acc={out['train/temporal/bin_macro_acc']:.4f} | "
        f"temporal/multi_macro_acc={out['train/temporal/multi_macro_acc']:.4f} | "
        f"temporal/cls_macro_acc={out['train/temporal/cls_macro_acc']:.4f} | "
        f"temporal/duration_mae={out['train/temporal/duration_mae']:.4f}"
    )
    return out


@torch.no_grad()
def validate_mixed(args, loader, model, device, epoch) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0

    cls_stats = {
        ANSWER_TASK: {"n": 0, "correct": 0},
        COUNT_TASK:  {"n": 0, "correct": 0},
    }
    reg_stats = {
        "duration_reg": {"n": 0, "abs_err": 0.0},
        "overall_reg":  {"n": 0, "abs_err": 0.0},
    }
    bin_pool: Dict[str, Dict[str, float]] = {}
    multi_pool: Dict[str, Dict[str, float]] = {}

    task_loss_sum: Dict[str, float] = {}
    task_loss_n:   Dict[str, int]   = {}

    head_dims = _introspect_head_dims(model)

    for batch in loader:
        text = {k: v.to(device) for k, v in batch["text"].items()}
        vis  = {k: v.to(device) for k, v in batch["visual"].items()}
        tasks:  List[str]        = batch["tasks"]
        labels                   = batch["labels"]
        metas:  List[Dict[str, Any]] = batch.get("meta", [None] * len(tasks))

        by_task = _split_tasks(tasks)

        for tname, idx in by_task.items():
            if not idx:
                continue
            sub_text = {k: v[idx] for k, v in text.items()}
            sub_vis  = {k: v[idx] for k, v in vis.items()}
            sub_meta = [metas[i] for i in idx]

            kind = _task_kind(tname)

            if tname == ANSWER_TASK:
                keep, lids = [], []
                for j in idx:
                    lab = labels[j]
                    if torch.is_tensor(lab) and lab.dtype == torch.long:
                        keep.append(j); lids.append(int(lab.item()))
                    elif isinstance(lab, (int, np.integer)):
                        keep.append(j); lids.append(int(lab))
                if not keep:
                    continue
                take = [p for p, _ in enumerate(idx) if idx[p] in keep]
                sub_text = {k: v[take] for k, v in sub_text.items()}
                sub_vis  = {k: v[take] for k, v in sub_vis.items()}
                sub_meta = [sub_meta[p] for p in take]
                y = torch.tensor(lids, dtype=torch.long, device=device)

                out = model(sub_text, visual=sub_vis, task=ANSWER_TASK, labels=y, meta=sub_meta)
                lval = float(out["loss"].item())
                total_loss += lval

                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + lval
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                pred = out["logits"].argmax(dim=-1)
                _accum_acc(cls_stats, ANSWER_TASK, pred, y)

            elif tname == COUNT_TASK:
                y = _tensorize_long_from_list(labels, idx, device)
                out = model(sub_text, visual=sub_vis, task=COUNT_TASK, labels=y, meta=sub_meta)
                lval = float(out["loss"].item())
                total_loss += lval

                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + lval
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                pred = out["logits"].argmax(dim=-1)
                _accum_acc(cls_stats, COUNT_TASK, pred, y)

            elif kind == "reg":
                y = _tensorize_float_from_list(labels, idx, device)
                out = model(sub_text, visual=sub_vis, task=tname, labels=y, meta=sub_meta)
                lval = float(out["loss"].item())
                total_loss += lval
                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + lval
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                pred = out["logits"].view(-1)
                ae = torch.abs(pred - y).sum().item()
                reg_stats["overall_reg"]["abs_err"] += ae
                reg_stats["overall_reg"]["n"] += y.size(0)
                if tname == "duration_reg":
                    reg_stats["duration_reg"]["abs_err"] += ae
                    reg_stats["duration_reg"]["n"] += y.size(0)

            elif kind == "bin":
                y = _tensorize_long_from_list(labels, idx, device)
                out = model(sub_text, visual=sub_vis, task=tname, labels=y, meta=sub_meta)
                lval = float(out["loss"].item())
                total_loss += lval
                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + lval
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                pred = out["logits"].argmax(dim=-1)
                if tname not in bin_pool:
                    bin_pool[tname] = {"n": 0, "correct": 0}
                _accum_acc(bin_pool, tname, pred, y)

            else:
                y = _tensorize_long_from_list(labels, idx, device)
                out = model(sub_text, visual=sub_vis, task=tname, labels=y, meta=sub_meta)
                lval = float(out["loss"].item())
                total_loss += lval
                task_loss_sum[tname] = task_loss_sum.get(tname, 0.0) + lval
                task_loss_n[tname]   = task_loss_n.get(tname, 0) + y.size(0)

                pred = out["logits"].argmax(dim=-1)
                if tname not in multi_pool:
                    multi_pool[tname] = {"n": 0, "correct": 0}
                _accum_acc(multi_pool, tname, pred, y)

    out = {
        "val/loss_sum": float(total_loss),
    }
    sums = _finalize_epoch_stats(task_loss_sum, task_loss_n, bin_pool, multi_pool, cls_stats, reg_stats)
    for k, v in sums.items():
        out[f"val/{k}"] = v

    # 记录 head 维度（类别数）
    for k, d in head_dims.items():
        out[f"val/model/{k}_classes"] = int(d)

    print(
        f"Val(mixed): epoch={epoch} | loss_sum={total_loss:.2f} | "
        f"static/answer_acc={out['val/static/answer_acc']:.4f} | "
        f"temporal/counting_acc={out['val/temporal/counting_acc']:.4f} | "
        f"temporal/bin_macro_acc={out['val/temporal/bin_macro_acc']:.4f} | "
        f"temporal/multi_macro_acc={out['val/temporal/multi_macro_acc']:.4f} | "
        f"temporal/cls_macro_acc={out['val/temporal/cls_macro_acc']:.4f} | "
        f"temporal/duration_mae={out['val/temporal/duration_mae']:.4f}"
    )
    return out


# -----------------------------
# Main
# -----------------------------
def _load_fixed_vocabs(path: str, logger) -> Dict[str, List[str]]:
    NEED = ["extreme", "phase_transition", "boundary", "ordering", "motion"]
    vocabs = {}
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for k in NEED:
                lst = list(data.get(k, []))
                if not lst:
                    lst = ["<UNK>"]
                    logger.warning(f"[fixed_vocabs] '{k}' missing/empty in {path}; fallback to ['<UNK>']")
                else:
                    if lst[0] != "<UNK>":
                        if "<UNK>" in lst:
                            lst = ["<UNK>"] + [x for x in lst if x != "<UNK>"]
                        else:
                            lst = ["<UNK>"] + lst
                vocabs[k] = lst
            return vocabs
        except Exception as e:
            logger.error(f"[fixed_vocabs] failed to load {path}: {e}")
    # fallback
    logger.warning("[fixed_vocabs] using minimal fallback for all keys")
    for k in NEED:
        vocabs[k] = ["<UNK>"]
    return vocabs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SSGQA VQA Training (single-frame & mixed-temporal)")

    # Model params
    parser.add_argument("--emb_dim", type=int, default=300)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder_layers", type=int, default=6)

    # Temporal heads & vocabs
    parser.add_argument("--w_temporal_cls", type=float, default=1.0)
    parser.add_argument("--extreme_vocab_size", type=int, default=0)
    parser.add_argument("--phase_vocab_size", type=int, default=0)

    # Training params
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=100)

    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/final_vb/")
    parser.add_argument("--dataset_cat", default="None")

    parser.add_argument("--transformer_ver", default="vb", help="vb/pure_visual/pure_language")
    parser.add_argument("--tokenizer_ver", default="v2", help="v2/v3/v4")
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--temporal_size", type=int, default=3)
    parser.add_argument("--question_len", type=int, default=77)
    parser.add_argument("--num_class", type=int, default=51)
    parser.add_argument("--validate", type=lambda x: str(x).lower() == "true", default=False)

    # Dataset type
    parser.add_argument("--dataset_type", default="mixed-temporal",
                        help="ssg-qa-roi_coord / ssg-qa-roi-analysis / mixed-temporal")
    # whether include static qa from ssgvqa
    parser.add_argument("--include_static", type=lambda x: str(x).lower()=="true", default=True)

    # paths
    parser.add_argument("--ssg_qa_root", default=r"E:\LJ\datasets\SSGVQA\ssg-qa\ssg-qa")
    parser.add_argument("--stsg_qa_root", default=r"E:\LJ\datasets\STSG_QA")
    parser.add_argument("--feature_root", default=r"E:\LJ\datasets\SSGVQA")
    parser.add_argument("--fixed_vocabs_json", default=None,
                        help="json with keys: extreme, phase_transition, boundary, ordering, motion")

    # temporal hyper
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--count_bins", type=int, default=20)  # 0..19

    # multi-task weights
    parser.add_argument("--w_answer", type=float, default=1.0)
    parser.add_argument("--w_count", type=float, default=1.0)
    parser.add_argument("--w_duration", type=float, default=1.0)

    parser.add_argument("--boundary_text_size", type=int, default=0)
    parser.add_argument("--ordering_text_size", type=int, default=0)
    parser.add_argument("--motion_classes", type=int, default=8)

    # wandb
    parser.add_argument("--use_wandb", type=lambda x: str(x).lower() == "true", default=False)
    parser.add_argument("--wandb_project", default="SSGVQA")
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_mode", default="online")  # online/offline
    parser.add_argument("--run_name", default=None)

    # rationale
    parser.add_argument("--use_rationale", type=lambda x: str(x).lower()=="true", default=True)
    parser.add_argument("--rationale_mode", default="append", choices=["append","prefix","pair"])
    parser.add_argument("--rationale_tag", default="Rationale:")
    parser.add_argument("--rationale_dropout", type=float, default=0.2)
    parser.add_argument("--text_max_len", type=int, default=320)

    parser.add_argument("--ts_ratio", type=str, default="2:1",
                    help="temporal:static batch ratio for mixed training, e.g., '2:1'. Ignored if include_static=False.")
    parser.add_argument("--loss_norm", type=str, default="per_sample",
                        choices=["per_sample", "per_task", "none"],
                        help="how to normalize per-task losses within a step. "
                            "'per_sample' weights each task loss by (#samples_of_task / total_samples_in_step); "
                            "'per_task' gives equal weight to each task present in the step; "
                            "'none' keeps the previous behavior.")
    

    # static (for multi-frames)
    parser.add_argument("--static_multiframe", type=lambda x: str(x).lower()=="true",
                        default=False,
                        help="If True, SSGVQAFrameDataset will load multiple frames per static QA sample.")

    parser.add_argument("--static_frames", type=int, default=12,
                        help="Number of frames per static QA sample when static_multiframe=True.")

    # whether use temporal QA (only for static-only mode)
    parser.add_argument("--temporal_enabled", type=lambda x: str(x).lower()=="true",
                        default=True,
                        help="If False (and dataset_type=mixed-temporal), disable STSG temporal QA and train on static QA only.")

    args = parser.parse_args()

    TEST_SET = {"VID22", "VID74", "VID60", "VID02", "VID43"}

    if args.use_wandb:
        os.environ["WANDB_MODE"] = args.wandb_mode
        wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=args.run_name or os.getenv("WANDB_NAME"),
            group=os.getenv("WANDB_RUN_GROUP"),
            tags=[t.strip() for t in os.getenv("WANDB_TAGS","").split(",") if t.strip()],  # <--- 可选
            config=vars(args),
            dir=args.checkpoint_dir,
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*",   step_metric="epoch")

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    print("device =", device)

    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0

    # -----------------------------
    # Datasets & Tokenizer
    # -----------------------------
    if args.dataset_type == "ssg-qa-roi_coord":
        if args.tokenizer_ver == "v2":
            tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.tokenizer_ver == "v3":
            tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", do_lower_case=True)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        train_seq = ["VID73","VID40","VID62","VID42","VID29","VID56","VID50","VID78","VID66","VID13",
                     "VID52","VID06","VID36","VID05","VID12","VID26","VID68","VID32","VID49","VID65",
                     "VID47","VID04","VID23","VID79","VID51","VID10","VID57","VID75","VID25","VID14",
                     "VID15","VID08","VID80","VID27","VID70"]
        val_seq = ["VID18", "VID48", "VID01", "VID35", "VID31"]
        test_seq = ["VID22", "VID74", "VID60", "VID02", "VID43"]

        folder_head = "./data/"
        folder_tail = "/*.txt"

        train_dataset = SSGVQAClassification_full_roi_coord(train_seq, folder_head, folder_tail, patch_size=args.patch_size)
        val_dataset   = SSGVQAClassification_full_roi_coord(val_seq, folder_head, folder_tail, patch_size=args.patch_size)
        test_dataset  = SSGVQAClassification_full_roi_coord(test_seq, folder_head, folder_tail, patch_size=args.patch_size)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers)
        val_dataloader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        test_dataloader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

        args.num_class = 51

    elif args.dataset_type == "ssg-qa-roi-analysis":
        if args.tokenizer_ver == "v2":
            tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        else:
            tokenizer = BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", do_lower_case=True)

        test_seq = ["VID22", "VID74", "VID02", "VID60", "VID43"]
        folder_head = "./data/"; folder_tail = "/*.txt"

        test_sets = [
            SSGVQAClassification_full_roi_analysis(test_seq, folder_head, folder_tail, ana_type=["zero_hop.json"],  patch_size=args.patch_size),
            SSGVQAClassification_full_roi_analysis(test_seq, folder_head, folder_tail, ana_type=["one_hop.json"],   patch_size=args.patch_size),
            SSGVQAClassification_full_roi_analysis(test_seq, folder_head, folder_tail, ana_type=["single_and.json"],patch_size=args.patch_size),
            SSGVQAClassification_full_roi_analysis(test_seq, folder_head, folder_tail, ana_type=["query_color","query_type","query_location"], patch_size=args.patch_size),
            SSGVQAClassification_full_roi_analysis(test_seq, folder_head, folder_tail, ana_type=["query_component"],patch_size=args.patch_size),
            SSGVQAClassification_full_roi_analysis(test_seq, folder_head, folder_tail, ana_type=["exist"],          patch_size=args.patch_size),
            SSGVQAClassification_full_roi_analysis(test_seq, folder_head, folder_tail, ana_type=["count"],          patch_size=args.patch_size),
        ]
        test_dataloader = [DataLoader(ds, batch_size=args.batch_size, shuffle=False) for ds in test_sets]
        args.num_class = 51

    elif args.dataset_type == "mixed-temporal":
        # Tokenizer
        if args.tokenizer_ver in ("v2", "v3"):
            tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",
                                                      do_lower_case=(args.tokenizer_ver == "v3"))
        else:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        logger = build_logger(log_file=os.path.join(args.checkpoint_dir, "dataset_mixed.log"))

        train_vids = [
            "VID73","VID40","VID62","VID42","VID29","VID56","VID50","VID78","VID66","VID13",
            "VID52","VID06","VID36","VID05","VID12","VID26","VID68","VID32","VID49","VID65",
            "VID47","VID04","VID23","VID79","VID51","VID10","VID57","VID75","VID25","VID14",
            "VID15","VID08","VID80","VID27","VID70"
        ]
        val_vids  = ["VID18", "VID48", "VID01", "VID35", "VID31"]
        test_vids = ["VID22", "VID74", "VID60", "VID02", "VID43"]

        print(f"[Split fixed] train={len(train_vids)} vids, val={len(val_vids)} vids (held-out test: {test_vids})")

        # single-frame (SSG) only when include_static=True
        ds_single_train = None
        ds_single_val   = None
        if args.include_static:
            ds_single_train = SSGVQAFrameDataset(
                qa_root=args.ssg_qa_root,
                folder_head=args.feature_root,
                seq=train_vids,
                logger=logger,
                multi_frames=args.static_multiframe,
                frames_per_sample=args.static_frames,
            )
            ds_single_val = SSGVQAFrameDataset(
                qa_root=args.ssg_qa_root,
                folder_head=args.feature_root,
                seq=val_vids,
                label_vocab=ds_single_train.label_vocab,
                logger=logger,
                multi_frames=args.static_multiframe,
                frames_per_sample=args.static_frames,
            )

        # temporal (STSG)
        def _exist_in(root: str, vids: list) -> list:
            kept = [v for v in vids if os.path.isdir(os.path.join(root, v))]
            missing = [v for v in vids if v not in kept]
            if len(missing):
                logger.warning(f"[STSG_QA] Missing videos (skipped): {missing}")
            return kept

        # stsg_train_vids = _exist_in(args.stsg_qa_root, train_vids)
        # stsg_val_vids   = _exist_in(args.stsg_qa_root, val_vids)

        ALL_TEMPORAL_CATS = [
            "count", "duration", "ordering", "extreme",
            "boundary", "motion", "concurrency", "phase_transition"
        ]

        default_vocab_path = os.path.join(args.stsg_qa_root, "vocabs_train.json")
        vocabs_path = args.fixed_vocabs_json or default_vocab_path
        fixed_vocabs = _load_fixed_vocabs(vocabs_path, logger)

        ds_temporal_train = None
        ds_temporal_val   = None

        if args.temporal_enabled:
            stsg_train_vids = _exist_in(args.stsg_qa_root, train_vids)
            stsg_val_vids   = _exist_in(args.stsg_qa_root, val_vids)

            ds_temporal_train = STSGTemporalQADataset(
                qa_root=args.stsg_qa_root,
                folder_head=args.feature_root,
                vids=stsg_train_vids,
                allowed_categories=ALL_TEMPORAL_CATS,
                max_frames=args.max_frames,
                fps=args.fps,
                logger=logger,
                fixed_vocabs=fixed_vocabs
            )
            ds_temporal_val = STSGTemporalQADataset(
                qa_root=args.stsg_qa_root,
                folder_head=args.feature_root,
                vids=stsg_val_vids,
                allowed_categories=ALL_TEMPORAL_CATS,
                max_frames=args.max_frames,
                fps=args.fps,
                logger=logger,
                fixed_vocabs=fixed_vocabs
            )
        else:
            logger.warning("[mixed-temporal] temporal_enabled=False -> no STSG temporal QA will be used.")

        # vocab sizes from fixed_vocabs
        args.extreme_vocab_size  = len(fixed_vocabs["extreme"])
        args.phase_vocab_size    = len(fixed_vocabs["phase_transition"])
        args.boundary_text_size  = len(fixed_vocabs["boundary"])
        args.ordering_text_size  = len(fixed_vocabs["ordering"])
        args.motion_classes      = len(fixed_vocabs["motion"])

        args.count_bins = max(int(args.count_bins), 20)

        print(f"[VOCABS] extreme={args.extreme_vocab_size}, phase={args.phase_vocab_size}, "
              f"boundary_text={args.boundary_text_size}, ordering_text={args.ordering_text_size}, "
              f"motion_classes={args.motion_classes}, count_bins={args.count_bins}")


        # collate (temporal + rationale)
        collate_train = make_collate_fn(
            tokenizer,
            use_rationale=args.use_rationale,
            rationale_mode=args.rationale_mode,
            rationale_tag=args.rationale_tag,
            rationale_dropout_p=args.rationale_dropout,
            max_length=args.text_max_len,
        )
        collate_val = make_collate_fn(
            tokenizer,
            use_rationale=args.use_rationale,
            rationale_mode=args.rationale_mode,
            rationale_tag=args.rationale_tag,
            rationale_dropout_p=0.0,
            max_length=args.text_max_len,
        )

        # --- build separate loaders for training ---
        train_loader_temporal = None
        train_loader_static   = None

        if args.temporal_enabled:
            # 1) 有 temporal QA
            if ds_temporal_train is None:
                raise RuntimeError("[mixed-temporal] temporal_enabled=True but no temporal dataset was built.")

            train_loader_temporal = DataLoader(
                ds_temporal_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(0, min(2, int(args.workers))),
                collate_fn=collate_train,
                pin_memory=True,
            )

            if args.include_static and ds_single_train is not None:
                # hybrid：temporal + static
                args.num_class = len(ds_single_train.label_vocab)
                train_loader_static = DataLoader(
                    ds_single_train,
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=max(0, min(2, int(args.workers))),
                    collate_fn=collate_train,
                    pin_memory=True,
                )
            else:
                # pure temporal-only 模式：没有静态 QA 头
                args.num_class = 0

        else:
            # 2) static-only 模式：只用 SSG 静态 QA（可以 multi-frame）
            if not (args.include_static and ds_single_train is not None):
                raise RuntimeError("[mixed-temporal] temporal_enabled=False requires include_static=True and valid SSG QA data.")
            args.num_class = len(ds_single_train.label_vocab)

            # 注意：这里我们把静态 QA 放在 loader_t（temporal 通道），loader_s=None
            # train_mixed 会退化为「只有一个任务 answer_cls」的训练
            train_loader_temporal = DataLoader(
                ds_single_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=max(0, min(2, int(args.workers))),
                collate_fn=collate_train,
                pin_memory=True,
            )
            train_loader_static = None

        # validation
        val_datasets = []

        if args.include_static and ds_single_val is not None:
            val_datasets.append(ds_single_val)

        if args.temporal_enabled and ds_temporal_val is not None:
            val_datasets.append(ds_temporal_val)

        if not val_datasets:
            raise RuntimeError("[mixed-temporal] No validation dataset constructed. Check include_static / temporal_enabled / paths.")

        val_dataset = val_datasets[0] if len(val_datasets) == 1 else ConcatDataset(val_datasets)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=max(0, min(2, int(args.workers))),
            collate_fn=collate_val,
            pin_memory=True
        )

        test_dataloader = val_dataloader

    else:
        raise NotImplementedError(f"Unknown dataset_type: {args.dataset_type}")

    # -----------------------------
    # Model & Optimizer
    # -----------------------------
    final_args = {
        "emb_dim": args.emb_dim,
        "n_heads": args.n_heads,
        "dropout": args.dropout,
        "encoder_layers": args.encoder_layers,
    }

    if args.dataset_type == "mixed-temporal":
        print("loading MultiTaskSSGVQA (temporal-enabled VisualBERT)")
        model = MultiTaskSSGVQA(
            tokenizer_vocab_size=getattr(tokenizer, "vocab_size", len(tokenizer)),
            # answer_vocab_size=args.num_class,
            answer_vocab_size=(args.num_class if (args.include_static and args.num_class > 1) else 0),
            count_bins=args.count_bins,
            max_video_frames=args.max_frames,
            extreme_vocab_size=args.extreme_vocab_size,
            phase_vocab_size=args.phase_vocab_size,
            boundary_text_size=args.boundary_text_size,
            ordering_text_size=args.ordering_text_size,
            motion_classes=args.motion_classes,
        )
        # 退化检查（文本头类别数必须 > 1）
        _assert_text_heads(model)
    else:
        if args.transformer_ver == "vb":
            print("loading VisualBert")
            model = VisualBertClassification(
                vocab_size=len(tokenizer),
                layers=args.encoder_layers,
                n_heads=args.n_heads,
                num_class=args.num_class,
            )
        else:
            raise NotImplementedError

    if args.checkpoint is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = 1
    else:
        print("loading from checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint.get("epoch", 1)
        epochs_since_improvement = checkpoint.get("epochs_since_improvement", 0)
        best_Acc = checkpoint.get("Acc", 0.0)
        model_dict = checkpoint.get("model", checkpoint)

        incompatible = model.load_state_dict(model_dict, strict=False)
        try:
            # IncompatibleKeys(missing_keys=[...], unexpected_keys=[...])
            print("[load_state_dict] missing:", getattr(incompatible, "missing_keys", []))
            print("[load_state_dict] unexpected:", getattr(incompatible, "unexpected_keys", []))
        except Exception:
            pass

        optimizer = checkpoint.get("optimizer", torch.optim.Adam(model.parameters(), lr=args.lr))
        final_args = checkpoint.get("final_args", final_args)

    model = model.to(device)

    print(final_args)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("model params:", pytorch_total_params)

    criterion = nn.CrossEntropyLoss().to(device)  # only for single-frame path

    # -----------------------------
    # Train / Validate
    # -----------------------------
    if args.validate:
        if args.dataset_type == "mixed-temporal":
            v_metrics = validate_mixed(
                args=args, loader=val_dataloader, model=model, device=device, epoch=args.epochs - 1
            )
            if args.use_wandb:
                payload = {"epoch": args.epochs - 1}
                payload.update(v_metrics)
                wandb.log(payload, step=args.epochs - 1)

        elif "analysis" in args.dataset_type:
            for i, loader in enumerate(test_dataloader):
                v_metrics = validate_single(
                    args=args, val_loader=loader, model=model, criterion=criterion,
                    epoch=(args.epochs - 1), tokenizer=tokenizer, device=device,
                    save_output=True
                )
                if args.use_wandb:
                    payload = {"epoch": args.epochs - 1}
                    payload.update(v_metrics)
                    wandb.log(payload, step=args.epochs - 1)
        else:
            v_metrics = validate_single(
                args=args, val_loader=val_dataloader, model=model, criterion=criterion,
                epoch=(args.epochs - 1), tokenizer=tokenizer, device=device,
                save_output=True
            )
            if args.use_wandb:
                payload = {"epoch": args.epochs - 1}
                payload.update(v_metrics)
                wandb.log(payload, step=args.epochs - 1)
    else:
        best_metric = best_results[0]
        for epoch in range(start_epoch, args.epochs + 1):
            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)

            if args.dataset_type == "mixed-temporal":
                # tr_metrics = train_mixed(
                #     args=args, loader=train_dataloader, model=model,
                #     optimizer=optimizer, device=device, epoch=epoch
                # )
                tr_metrics = train_mixed(
                    args=args,
                    loader_t=train_loader_temporal,
                    loader_s=train_loader_static,   # can be None if include_static=False
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    epoch=epoch
                )
                v_metrics = validate_mixed(
                    args=args, loader=val_dataloader, model=model,
                    device=device, epoch=epoch
                )

                if args.use_wandb:
                    payload = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
                    payload.update(tr_metrics)
                    payload.update(v_metrics)
                    wandb.log(payload, step=epoch)

                if args.temporal_enabled:
                    # 原来的逻辑：用 temporal 分类宏平均指标选最好
                    current_score = v_metrics.get("val/temporal/cls_macro_acc", 0.0)
                else:
                    # static-only：用静态 answer head 的准确率选 best epoch
                    current_score = v_metrics.get("val/static/answer_acc", 0.0)
                if current_score >= best_metric:
                    epochs_since_improvement = 0
                    best_metric = current_score
                    best_epoch[0] = epoch
                    print(f"Best epoch: {best_epoch[0]} | Best AnsAcc: {best_metric:.6f}")
                    save_clf_checkpoint(
                        args.checkpoint_dir, epoch, epochs_since_improvement,
                        model, optimizer, best_metric, final_args
                    )
                else:
                    epochs_since_improvement += 1
                    print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")

            else:
                tr_metrics = train_single_epoch(
                    args=args, train_dataloader=train_dataloader, model=model,
                    criterion=criterion, optimizer=optimizer, epoch=epoch,
                    tokenizer=tokenizer, device=device
                )
                v_metrics = validate_single(
                    args=args, val_loader=val_dataloader, model=model,
                    criterion=criterion, epoch=epoch, tokenizer=tokenizer,
                    device=device
                )

                if args.use_wandb:
                    payload = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
                    payload.update(tr_metrics)
                    payload.update(v_metrics)
                    wandb.log(payload, step=epoch)

                current_score = v_metrics.get("val/static/answer_acc", 0.0)
                if current_score >= best_metric:
                    epochs_since_improvement = 0
                    best_metric = current_score
                    best_epoch[0] = epoch
                    print(f"Best epoch: {best_epoch[0]} | Best acc: {best_metric:.6f}")
                    save_clf_checkpoint(
                        args.checkpoint_dir, epoch, epochs_since_improvement,
                        model, optimizer, best_metric, final_args
                    )
                else:
                    epochs_since_improvement += 1
                    print(f"\nEpochs since last improvement: {epochs_since_improvement}\n")
