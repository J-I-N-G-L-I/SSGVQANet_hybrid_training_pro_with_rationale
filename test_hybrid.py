# # -*- coding: utf-8 -*-
import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer

from utils.mixed_datasets import STSGTemporalQADataset, build_logger
from utils.collate_temporal import make_collate_fn
from models.multitask_ssgvqa import MultiTaskSSGVQA


def str2bool(x: str) -> bool:
    return str(x).lower() in {"1", "true", "t", "yes", "y"}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stsg_qa_root", required=True)
    p.add_argument("--feature_root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--temporal_videos", nargs="+", required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--count_bins", type=int, default=20)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--fps", type=float, default=1.0)

    p.add_argument("--tokenizer_ver", default="v2", choices=["v2", "v3", "v4"])
    p.add_argument("--text_max_len", type=int, default=320)
    p.add_argument("--use_rationale", type=str2bool, default=True)

    p.add_argument("--vocabs_json",
                   default="/users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/utils/vocabs_train.json",
                   help="path to vocabs_train.json; optional but recommended")

    return p.parse_args()


def select_tokenizer(tokenizer_ver: str):
    if tokenizer_ver in ("v2", "v3"):
        return BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT",
            do_lower_case=(tokenizer_ver == "v3")
        )
    else:
        return AutoTokenizer.from_pretrained("bert-base-uncased")


def infer_heads_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, Optional[int]]:
    def rows(name):
        w = state.get(f"{name}.weight", None)
        return int(w.shape[0]) if w is not None else None
    return {
        "answer_vocab_size":   rows("heads.answer_cls"),
        "count_bins":          rows("heads.count_cls"),
        "motion_classes":      rows("heads.motion_text"),
        "extreme_vocab_size":  rows("heads.extreme_text"),
        "phase_vocab_size":    rows("heads.phase_text"),
        "boundary_text_size":  rows("heads.boundary_text"),
        "ordering_text_size":  rows("heads.ordering_text"),
    }


def build_model_from_ckpt(ckpt_path: str, tokenizer_vocab_size: int,
                          max_video_frames: int, default_count_bins: int) -> MultiTaskSSGVQA:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    sizes = infer_heads_from_state(state)

    model = MultiTaskSSGVQA(
        tokenizer_vocab_size=tokenizer_vocab_size,
        answer_vocab_size=(sizes["answer_vocab_size"] or 51),
        count_bins=(sizes["count_bins"] or default_count_bins),
        max_video_frames=max_video_frames,
        motion_classes=(sizes["motion_classes"] or 8),
        extreme_vocab_size=(sizes["extreme_vocab_size"] or 0),
        phase_vocab_size=(sizes["phase_vocab_size"] or 0),
        boundary_text_size=(sizes["boundary_text_size"] or 0),
        ordering_text_size=(sizes["ordering_text_size"] or 0),
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  missing keys (truncated):", missing[:10], "...")
    if unexpected:
        print("  unexpected keys (truncated):", unexpected[:10], "...")
    return model


def safe_div(a: float, b: float) -> float:
    return a / b if b > 0 else 0.0


def binary_prf_auroc(y_true: List[int], y_pred: List[int], y_score_pos: List[float]) -> Dict[str, float]:
    """Compute acc/precision/recall/f1/auroc for binary with positive=1."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    y_score = np.asarray(y_score_pos, dtype=np.float64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = safe_div(tp + tn, len(y_true))
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0


    if (y_true.min() == y_true.max()):

        auroc = 0.5
    else:
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        score_sorted = y_score[order]
        P = float((y_true == 1).sum())
        N = float((y_true == 0).sum())

        tp_cum = 0.0
        fp_cum = 0.0
        prev = None
        roc = [(0.0, 0.0)]
        for i in range(len(y_sorted)):
            s = score_sorted[i]
            if prev is None or s != prev:
                roc.append((safe_div(fp_cum, N), safe_div(tp_cum, P)))
                prev = s
            if y_sorted[i] == 1:
                tp_cum += 1.0
            else:
                fp_cum += 1.0
        roc.append((safe_div(fp_cum, N), safe_div(tp_cum, P)))


        auroc = 0.0
        for i in range(1, len(roc)):
            x0, y0 = roc[i - 1]
            x1, y1 = roc[i]
            auroc += (x1 - x0) * (y0 + y1) / 2.0
        auroc = float(auroc)

    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": float(auroc),
    }


def topk_hit(logits: torch.Tensor, y_true: torch.Tensor, k: int) -> int:
    """#hits where true label is within top-k."""
    if logits.ndim != 2:
        logits = logits.view(logits.size(0), -1)
    k = min(k, logits.size(1))
    topk = torch.topk(logits, k=k, dim=1).indices  # (B,k)
    y = y_true.view(-1, 1).expand(-1, k)
    return int((topk == y).any(dim=1).sum().item())


def macro_f1_multiclass(y_true: List[int], y_pred: List[int]) -> float:
    """Macro-F1 over classes present in ground truth."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    classes = np.unique(y_true)
    if classes.size == 0:
        return 0.0
    f1s = []
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s)) if f1s else 0.0

class BinAgg:
    def __init__(self):
        self.yt, self.yp, self.score = [], [], []

    def add(self, yt: int, yp: int, score_pos: float):
        self.yt.append(int(yt))
        self.yp.append(int(yp))
        self.score.append(float(score_pos))

    def finalize(self) -> Dict[str, Any]:
        n = len(self.yt)
        out = {"n": n, "type": "bool", "acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auroc": 0.5}
        if n == 0:
            return out
        out.update(binary_prf_auroc(self.yt, self.yp, self.score))
        return out


class CatAgg:
    def __init__(self):
        self.yt, self.yp, self.logits = [], [], []

    def add(self, yt: int, logits: torch.Tensor):
        self.yt.append(int(yt))
        self.yp.append(int(torch.argmax(logits).item()))
        self.logits.append(logits.detach().cpu())

    def finalize_basic(self) -> Dict[str, Any]:
        n = len(self.yt)
        out = {"n": n, "type": "cat", "top1_acc": 0.0, "top3_acc": 0.0}
        if n == 0:
            return out
        y_true = torch.tensor(self.yt)
        y_pred = torch.tensor(self.yp)
        out["top1_acc"] = float((y_true == y_pred).float().mean().item())
        L = torch.stack(self.logits, dim=0) if self.logits else torch.empty(0)
        if L.numel() > 0:
            hits3 = topk_hit(L, y_true, k=3)
            out["top3_acc"] = float(hits3 / n)
        else:
            out["top3_acc"] = out["top1_acc"]
        return out

    def finalize_with_macro_f1(self) -> Dict[str, Any]:
        z = self.finalize_basic()
        if z["n"] == 0:
            z["macro_f1"] = 0.0
            return z
        z["macro_f1"] = macro_f1_multiclass(self.yt, self.yp)
        return z


class CountAgg:
    def __init__(self):
        self.yt, self.yp, self.logits = [], [], []

    def add(self, yt: int, logits: torch.Tensor):
        self.yt.append(int(yt))
        pred = int(torch.argmax(logits).item())
        self.yp.append(pred)
        self.logits.append(logits.detach().cpu())

    def finalize(self) -> Dict[str, Any]:
        n = len(self.yt)
        out = {"n": n, "type": "count", "top1_acc": 0.0, "within1_acc": 0.0, "macro_f1": 0.0, "top3_acc": 0.0}
        if n == 0:
            return out
        y_true = np.asarray(self.yt)
        y_pred = np.asarray(self.yp)
        out["top1_acc"] = float((y_true == y_pred).mean())
        out["within1_acc"] = float((np.abs(y_true - y_pred) <= 1).mean())
        out["macro_f1"] = macro_f1_multiclass(self.yt, self.yp)
        L = torch.stack(self.logits, dim=0) if self.logits else torch.empty(0)
        if L.numel() > 0:
            hits3 = topk_hit(L, torch.tensor(self.yt), k=3)
            out["top3_acc"] = float(hits3 / n)
        else:
            out["top3_acc"] = out["top1_acc"]
        return out

    def finalize_as_index(self) -> Dict[str, Any]:
        z = self.finalize()
        z["type"] = "index"
        return z


class SecAgg:
    def __init__(self):
        self.gt, self.pr = [], []

    def add(self, gt: float, pr: float):
        self.gt.append(float(gt))
        self.pr.append(float(pr))

    def finalize(self) -> Dict[str, Any]:
        n = len(self.gt)
        out = {
            "n": n, "type": "seconds",
            "mae": 0.0, "rmse": 0.0, "nmae_gt": 0.0,
            "within_tau": {"1.0": 0.0, "2.0": 0.0, "5.0": 0.0}
        }
        if n == 0:
            return out
        gt = np.asarray(self.gt, dtype=np.float64)
        pr = np.asarray(self.pr, dtype=np.float64)
        err = np.abs(pr - gt)
        out["mae"] = float(err.mean())
        out["rmse"] = float(np.sqrt(np.mean((pr - gt) ** 2)))
        denom = np.maximum(np.abs(gt), 1e-6)
        out["nmae_gt"] = float(np.mean(err / denom))
        for tau in (1.0, 2.0, 5.0):
            out["within_tau"][f"{tau}"] = float((err <= tau).mean())
        return out


# Helpers
def _extract_choice_ids_from_meta(m: Any) -> Optional[List[int]]:
    """从 meta 中尽力取出候选类索引列表；找不到就返回 None"""
    if isinstance(m, dict):
        v = m.get("choice_K", None) # meta {"choice_K"...}
        if v is not None:
            if isinstance(v, (list, tuple)):
                return [int(x) for x in v]
            try:
                if isinstance(v, np.ndarray):
                    return [int(x) for x in v.tolist()]
            except Exception:
                pass
    return None

def _build_mask_from_ids(ids: Optional[List[int]], num_classes: int,
                         gt_id: Optional[int] = None, device=None) -> torch.Tensor:
    # ids 缺失/空  不做掩码（全 True），避免退化为“只留真值”
    if not ids:
        return torch.ones(num_classes, dtype=torch.bool, device=device)

    # 正常：把候选标 True
    m = torch.zeros(num_classes, dtype=torch.bool, device=device)
    ids_t = torch.tensor([max(0, min(num_classes - 1, int(i))) for i in ids],
                         device=device, dtype=torch.long)
    if ids_t.numel() > 0:
        m.scatter_(0, ids_t.unique(), True)

    # 为了保险，把 GT 也并入（避免候选遗漏 GT）
    if gt_id is not None and 0 <= int(gt_id) < num_classes:
        m[int(gt_id)] = True

    # 极端防御：若仍全 False（基本不会发生），回退全 True
    if not bool(m.any()):
        m[:] = True
    return m


def _extract_categories(batch: Dict[str, Any], n: int) -> List[str]:
    cats = batch.get("categories", None)
    if cats is not None:
        return list(map(str, cats))
    metas = batch.get("meta", [None] * n)
    out = []
    for m in metas:
        if isinstance(m, dict) and ("category" in m):
            out.append(str(m["category"]))
        else:
            out.append("unknown")
    return out


def evaluate_same_schema(loader: DataLoader, model: MultiTaskSSGVQA, device) -> Dict[str, Any]:
    model.eval()

    by_category = {
        "boundary":       {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg()},
        "concurrency":    {"bool": BinAgg(), "seconds": SecAgg(), "choice": CatAgg()},
        "count":          {"count": CountAgg()},
        "duration":       {"duration": SecAgg()},
        "extreme":        {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg()},
        "motion":         {"cat": CatAgg()},
        "ordering":       {"bool": BinAgg(), "cat": CatAgg(), "choice": CatAgg()},
        "phase_transition": {"bool": BinAgg(), "cat": CatAgg(), "seconds": SecAgg(), "choice": CatAgg()},
    }

    task_hist = {k: 0 for k in [
        "count", "duration", "motion",
        "ordering_text", "ordering_bin",
        "extreme_text", "extreme_reg", "extreme_bin",
        "concurrency_bin", "concurrency_reg",
        "boundary_bin", "boundary_reg", "boundary_text",
        "phase_reg", "phase_text", "phase_bin",
        "ordering_choice", "concurrency_choice", "phase_choice",
    ]}

    with torch.no_grad():
        for batch in loader:
            text = {k: v.to(device) for k, v in batch["text"].items()}
            vis  = {k: v.to(device) for k, v in batch["visual"].items()}
            tasks: List[str] = list(batch["tasks"])
            labels = batch["labels"]
            cats = _extract_categories(batch, n=len(tasks))

            metas = batch.get("meta", [None] * len(tasks))
            

            uniq_tasks = sorted(set(tasks))
            for tname in uniq_tasks:
                idx = [i for i, t in enumerate(tasks) if t == tname]
                if not idx:
                    continue

                sub_text = {k: v[idx] for k, v in text.items()}
                sub_vis  = {k: v[idx] for k, v in vis.items()}
                sub_lab  = [labels[i] for i in idx]
                sub_cat  = [cats[i] for i in idx]
                sub_meta = [metas[i] for i in idx]

                out = model(sub_text, visual=sub_vis, task=tname, labels=None, meta=sub_meta)
                logits = out["logits"]

                if tname.endswith("_reg"):  # duration_reg
                    pred = logits.view(-1).float().cpu().numpy()
                    for j, gt in enumerate(sub_lab):
                        gt_val = float(gt.item()) if torch.is_tensor(gt) else float(gt)
                        cat = str(sub_cat[j])
                        if cat == "duration":
                            by_category["duration"]["duration"].add(gt_val, float(pred[j]))
                            task_hist["duration"] += 1
                        elif cat == "boundary":
                            by_category["boundary"]["seconds"].add(gt_val, float(pred[j]))
                            task_hist["boundary_reg"] += 1
                        elif cat == "concurrency":
                            by_category["concurrency"]["seconds"].add(gt_val, float(pred[j]))
                            task_hist["concurrency_reg"] += 1
                        elif cat in ("phase", "phase_transition"):
                            by_category["phase_transition"]["seconds"].add(gt_val, float(pred[j]))
                            task_hist["phase_reg"] += 1
                        elif cat == "extreme":
                            by_category["extreme"]["seconds"].add(gt_val, float(pred[j]))
                            task_hist["extreme_reg"] += 1
                        else:
                            pass

                elif tname == "count_cls":
                    for j, gt in enumerate(sub_lab):
                        gt_id = int(gt.item()) if torch.is_tensor(gt) else int(gt)
                        by_category["count"]["count"].add(gt_id, logits[j])
                        task_hist["count"] += 1

                elif tname.endswith("_bool"):
                    # boundary_bool / concurrency_bool / ordering_bool / phase_transition_bool / extreme_bool
                    prob_pos = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
                    y_pred = torch.argmax(logits, dim=-1).cpu().numpy()
                    for j, gt in enumerate(sub_lab):
                        gt_id = int(gt.item()) if torch.is_tensor(gt) else int(gt)
                        cname = str(sub_cat[j])
                        if "boundary" in tname or cname == "boundary":
                            by_category["boundary"]["bool"].add(gt_id, int(y_pred[j]), float(prob_pos[j]))
                            task_hist["boundary_bin"] += 1
                        elif "concurrency" in tname or cname == "concurrency":
                            by_category["concurrency"]["bool"].add(gt_id, int(y_pred[j]), float(prob_pos[j]))
                            task_hist["concurrency_bin"] += 1
                        elif "ordering" in tname or cname == "ordering":
                            by_category["ordering"]["bool"].add(gt_id, int(y_pred[j]), float(prob_pos[j]))
                            task_hist["ordering_bin"] += 1
                        elif "phase" in tname or cname in ("phase", "phase_transition"):
                            by_category["phase_transition"]["bool"].add(gt_id, int(y_pred[j]), float(prob_pos[j]))
                            task_hist["phase_bin"] += 1
                        elif "extreme" in tname or cname == "extreme":
                            by_category["extreme"]["bool"].add(gt_id, int(y_pred[j]), float(prob_pos[j]))
                            task_hist["extreme_bin"] += 1
                elif tname.endswith("_choice"):
                    # ordering_choice / concurrency_choice / phase_choice
                    # 逐样本应用候选掩码，再送入 CatAgg 统计
                    C = logits.size(1)
                    masked_logits = logits.clone()

                    for j, gt in enumerate(sub_lab):
                        gt_id = int(gt.item()) if torch.is_tensor(gt) else int(gt)
                        m_ids = _extract_choice_ids_from_meta(sub_meta[j])  # 期望 meta 里有候选；没有则 None
                        m = _build_mask_from_ids(m_ids, num_classes=C, gt_id=gt_id, device=masked_logits.device)
                        # 非候选置 -inf，使得 argmax/topk 仅在候选内
                        masked_logits[j].masked_fill_(~m, float("-inf"))

                        cname = str(sub_cat[j])
                        if "ordering" in tname or cname == "ordering":
                            by_category["ordering"]["choice"].add(gt_id, masked_logits[j])
                            task_hist["ordering_choice"] += 1
                        elif "concurrency" in tname or cname == "concurrency":
                            by_category["concurrency"]["choice"].add(gt_id, masked_logits[j])
                            task_hist["concurrency_choice"] += 1
                        elif "phase" in tname or cname in ("phase", "phase_transition"):
                            by_category["phase_transition"]["choice"].add(gt_id, masked_logits[j])
                            task_hist["phase_choice"] += 1
                        else:
                            # 兜底：如果出现未知类别，就按 ordering 记（也可直接 continue）
                            by_category["ordering"]["choice"].add(gt_id, masked_logits[j])
                            task_hist["ordering_choice"] += 1



                elif tname.endswith("_text") or (tname.endswith("_cls") and tname != "count_cls"):
                    # boundary_text / ordering_text / phase_text / extreme_text / motion_cls(或 motion_text)
                    for j, gt in enumerate(sub_lab):
                        gt_id = int(gt.item()) if torch.is_tensor(gt) else int(gt)
                        cname = str(sub_cat[j])
                        if "boundary" in tname or cname == "boundary":
                            by_category["boundary"]["cat"].add(gt_id, logits[j])
                            task_hist["boundary_text"] += 1
                        elif "ordering" in tname or cname == "ordering":
                            by_category["ordering"]["cat"].add(gt_id, logits[j])
                            # task_hist["ordering_text"] += 1
                            if tname.endswith("_choice"):
                                task_hist["ordering_choice"] += 1
                            else:
                                task_hist["ordering_text"] += 1

                        elif "phase" in tname or cname in ("phase", "phase_transition"):
                            by_category["phase_transition"]["cat"].add(gt_id, logits[j])
                            # task_hist["phase_text"] += 1
                            if tname.endswith("_choice"):
                                task_hist["phase_choice"] += 1
                            else:
                                task_hist["phase_text"] += 1


                        elif "extreme" in tname or cname == "extreme":
                            by_category["extreme"]["cat"].add(gt_id, logits[j])
                            task_hist["extreme_text"] += 1
                        elif "motion" in tname or cname == "motion":
                            by_category["motion"]["cat"].add(gt_id, logits[j])
                            task_hist["motion"] += 1
                        elif "concurrency" in tname or cname == "concurrency":
                            by_category["concurrency"]["cat"].add(gt_id, logits[j])
                            if tname.endswith("_choice"):
                                task_hist["concurrency_choice"] += 1
                            # concurrency 并不存在 *_text，故无需 else 分支

                        else:
                            pass

                else:
                    pass

    def pick_bool(d: Dict[str, Any]):   return d["f1"] if d["n"] > 0 else None
    def pick_cat(d: Dict[str, Any]):    return d["top1_acc"] if d["n"] > 0 else None
    def pick_count(d: Dict[str, Any]):  return d["within1_acc"] if d["n"] > 0 else None
    def pick_secs(d: Dict[str, Any]):   return d["within_tau"]["5.0"] if d["n"] > 0 else None

    # finalize 各 cell
    bd_bool  = by_category["boundary"]["bool"].finalize()
    bd_cat   = by_category["boundary"]["cat"].finalize_with_macro_f1()
    bd_sec   = by_category["boundary"]["seconds"].finalize()

    cc_bool  = by_category["concurrency"]["bool"].finalize()
    cc_sec   = by_category["concurrency"]["seconds"].finalize()
    # cc_cat   = by_category["concurrency"]["cat"].finalize_basic()
    cc_choice= by_category["concurrency"]["choice"].finalize_basic()

    ct       = by_category["count"]["count"].finalize()

    dur      = by_category["duration"]["duration"].finalize()
    dur["type"] = "duration"  

    ex_bool  = by_category["extreme"]["bool"].finalize()
    ex_cat   = by_category["extreme"]["cat"].finalize_basic()
    ex_sec   = by_category["extreme"]["seconds"].finalize()

    mo_cat   = by_category["motion"]["cat"].finalize_with_macro_f1()

    od_bool  = by_category["ordering"]["bool"].finalize()
    od_cat   = by_category["ordering"]["cat"].finalize_basic()
    od_choice= by_category["ordering"]["choice"].finalize_basic()

    ph_bool  = by_category["phase_transition"]["bool"].finalize()
    ph_cat   = by_category["phase_transition"]["cat"].finalize_basic()
    ph_sec   = by_category["phase_transition"]["seconds"].finalize()
    ph_choice= by_category["phase_transition"]["choice"].finalize_basic()

    result = {
        "task_hist": {k: int(v) for k, v in task_hist.items()},
        "by_category": {
            "boundary":        {"bool": bd_bool, "cat": bd_cat, "seconds": bd_sec},
            "concurrency":     {"bool": cc_bool, "seconds": cc_sec, "choice": cc_choice},
            "count":           {"count": ct},
            "duration":        {"duration": dur},
            "extreme":         {"bool": ex_bool, "cat": ex_cat, "seconds": ex_sec},
            "motion":          {"cat": mo_cat},
            "ordering":        {"bool": od_bool, "cat": od_cat, "choice": od_choice},
            "phase_transition": {"bool": ph_bool, "cat": ph_cat, "seconds": ph_sec, "choice": ph_choice},
        }
    }

    cell_scores: List[float] = []
    for sc in [
        # boundary
        pick_bool(bd_bool), pick_cat(bd_cat), pick_secs(bd_sec),
        # concurrency
        pick_bool(cc_bool), pick_secs(cc_sec), pick_cat(cc_choice),
        # count
        pick_count(ct),
        # duration
        pick_secs(dur),
        # extreme
        pick_bool(ex_bool), pick_cat(ex_cat), pick_secs(ex_sec),
        # motion
        pick_cat(mo_cat),
        # ordering
        pick_bool(od_bool), pick_cat(od_cat), pick_cat(od_choice),
        # phase_transition
        pick_bool(ph_bool), pick_cat(ph_cat), pick_secs(ph_sec), pick_cat(ph_choice),
    ]:
        if sc is not None:
            cell_scores.append(sc)

    result["overall_macro_score"] = float(np.mean(cell_scores)) if cell_scores else 0.0
    return result

def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    logger = build_logger("eval_temporal",
                          log_file=os.path.join(args.checkpoint_dir, "eval_temporal.log"))

    print(f"[Eval] use_rationale={args.use_rationale} | tokenizer={args.tokenizer_ver} | text_max_len={args.text_max_len}")

    # Tokenizer
    tokenizer = select_tokenizer(args.tokenizer_ver)

    # Fixed vocabs
    fixed_vocabs = None
    if args.vocabs_json and os.path.isfile(args.vocabs_json):
        with open(args.vocabs_json, "r", encoding="utf-8") as f:
            fixed_vocabs = json.load(f)
        print("[Load] training vocabs:", {k: len(v) for k, v in fixed_vocabs.items() if isinstance(v, list)})

    # Dataset + Collate
    ds = STSGTemporalQADataset(
        qa_root=args.stsg_qa_root,
        folder_head=args.feature_root,
        vids=args.temporal_videos,
        allowed_categories=[
            "count", "duration", "ordering", "extreme",
            "boundary", "motion", "concurrency", "phase_transition"
        ],
        max_frames=args.max_frames,
        fps=args.fps,
        logger=logger,
        fixed_vocabs=fixed_vocabs,
    )

    collate = make_collate_fn(
        tokenizer,
        use_rationale=args.use_rationale,
        rationale_mode="append",
        rationale_tag="Rationale:",
        rationale_dropout_p=0.0,
        max_length=args.text_max_len,
    )

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collate
    )

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_ckpt(
        args.checkpoint,
        tokenizer_vocab_size=getattr(tokenizer, "vocab_size", len(tokenizer)),
        max_video_frames=args.max_frames,
        default_count_bins=args.count_bins
    ).to(device)

    # Evaluate
    results = evaluate_same_schema(loader, model, device)

    # Save & print
    suffix = "with_rationale" if args.use_rationale else "no_rationale"
    out_path = Path(args.checkpoint_dir) / f"temporal_eval_{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    logger.info(f"[Done] Saved to {out_path}")


if __name__ == "__main__":
    main()
