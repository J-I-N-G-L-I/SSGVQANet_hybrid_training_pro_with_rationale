#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Re-designed evaluation for LLaVA-Med on STSG temporal QA.

This version:
- Uses vocabs_train.json to cast text answers into closed-set classes.
- Parses answers from the substring after 'FINAL_ANSWER:'.
- For 'seconds' style tasks, supports floating-point numbers.
- Metrics follow the same scheme as test_hybrid.py for comparability.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------
# Basic utilities
# ----------------------


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b > 0 else 0.0


def macro_f1_multiclass(y_true: List[int], y_pred: List[int]) -> float:
    """Macro-F1 over classes present in ground truth."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    classes = np.unique(y_true)
    if classes.size == 0:
        return 0.0

    f1s: List[float] = []
    for c in classes:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = safe_div(2 * prec * rec, (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)

    return float(np.mean(f1s)) if f1s else 0.0


def binary_prf_auroc(
    y_true: List[int], y_pred: List[int], y_score_pos: List[float]
) -> Dict[str, float]:
    """
    Compute acc / precision / recall / f1 / auroc for binary with positive=1.

    Here y_score_pos can be "hard" scores (1.0 for pred=1, 0.0 for pred=0),
    AUROC is still a meaningful approximation.
    """
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

    # degenerate: only one class present
    if y_true.min() == y_true.max():
        auroc = 0.5
    else:
        # manual ROC
        order = np.argsort(-y_score)
        y_sorted = y_true[order]
        score_sorted = y_score[order]

        P = float((y_true == 1).sum())
        N = float((y_true == 0).sum())

        tp_cum = 0.0
        fp_cum = 0.0
        prev = None
        roc: List[Tuple[float, float]] = [(0.0, 0.0)]

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


# ----------------------
# Aggregators (no logits version)
# ----------------------


class BinAgg:
    def __init__(self):
        self.yt: List[int] = []
        self.yp: List[int] = []
        self.score: List[float] = []

    def add(self, yt: int, yp: int, score_pos: float):
        self.yt.append(int(yt))
        self.yp.append(int(yp))
        self.score.append(float(score_pos))

    def finalize(self) -> Dict[str, Any]:
        n = len(self.yt)
        out: Dict[str, Any] = {
            "n": n,
            "type": "bool",
            "acc": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "auroc": 0.5,
        }
        if n == 0:
            return out
        out.update(binary_prf_auroc(self.yt, self.yp, self.score))
        return out


class CatAgg:
    """
    For generative models: we only have single predicted class (no logits).
    We still report top1_acc and macro_f1. top3_acc is set equal to top1_acc.
    """

    def __init__(self):
        self.yt: List[int] = []
        self.yp: List[int] = []

    def add(self, yt: int, yp: int):
        self.yt.append(int(yt))
        self.yp.append(int(yp))

    def finalize_basic(self) -> Dict[str, Any]:
        n = len(self.yt)
        out: Dict[str, Any] = {"n": n, "type": "cat", "top1_acc": 0.0, "top3_acc": 0.0}
        if n == 0:
            return out
        y_true = np.asarray(self.yt, dtype=np.int64)
        y_pred = np.asarray(self.yp, dtype=np.int64)
        top1 = float((y_true == y_pred).mean())
        out["top1_acc"] = top1
        # no logits / ranking, fallback: top3 == top1
        out["top3_acc"] = top1
        return out

    def finalize_with_macro_f1(self) -> Dict[str, Any]:
        z = self.finalize_basic()
        if z["n"] == 0:
            z["macro_f1"] = 0.0
            return z
        z["macro_f1"] = macro_f1_multiclass(self.yt, self.yp)
        return z


class CountAgg:
    """
    For count questions: directly use integer counts (no logits).
    """

    def __init__(self):
        self.yt: List[int] = []
        self.yp: List[int] = []

    def add(self, yt: int, yp: int):
        self.yt.append(int(yt))
        self.yp.append(int(yp))

    def finalize(self) -> Dict[str, Any]:
        n = len(self.yt)
        out: Dict[str, Any] = {
            "n": n,
            "type": "count",
            "top1_acc": 0.0,
            "within1_acc": 0.0,
            "macro_f1": 0.0,
            "top3_acc": 0.0,  # no logits, set equal to top1_acc later
        }
        if n == 0:
            return out
        y_true = np.asarray(self.yt, dtype=np.int64)
        y_pred = np.asarray(self.yp, dtype=np.int64)
        out["top1_acc"] = float((y_true == y_pred).mean())
        out["within1_acc"] = float((np.abs(y_true - y_pred) <= 1).mean())
        out["macro_f1"] = macro_f1_multiclass(self.yt, self.yp)
        # no logits, so top3_acc == top1_acc
        out["top3_acc"] = out["top1_acc"]
        return out


class SecAgg:
    """
    For 'seconds' style regression metrics: identical to test_hybrid.py.
    """

    def __init__(self):
        self.gt: List[float] = []
        self.pr: List[float] = []

    def add(self, gt: float, pr: float):
        self.gt.append(float(gt))
        self.pr.append(float(pr))

    def finalize(self) -> Dict[str, Any]:
        n = len(self.gt)
        out: Dict[str, Any] = {
            "n": n,
            "type": "seconds",
            "mae": 0.0,
            "rmse": 0.0,
            "nmae_gt": 0.0,
            "within_tau": {"1.0": 0.0, "2.0": 0.0, "5.0": 0.0},
        }
        if n == 0:
            return out
        gt = np.asarray(self.gt, dtype=np.float64)
        pr = np.asarray(self.pr, dtype=np.float64)
        err = np.abs(pr - gt)
        out["mae"] = float(err.mean())
        out["rmse"] = float(np.sqrt(np.mean((pr - gt) ** 2)))
        # keep same nmae definition as test_hybrid, although it's unstable for near-zero gt
        denom = np.maximum(np.abs(gt), 1e-6)
        out["nmae_gt"] = float(np.mean(err / denom))
        for tau in (1.0, 2.0, 5.0):
            out["within_tau"][f"{tau}"] = float((err <= tau).mean())
        return out


# ----------------------
# Vocab-based mapping for text tasks
# ----------------------


_INT_RE = re.compile(r"-?\d+")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def extract_first_int(text: Any) -> Optional[int]:
    if not isinstance(text, str):
        text = str(text)
    m = _INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0))
    except Exception:
        return None


def extract_first_number(text: Any) -> Optional[float]:
    """
    Extract the first integer or floating point number (for 'seconds' tasks).
    """
    if not isinstance(text, str):
        text = str(text)
    m = _NUM_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER\s*:\s*(.*)", re.IGNORECASE | re.DOTALL)


def extract_final_answer(text: Any) -> str:
    """
    Extract the substring after 'FINAL_ANSWER:' and strip trailing clutter.

    If the pattern is not found, return the stripped original text.
    Only the first line after FINAL_ANSWER is kept.
    """
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()
    m = _FINAL_ANSWER_RE.search(s)
    if not m:
        return s
    ans = m.group(1).strip()
    # cut at first newline / carriage return to avoid picking up anything else
    for sep in ("\n", "\r"):
        if sep in ans:
            ans = ans.split(sep, 1)[0].strip()
    return ans


# def simple_stem(token: str) -> str:
#     """
#     Very small stemmer to make verbs like 'grasps'/'grasping' -> 'grasp'.
#     This is intentionally conservative.
#     """
#     t = token
#     for suf in ("ing", "ed", "es", "s"):
#         if t.endswith(suf) and len(t) > len(suf) + 2:
#             t = t[: -len(suf)]
#             break
#     return t

def simple_stem(token: str) -> str:
    """
    极简但稍微“聪明”一点的 stemmer，专门为当前外科动词 / 名词做的规则。
    目标只是把诸如 'retraction' / 'retracting' / 'retracted' 都尽量
    拉到同一个词干上，避免 GT 和预测因为轻微形态差异对不上。

    注意：这里故意不做太激进的规则，只覆盖我们任务里常见的一些词形。
    """
    t = token

    # 一些常见的不规则/名词化形式，直接手工映射
    irregular = {
        "retraction": "retract",
        "retracting": "retract",
        "retracted": "retract",
        "dissection": "dissect",
        "dissecting": "dissect",
        "dissected": "dissect",
        "coagulation": "coagulate",
        "coagulating": "coagulate",
        "coagulated": "coagulate",
        "extraction": "extract",
        "extracting": "extract",
        "extracted": "extract",
        "grasping": "grasp",
        "clipping": "clip",
    }
    if t in irregular:
        return irregular[t]

    # 先处理一些常见的名词化后缀（-tion, -ation ...）
    for suf in ("ization", "isation", "ation", "tion", "sion"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break

    # 再处理一般动词后缀
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break

    return t


# def normalize_for_vocab_text(s: Any) -> str:
#     """
#     Normalization used for mapping both GT labels and LLaVA-Med predictions
#     into the same canonical space, *before* looking them up in vocabs_train.

#     Heuristics:
#       - lowercase
#       - keep only [a-z0-9_ ] (punctuation -> space)
#       - map frequent multi-word phrases (gall bladder -> gallbladder, etc.)
#       - drop simple stopwords (the/a/an/of/on/in/for/to/at)
#       - simple stemming on tokens
#     """
#     if not isinstance(s, str):
#         s = str(s)
#     s = s.strip().lower()

#     # unify some known multi-word concepts first
#     replacements = {
#         "gall bladder": "gallbladder",
#         "gall-bladder": "gallbladder",
#         "cystic duct": "cystic_duct",
#         "cystic artery": "cystic_artery",
#         "cystic pedicle": "cystic_pedicle",
#         "cystic plate": "cystic_plate",
#         "specimen bag": "specimen_bag",
#         "abdominal wall cavity": "abdominal_wall_cavity",
#     }
#     for k, v in replacements.items():
#         s = s.replace(k, v)

#     # replace underscores with spaces so that both 'specimen_bag'
#     # and 'specimen bag' are treated consistently
#     s = s.replace("_", " ")

#     # keep only alnum + space
#     s = re.sub(r"[^a-z0-9\s]", " ", s)

#     # split & clean tokens
#     tokens = re.split(r"\s+", s)
#     stopwords = {"", "the", "a", "an", "of", "on", "in", "to", "at", "for"}
#     processed: List[str] = []
#     for t in tokens:
#         if not t or t in stopwords:
#             continue
#         processed.append(simple_stem(t))

#     return " ".join(processed)

def normalize_for_vocab_text(s: Any) -> str:
    """
    用于把 GT label 和 LLaVA 预测统一到一个“词袋空间”里，再去查 vocabs。

    处理步骤：
      - 转小写
      - 一些已知复合名词的规范化（gall bladder -> gallbladder）
      - 把下划线也当成空格处理（specimen_bag == specimen bag）
      - 去掉标点，只保留 [a-z0-9 和空格]
      - 去掉常见功能词（the/is/was/...）
      - 对剩余 token 做一个非常轻量的 simple_stem
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()

    # 先做一些 domain-specific 替换
    replacements = {
        "gall bladder": "gallbladder",
        "gall-bladder": "gallbladder",
        "cystic duct": "cystic_duct",
        "cystic artery": "cystic_artery",
        "cystic pedicle": "cystic_pedicle",
        "cystic plate": "cystic_plate",
        "specimen bag": "specimen_bag",
        "abdominal wall cavity": "abdominal_wall_cavity",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # 下划线统一处理成空格，这样 "specimen_bag" 和 "specimen bag" 一样
    s = s.replace("_", " ")

    # 只保留字母数字和空格
    s = re.sub(r"[^a-z0-9\s]", " ", s)

    # 分词 + 去停用词 + 轻量 stem
    tokens = re.split(r"\s+", s)

    stopwords = {
        "", "the", "a", "an", "of", "on", "in", "to", "at", "for",
        "is", "was", "are", "were", "be", "been", "being",
        "by", "from", "with", "without",
        "this", "that", "these", "those",
    }

    processed: List[str] = []
    for t in tokens:
        if not t or t in stopwords:
            continue
        processed.append(simple_stem(t))

    return " ".join(processed)


class LabelEncoder:
    """
    Simple string -> int encoder (used only as a *fallback* when a category
    does not have a fixed vocab in vocabs_train.json).
    """

    def __init__(self):
        self.str2id: Dict[str, int] = {}

    def encode(self, s: str) -> int:
        if s not in self.str2id:
            self.str2id[s] = len(self.str2id)
        return self.str2id[s]


class VocabMapper:
    """
    Use vocabs_train.json to map textual labels to discrete class indices.

    Expected JSON structure (keys are category names, values are lists of
    label strings; index 0 must be '<UNK>'):

      {
        "extreme": [...],
        "phase_transition": [...],
        "boundary": [...],
        "ordering": [...],
        "motion": [...],
        ...
      }

    For GT labels we only apply direct lookup (after normalization).
    For model predictions we allow a fuzzy mapping based on token overlap.
    """

    def __init__(self, vocabs_path: Path):
        with vocabs_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocabs: Dict[str, List[str]] = {}
        self.norm2id: Dict[str, Dict[str, int]] = {}
        self.id2tokens: Dict[str, Dict[int, set]] = {}

        known_keys = [
            "boundary",
            "extreme",
            "motion",
            "ordering",
            "phase_transition",
        ]

        for key in known_keys:
            if key not in data:
                continue
            labels = data[key]
            if not isinstance(labels, list):
                raise ValueError(f"vocabs[{key}] must be a list.")
            self.vocabs[key] = labels

            norm_map: Dict[str, int] = {}
            tok_map: Dict[int, set] = {}

            for idx, lab in enumerate(labels):
                norm = normalize_for_vocab_text(lab)
                if norm:
                    norm_map.setdefault(norm, idx)
                tokens = set(norm.split()) if norm else set()
                tok_map[idx] = tokens

            self.norm2id[key] = norm_map
            self.id2tokens[key] = tok_map

        # fallback encoders for any category not covered above
        self.fallback_encoders: Dict[str, LabelEncoder] = {}

    @staticmethod
    def _canon_category_name(category: str) -> str:
        """
        Map metadata["category"] into a key for vocabs.

        - 'phase' and 'phase_transition' share 'phase_transition' vocab.
        - others (boundary/extreme/motion/ordering) map to themselves.
        """
        c = category.lower()
        if c in ("phase", "phase_transition"):
            return "phase_transition"
        return c

    def _get_fallback_encoder(self, key: str) -> LabelEncoder:
        if key not in self.fallback_encoders:
            self.fallback_encoders[key] = LabelEncoder()
        return self.fallback_encoders[key]

    def encode_gt(self, category: str, label: Any) -> int:
        """
        Encode GT label into an integer.

        For known vocab categories: strict lookup after normalization.
        For others: use a per-category fallback LabelEncoder.
        """
        key = self._canon_category_name(category)
        norm = normalize_for_vocab_text(label)

        if key in self.vocabs:
            id_map = self.norm2id[key]
            if norm in id_map:
                return id_map[norm]
            # GT 理论上都在 vocab 里；没找到就当 <UNK>=0
            return 0

        enc = self._get_fallback_encoder(key)
        return enc.encode(norm)

    # def encode_pred(self, category: str, label: Any) -> int:
    #     """
    #     Encode model prediction.

    #     For known vocab categories, we first try exact normalized match,
    #     then fuzzy token-overlap mapping. If everything fails, we map to UNK(0).

    #     For unknown categories we use a fallback LabelEncoder so that at
    #     least exact string matches count as correct.
    #     """
    #     key = self._canon_category_name(category)
    #     norm = normalize_for_vocab_text(label)

    #     if key in self.vocabs:
    #         id_map = self.norm2id[key]
    #         tok_map = self.id2tokens[key]

    #         # 1) direct normalized lookup
    #         if norm in id_map:
    #             return id_map[norm]

    #         # 2) fuzzy mapping: Jaccard overlap on token sets
    #         tokens_pred = set(norm.split())
    #         if not tokens_pred:
    #             return 0

    #         best_idx = 0  # default UNK
    #         best_score = 0.0

    #         for idx, tokens_lab in tok_map.items():
    #             if idx == 0:  # skip UNK
    #                 continue
    #             if not tokens_lab:
    #                 continue
    #             inter = len(tokens_pred & tokens_lab)
    #             if inter == 0:
    #                 continue
    #             union = len(tokens_pred | tokens_lab)
    #             score = inter / float(union)
    #             if score > best_score:
    #                 best_score = score
    #                 best_idx = idx

    #         if best_score >= 0.5:
    #             return best_idx
    #         return 0

    #     enc = self._get_fallback_encoder(key)
    #     return enc.encode(norm)
    def encode_pred(self, category: str, label: Any) -> int:
        """
        Encode model prediction.

        对于有固定 vocab 的类别：
          1. 先做规范化字符串的 exact match；
          2. 否则根据 token 集合做 fuzzy 匹配：
             - 优先选择「label token 全部被覆盖」的类别（coverage == 1）；
             - 如果没有 full coverage，再看 Jaccard 得分；
             - 实在匹配不上则映射到 UNK(0)。

        对于没有固定 vocab 的类别，退回到每类一个 LabelEncoder。
        """
        key = self._canon_category_name(category)
        norm = normalize_for_vocab_text(label)

        if key in self.vocabs:
            id_map = self.norm2id[key]
            tok_map = self.id2tokens[key]

            # 1) 先试规范化后的 exact match
            if norm in id_map:
                return id_map[norm]

            # 2) fuzzy 匹配：先看 label token 是否被“完全覆盖”
            tokens_pred = set(norm.split())
            if not tokens_pred:
                return 0  # 预测里什么内容词都没有，直接判 UNK

            best_full_idx = 0
            best_full_cover = 0.0

            best_jacc_idx = 0
            best_jacc = 0.0

            for idx, tokens_lab in tok_map.items():
                if idx == 0:  # 跳过 UNK
                    continue
                if not tokens_lab:
                    continue

                inter = len(tokens_pred & tokens_lab)
                if inter == 0:
                    continue

                # label token 的覆盖率：交集 / label token 数
                cover = inter / float(len(tokens_lab))
                # 普通 Jaccard：交集 / 并集
                union = len(tokens_pred | tokens_lab)
                jacc = inter / float(union)

                # 记录 “覆盖率” 最好的那个 label
                if cover > best_full_cover:
                    best_full_cover = cover
                    best_full_idx = idx

                # 同时记录 Jaccard 最好的那个 label（兜底用）
                if jacc > best_jacc:
                    best_jacc = jacc
                    best_jacc_idx = idx

            # 2a) 如果存在一个 label 的所有 token 都出现在预测中，
            #     直接把它当作预测结果（extra token 不算错误）
            if best_full_cover >= 1.0:
                return best_full_idx

            # 2b) 否则退回到 Jaccard；阈值可以稍微宽松一点
            #     对于 ordering 这种 phrase 很长的类别，Jaccard 正常也会偏低。
            min_jacc = 0.5
            if key == "ordering":
                # ordering 的 vocab 往往比较短，而预测 answer 是完整的事件短语，
                # Jaccard 会被额外 token 稀释，适当放宽一点阈值。
                min_jacc = 0.4

            if best_jacc >= min_jacc:
                return best_jacc_idx

            # 2c) 实在匹配不上就判成 UNK
            return 0

        # 没有固定 vocab 的类别：用 fallback encoder，这样至少 exact string match 能算对
        enc = self._get_fallback_encoder(key)
        return enc.encode(norm)

# ----------------------
# Parsing helpers for non-text label_types
# ----------------------


def parse_bool_label(x: Any) -> Optional[int]:
    """
    Parse GT boolean label from metadata["answer"].

    Returns 1 for True/yes, 0 for False/no, or None on failure.
    """
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, np.integer)):
        return 1 if int(x) != 0 else 0
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"yes", "y", "true", "1"}:
            return 1
        if s in {"no", "n", "false", "0"}:
            return 0
    return None


def parse_bool_text(s: Any) -> Optional[int]:
    """
    Parse bool from model's free-form answer text.
    We primarily look at the first token.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    if not s:
        return None

    first = s.split()[0]
    if first.startswith("yes"):
        return 1
    if first.startswith("no"):
        return 0

    if "yes" in s and "no" not in s:
        return 1
    if "no" in s and "yes" not in s:
        return 0
    return None


def parse_numeric_gt(x: Any) -> Optional[float]:
    """
    Parse GT numeric (count or seconds) from metadata["answer"].
    """
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    if isinstance(x, str):
        x = x.strip()
        try:
            return float(x)
        except Exception:
            pass
        v = extract_first_int(x)
        if v is not None:
            return float(v)

    return None


# ----------------------
# Evaluation loop
# ----------------------


def evaluate_llavamed(answers_path: Path, vocabs_path: Path) -> Dict[str, Any]:
    """
    Main evaluation entry point.

    answers_path: answers_llavamed_*.jsonl produced by run_stsg_vqa.sh
    vocabs_path:  vocabs_train.json used by STSG-Net (for text tasks)
    """
    vocab_mapper = VocabMapper(vocabs_path)

    # per-category aggregators, following the structure of test_hybrid.py
    by_category: Dict[str, Dict[str, Any]] = {
        "boundary": {
            "bool": BinAgg(),
            "cat": CatAgg(),
            "seconds": SecAgg(),
        },
        "concurrency": {
            "bool": BinAgg(),
            "seconds": SecAgg(),
            "choice": CatAgg(),
        },
        "count": {
            "count": CountAgg(),
        },
        "duration": {
            "duration": SecAgg(),
        },
        "extreme": {
            "bool": BinAgg(),
            "cat": CatAgg(),
            "seconds": SecAgg(),
        },
        "motion": {
            "cat": CatAgg(),
        },
        "ordering": {
            "bool": BinAgg(),
            "cat": CatAgg(),
            "choice": CatAgg(),
        },
        "phase_transition": {
            "bool": BinAgg(),
            "cat": CatAgg(),
            "seconds": SecAgg(),
            "choice": CatAgg(),
        },
    }

    # text label-types that should use vocabs_train.json
    vocab_text_categories = {
        "boundary",
        "extreme",
        "motion",
        "ordering",
        "phase",
        "phase_transition",
    }

    # task histogram: keep structure identical to your STSG-Net eval
    task_hist: Dict[str, int] = {
        k: 0
        for k in [
            "count",
            "duration",
            "motion",
            "ordering_text",
            "ordering_bin",
            "extreme_text",
            "extreme_reg",
            "extreme_bin",
            "concurrency_bin",
            "concurrency_reg",
            "boundary_bin",
            "boundary_reg",
            "boundary_text",
            "phase_reg",
            "phase_text",
            "phase_bin",
            "ordering_choice",
            "concurrency_choice",
            "phase_choice",
        ]
    }

    # 逐行读取 answers JSONL
    with answers_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            meta = rec.get("metadata", {}) or {}
            category = str(meta.get("category", "unknown"))
            label_type = str(meta.get("label_type", "text"))
            task = str(meta.get("task", ""))  # used only for task_hist
            gt_answer = meta.get("answer", None)
            if gt_answer is None:
                continue

            raw_text = rec.get("text", "")
            pred_text = extract_final_answer(raw_text)

            # ---- COUNT ----
            if label_type == "count" or task == "count":
                gt_val = parse_numeric_gt(gt_answer)
                if gt_val is None:
                    continue
                pred_val = extract_first_int(pred_text)
                if pred_val is None:
                    pred_val = 0
                by_category["count"]["count"].add(int(gt_val), int(pred_val))
                task_hist["count"] += 1
                continue

            # ---- SECONDS ----
            if label_type == "seconds":
                gt_val = parse_numeric_gt(gt_answer)
                if gt_val is None:
                    continue
                pred_num = extract_first_number(pred_text)
                pred_val = float(pred_num) if pred_num is not None else 0.0

                if category == "duration":
                    by_category["duration"]["duration"].add(gt_val, pred_val)
                    task_hist["duration"] += 1
                elif category == "boundary":
                    by_category["boundary"]["seconds"].add(gt_val, pred_val)
                    task_hist["boundary_reg"] += 1
                elif category == "concurrency":
                    by_category["concurrency"]["seconds"].add(gt_val, pred_val)
                    task_hist["concurrency_reg"] += 1
                elif category in ("phase", "phase_transition"):
                    by_category["phase_transition"]["seconds"].add(gt_val, pred_val)
                    task_hist["phase_reg"] += 1
                elif category == "extreme":
                    by_category["extreme"]["seconds"].add(gt_val, pred_val)
                    task_hist["extreme_reg"] += 1
                else:
                    pass
                continue

            # ---- BOOL ----
            if label_type == "bool":
                gt_bool = parse_bool_label(gt_answer)
                if gt_bool is None:
                    continue
                pred_bool = parse_bool_text(pred_text)
                if pred_bool is None:
                    pred_bool = 0
                score_pos = float(pred_bool)

                if category == "boundary":
                    by_category["boundary"]["bool"].add(gt_bool, pred_bool, score_pos)
                    task_hist["boundary_bin"] += 1
                elif category == "concurrency":
                    by_category["concurrency"]["bool"].add(gt_bool, pred_bool, score_pos)
                    task_hist["concurrency_bin"] += 1
                elif category == "ordering":
                    by_category["ordering"]["bool"].add(gt_bool, pred_bool, score_pos)
                    task_hist["ordering_bin"] += 1
                elif category in ("phase", "phase_transition"):
                    by_category["phase_transition"]["bool"].add(
                        gt_bool, pred_bool, score_pos
                    )
                    task_hist["phase_bin"] += 1
                elif category == "extreme":
                    by_category["extreme"]["bool"].add(gt_bool, pred_bool, score_pos)
                    task_hist["extreme_bin"] += 1
                else:
                    pass
                continue

            # ---- CHOICE ----
            if label_type == "choice_index":
                gt_val = parse_numeric_gt(gt_answer)
                if gt_val is None:
                    continue
                gt_idx = int(gt_val)
                pred_idx = extract_first_int(pred_text)
                if pred_idx is None:
                    pred_idx = 0

                if category == "ordering":
                    by_category["ordering"]["choice"].add(gt_idx, int(pred_idx))
                    task_hist["ordering_choice"] += 1
                elif category == "concurrency":
                    by_category["concurrency"]["choice"].add(gt_idx, int(pred_idx))
                    task_hist["concurrency_choice"] += 1
                elif category in ("phase", "phase_transition"):
                    by_category["phase_transition"]["choice"].add(
                        gt_idx, int(pred_idx)
                    )
                    task_hist["phase_choice"] += 1
                else:
                    pass
                continue

            # ---- TEXT ----
            gt_label_raw = str(gt_answer)
            pred_label_raw = pred_text

            if category in vocab_text_categories:
                gt_id = vocab_mapper.encode_gt(category, gt_label_raw)
                pred_id = vocab_mapper.encode_pred(category, pred_label_raw)
            else:
                norm_gt = normalize_for_vocab_text(gt_label_raw)
                norm_pred = normalize_for_vocab_text(pred_label_raw)
                enc = vocab_mapper._get_fallback_encoder(category)
                gt_id = enc.encode(norm_gt)
                pred_id = enc.encode(norm_pred)

            if category == "boundary":
                by_category["boundary"]["cat"].add(gt_id, pred_id)
                task_hist["boundary_text"] += 1
            elif category == "ordering":
                by_category["ordering"]["cat"].add(gt_id, pred_id)
                task_hist["ordering_text"] += 1
            elif category in ("phase", "phase_transition"):
                by_category["phase_transition"]["cat"].add(gt_id, pred_id)
                task_hist["phase_text"] += 1
            elif category == "extreme":
                by_category["extreme"]["cat"].add(gt_id, pred_id)
                task_hist["extreme_text"] += 1
            elif category == "motion":
                by_category["motion"]["cat"].add(gt_id, pred_id)
                task_hist["motion"] += 1
            else:
                pass

    # ---------- finalize, same pattern as test_hybrid.py ----------

    def pick_bool(d: Dict[str, Any]) -> Optional[float]:
        return d["f1"] if d.get("n", 0) > 0 else None

    def pick_cat(d: Dict[str, Any]) -> Optional[float]:
        return d["top1_acc"] if d.get("n", 0) > 0 else None

    def pick_count(d: Dict[str, Any]) -> Optional[float]:
        return d["within1_acc"] if d.get("n", 0) > 0 else None

    def pick_secs(d: Dict[str, Any]) -> Optional[float]:
        return d["within_tau"]["5.0"] if d.get("n", 0) > 0 else None

    # boundary
    bd_bool = by_category["boundary"]["bool"].finalize()
    bd_cat = by_category["boundary"]["cat"].finalize_with_macro_f1()
    bd_sec = by_category["boundary"]["seconds"].finalize()

    # concurrency
    cc_bool = by_category["concurrency"]["bool"].finalize()
    cc_sec = by_category["concurrency"]["seconds"].finalize()
    cc_choice = by_category["concurrency"]["choice"].finalize_basic()

    # count
    ct = by_category["count"]["count"].finalize()

    # duration
    dur = by_category["duration"]["duration"].finalize()
    dur["type"] = "duration"

    # extreme
    ex_bool = by_category["extreme"]["bool"].finalize()
    ex_cat = by_category["extreme"]["cat"].finalize_basic()
    ex_sec = by_category["extreme"]["seconds"].finalize()

    # motion
    mo_cat = by_category["motion"]["cat"].finalize_with_macro_f1()

    # ordering
    od_bool = by_category["ordering"]["bool"].finalize()
    od_cat = by_category["ordering"]["cat"].finalize_basic()
    od_choice = by_category["ordering"]["choice"].finalize_basic()

    # phase_transition
    ph_bool = by_category["phase_transition"]["bool"].finalize()
    ph_cat = by_category["phase_transition"]["cat"].finalize_basic()
    ph_sec = by_category["phase_transition"]["seconds"].finalize()
    ph_choice = by_category["phase_transition"]["choice"].finalize_basic()

    result: Dict[str, Any] = {
        "task_hist": {k: int(v) for k, v in task_hist.items()},
        "by_category": {
            "boundary": {
                "bool": bd_bool,
                "cat": bd_cat,
                "seconds": bd_sec,
            },
            "concurrency": {
                "bool": cc_bool,
                "seconds": cc_sec,
                "choice": cc_choice,
            },
            "count": {
                "count": ct,
            },
            "duration": {
                "duration": dur,
            },
            "extreme": {
                "bool": ex_bool,
                "cat": ex_cat,
                "seconds": ex_sec,
            },
            "motion": {
                "cat": mo_cat,
            },
            "ordering": {
                "bool": od_bool,
                "cat": od_cat,
                "choice": od_choice,
            },
            "phase_transition": {
                "bool": ph_bool,
                "cat": ph_cat,
                "seconds": ph_sec,
                "choice": ph_choice,
            },
        },
    }

    # overall_macro_score: same selection logic as test_hybrid.py
    cell_scores: List[float] = []
    for sc in [
        # boundary
        pick_bool(bd_bool),
        pick_cat(bd_cat),
        pick_secs(bd_sec),
        # concurrency
        pick_bool(cc_bool),
        pick_secs(cc_sec),
        pick_cat(cc_choice),
        # count
        pick_count(ct),
        # duration
        pick_secs(dur),
        # extreme
        pick_bool(ex_bool),
        pick_cat(ex_cat),
        pick_secs(ex_sec),
        # motion
        pick_cat(mo_cat),
        # ordering
        pick_bool(od_bool),
        pick_cat(od_cat),
        pick_cat(od_choice),
        # phase_transition
        pick_bool(ph_bool),
        pick_cat(ph_cat),
        pick_secs(ph_sec),
        pick_cat(ph_choice),
    ]:
        if sc is not None:
            cell_scores.append(sc)

    result["overall_macro_score"] = float(np.mean(cell_scores)) if cell_scores else 0.0
    return result


# ----------------------
# CLI
# ----------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--answers-file",
        required=True,
        help="answers_llavamed_*.jsonl produced by run_stsg_vqa.sh",
    )
    ap.add_argument(
        "--output-file",
        required=True,
        help="Where to save evaluation JSON",
    )
    ap.add_argument(
        "--vocabs-json",
        required=True,
        help="Path to vocabs_train.json used by STSG-Net (for text tasks).",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    answers_path = Path(args.answers_file)
    out_path = Path(args.output_file)
    vocabs_path = Path(args.vocabs_json)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[Eval LLaVA-Med] answers_file = {answers_path}")
    print(f"[Eval LLaVA-Med] vocabs_json   = {vocabs_path}")

    res = evaluate_llavamed(answers_path, vocabs_path)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"[Done] Saved to {out_path}")


if __name__ == "__main__":
    main()
