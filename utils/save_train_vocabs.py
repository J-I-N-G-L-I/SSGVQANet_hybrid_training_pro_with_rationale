# # save_train_vocabs.py
# import os, json, glob

# def collect_vocabs(stsg_root, vids):
#     vocabs = {k:set() for k in ["extreme","phase_transition","boundary_text","ordering_text"]}
#     for vid in vids:
#         jpath = os.path.join(stsg_root, vid, "temporal_qa.jsonl")
#         if not os.path.exists(jpath): 
#             print("[skip]", jpath); 
#             continue
#         with open(jpath,"r",encoding="utf-8") as f:
#             for ln in f:
#                 if not ln.strip(): continue
#                 obj = json.loads(ln)
#                 cat = (obj.get("category") 
#                        or obj.get("metadata",{}).get("category"))
#                 ans = obj.get("answer", None)
#                 if ans is None: continue
#                 if cat == "extreme":
#                     vocabs["extreme"].add(str(ans))
#                 elif cat == "phase_transition":
#                     vocabs["phase_transition"].add(str(ans) if not isinstance(ans,(int,float)) else str(ans))
#                 elif cat == "boundary":
#                     if isinstance(ans,str) and ans not in ("True","False"):
#                         vocabs["boundary_text"].add(ans)
#                 elif cat == "ordering":
#                     if isinstance(ans,str) and ans not in ("True","False"):
#                         vocabs["ordering_text"].add(ans)
#     return {k: sorted(list(v)) for k,v in vocabs.items()}

# if __name__ == "__main__":
#     STSG_TRAIN_ROOT = "/mnt/scratch/sc232jl/datasets/SSGVQA/data/STSG_QA_Pro_8_Classes/"
#     TRAIN_VIDS = ["VID73","VID40","VID62","VID42","VID29","VID56","VID50","VID78","VID66","VID13",
#                   "VID52","VID06","VID36","VID05","VID12","VID26","VID68","VID32","VID49","VID65",
#                   "VID47","VID04","VID23","VID79","VID51","VID10","VID57","VID75","VID25","VID14",
#                   "VID15","VID08","VID80","VID27","VID70",]
#                   # "VID18", "VID48", "VID01", "VID35", "VID31",]   # these are validation sets
#     vocabs = collect_vocabs(STSG_TRAIN_ROOT, TRAIN_VIDS)
#     with open("vocabs_train.json","w",encoding="utf-8") as f:
#         json.dump(vocabs,f,ensure_ascii=False,indent=2)
#     print({k: len(v) for k,v in vocabs.items()})


# /users/sc232jl/SSGVQANet_hybrid_training_pro_with_rationale/utils/save_train_vocabs.py
# -*- coding: utf-8 -*-
"""
构建训练集的文本类标签词表（仅字符串答案）：
- 只读取 JSON 文件： {STSG_ROOT}/VIDxx/temporal_qa.json
- 仅当 qa["answer_type"] == "string" 时，才把 qa["answer"] 计入词表
- 支持以下 5 个输出键：
    extreme
    phase_transition
    boundary
    ordering
    motion            (字符串答案；通常为 8 类)
- 每个词表都以 "<UNK>" 作为第 0 号类（索引 0）

注意：
- 本脚本不读取 .jsonl；若你的文件仍为 JSONL，请先转换为 JSON 数组文件。
- 对于 boundary/ordering，我们只收集“非布尔、非数值”的字符串答案（此处依据 answer_type == "string" 已经满足）。
- phase_transition：只收字符串；数字型过渡（若存在）在模型中应该走回归，不进入词表。
"""

import os
import json
from typing import Dict, List, Set

UNK_TOKEN = "<UNK>"
FILENAME = "temporal_qa.json"

_STRING_CAT_KEYS = ("extreme", "phase_transition", "boundary", "ordering", "motion")

def _safe_load_json_array(path: str) -> List[dict]:
    """
    读取 JSON 文件，要求顶层为数组（list[dict]）。
    若文件不存在或格式不合法，抛出 RuntimeError。
    """
    if not os.path.exists(path):
        raise RuntimeError(f"[missing] {path}")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"[malformed json] {path}: {e}")
    if not isinstance(data, list):
        raise RuntimeError(f"[not a json array] {path} (top-level should be a list of qa objects)")
    return data

def _finalize_vocab(vset: Set[str]) -> List[str]:
    """
    将集合转为排序后的 list，并把 UNK_TOKEN 放到第 0 位。
    若集合里本就包含 UNK_TOKEN，确保它仍在第 0 位（去重）。
    """
    # 去掉空白项
    clean = [s for s in (x.strip() for x in vset) if s]
    clean_sorted = sorted(set(clean))  # 排序 + 去重
    if UNK_TOKEN in clean_sorted:
        clean_sorted = [UNK_TOKEN] + [s for s in clean_sorted if s != UNK_TOKEN]
    else:
        clean_sorted = [UNK_TOKEN] + clean_sorted
    return clean_sorted

def collect_vocabs_from_json(stsg_root: str, vids: List[str]) -> Dict[str, List[str]]:
    """
    遍历指定视频列表，从 {stsg_root}/VIDxx/temporal_qa.json 中抽取字符串答案词表。
    返回结构：
    {
      "extreme": [...],
      "phase_transition": [...],
      "boundary": [...],
      "ordering": [...],
      "motion": [...]
    }
    """
    # 中间集合（先去重）
    store: Dict[str, Set[str]] = {
        "extreme": set(),
        "phase_transition": set(),
        "boundary": set(),
        "ordering": set(),
        "motion": set(),
    }

    missing_files = []
    total_read = 0
    taken_by_cat = {k: 0 for k in ["extreme", "phase_transition", "boundary", "ordering", "motion"]}
    skipped_by_type = 0
    skipped_no_cat = 0

    for vid in vids:
        jpath = os.path.join(stsg_root, vid, FILENAME)
        try:
            qas = _safe_load_json_array(jpath)
        except RuntimeError as e:
            print(str(e))
            missing_files.append(vid)
            continue

        for qa in qas:
            total_read += 1

            # 1) 仅保留字符串答案
            ans_type = str(qa.get("answer_type", "")).strip().lower()
            if ans_type != "string":
                skipped_by_type += 1
                continue

            # 2) 取类别：优先 qa["category"]，否则 evidence.category
            cat = qa.get("category") or (qa.get("evidence") or {}).get("category")
            if not cat:
                skipped_no_cat += 1
                continue

            cat = str(cat).strip()
            if cat not in _STRING_CAT_KEYS:
                continue

            # 3) 取字符串答案
            ans = qa.get("answer", None)
            if ans is None:
                # 理论上 string 类型不应为空，防御性跳过
                continue
            ans_str = str(ans).strip()
            if not ans_str:
                continue
            
            store[cat].add(ans_str)
            taken_by_cat[cat] += 1

    # 5) 输出：每个词表首位为 UNK
    out = {k: _finalize_vocab(v) for k, v in store.items()}

    print("save_train_vocabs: summary")
    print(f"videos scanned     : {len(vids)} (missing files: {len(missing_files)})")
    if missing_files:
        print(f"  missing json for : {sorted(missing_files)}")
    print(f"total qa read      : {total_read}")
    print(f"skipped by type    : {skipped_by_type}  (answer_type != 'string')")
    print(f"skipped no category: {skipped_no_cat}")
    print("taken counts       :", {k: v for k, v in taken_by_cat.items()})
    print("vocab sizes        :", {k: len(v) for k, v in out.items()})
    print("UNK token          :", UNK_TOKEN, "(index 0 for all vocabs)")

    return out

if __name__ == "__main__":
    STSG_TRAIN_ROOT = "/mnt/scratch/sc232jl/datasets/SSGVQA/data/STSG_QA_Pro_8_Classes/"
    TRAIN_VIDS = [
        "VID73","VID40","VID62","VID42","VID29","VID56","VID50","VID78","VID66","VID13",
        "VID52","VID06","VID36","VID05","VID12","VID26","VID68","VID32","VID49","VID65",
        "VID47","VID04","VID23","VID79","VID51","VID10","VID57","VID75","VID25","VID14",
        "VID15","VID08","VID80","VID27","VID70",
        # "VID18", "VID48", "VID01", "VID35", "VID31",]   # these are validation sets
    ]

    vocabs = collect_vocabs_from_json(STSG_TRAIN_ROOT, TRAIN_VIDS)

    out_path = "vocabs_train.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=2)

    print(f"[done] saved to: {os.path.abspath(out_path)}")
