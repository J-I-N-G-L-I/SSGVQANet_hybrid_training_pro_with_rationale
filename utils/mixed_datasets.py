# -*- coding: utf-8 -*-

import os
import json
import glob
import h5py
import logging
from typing import List, Dict, Any, Optional, Tuple

import torch
from torch.utils.data import Dataset
import re

# logger
def build_logger(name="dataset_logger", log_file=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        if log_file is not None:
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(fmt)
            fh.setLevel(level)
            logger.addHandler(fh)
    return logger


# unified visual features reading + processing some frames missing
class FrameFeatureStore:
    """
    统一读取一个视频的“ROI(530) + 全局(512)”并拼接成 530 维对象级特征：
      - ROI文件:  {folder_head}/visual_feats/roi_yolo_coord/{vid}/labels/vqa/img_features/roi/{frame:06d}.hdf5
      - 全局文件: {folder_head}/visual_feats/cropped_images/{vid}/vqa/img_features/1x1/{frame:06d}.hdf5
    若缺帧：
      - 优先在给定 candidates 中找最近有文件的帧；
      - 否则在 ±search_radius 内搜索；
      - 若仍然没有：回退为 1 个 “伪ROI token”，前18维=0，后512维=全局512（若全局也没文件则全0），并记录日志。
    """

    def __init__(self, folder_head: str,
                 search_radius: int = 8,
                 logger: Optional[logging.Logger] = None):
        self.folder_head = folder_head
        self.search_radius = search_radius
        self.logger = logger or build_logger()
        # existence cache: (vid, frame) -> {'roi': bool, 'pix': bool}
        self._exists_cache: Dict[Tuple[str, int], Dict[str, bool]] = {}

    # def _roi_path(self, vid: str, frame_idx: int) -> str:
    #     return os.path.join(
    #         self.folder_head, "visual_feats", "roi_yolo_coord", vid,
    #         "labels", "vqa", "img_features", "roi", f"{frame_idx:06d}.hdf5"
    #     )

    # def _pix_path(self, vid: str, frame_idx: int) -> str:
    #     return os.path.join(
    #         self.folder_head, "visual_feats", "cropped_images", vid,
    #         "vqa", "img_features", "1x1", f"{frame_idx:06d}.hdf5"
    #     )
    
    # change to new extracted features
    # def _roi_path(self, vid: str, frame_idx: int) -> str:
    #     return os.path.join(
    #         self.folder_head, "visual_feats", "roi_yolo_coord", vid,
    #         "labels", "vqa", "img_features", "roi", f"{frame_idx:06d}.hdf5"
    #     )

    # change to new extracted features
    def _roi_path(self, vid: str, frame_idx:int) -> str:
        return os.path.join(
            self.folder_head, "visual_feats", "roi_yolo_coord_new", vid,
            "vqa", "img_features", "roi", f"{frame_idx:06d}.hdf5"
        )

    # def _pix_path(self, vid: str, frame_idx: int) -> str:
    #     return os.path.join(
    #         self.folder_head, "visual_feats", "cropped_images", vid,
    #         "vqa", "img_features", "1x1", f"{frame_idx:06d}.hdf5"
    #     )

    def _pix_path(self, vid: str, frame_idx: int) -> str:
        return os.path.join(
            self.folder_head, "visual_feats", "cropped_images_new", vid,
            "vqa", "img_features", "1x1", f"{frame_idx:06d}.hdf5"
        )

    def _check_exists(self, vid: str, frame_idx: int) -> Dict[str, bool]:
        key = (vid, frame_idx)
        if key in self._exists_cache:
            return self._exists_cache[key]
        roi_p = self._roi_path(vid, frame_idx)
        pix_p = self._pix_path(vid, frame_idx)
        info = {"roi": os.path.exists(roi_p), "pix": os.path.exists(pix_p)}
        self._exists_cache[key] = info
        return info

    def _load_one(self, vid: str, frame_idx: int) -> Tuple[torch.Tensor, bool]:
        """
        return (features_530[K,530], ok)
        """
        roi_p = self._roi_path(vid, frame_idx)
        pix_p = self._pix_path(vid, frame_idx)
        if not os.path.exists(roi_p) and not os.path.exists(pix_p):
            return torch.empty(0, 530), False

        # global: 512 dims
        if os.path.exists(pix_p):
            with h5py.File(pix_p, "r") as f:
                glb = torch.from_numpy(f["visual_features"][:])  # (1,512) or (512,)
            glb = glb.float().view(-1)  # (512,)
        else:
            glb = torch.zeros(512, dtype=torch.float32)

        # ROI: 530, but only first 18 dims
        if os.path.exists(roi_p):
            with h5py.File(roi_p, "r") as f:
                roi = torch.from_numpy(f["visual_features"][:]).float()  # (K,530)
            if roi.numel() == 0:
                # degrade to a single token
                feat = torch.zeros(1, 530, dtype=torch.float32)
                feat[:, 18:] = glb.view(1, 512)
                return feat, True
            # use global 512 to replace the last 512 dim of ROI
            roi[:, 18:] = glb.view(1, 512).expand(roi.size(0), -1)
            return roi, True
        else:
            # if only have global features, construct a single token
            feat = torch.zeros(1, 530, dtype=torch.float32)
            feat[:, 18:] = glb.view(1, 512)
            return feat, True

    def _nearest_available(self, vid: str, target: int,
                           candidates: Optional[List[int]]) -> Optional[int]:
        """
        在 candidates 或 ±search_radius 中寻找最近的可用帧（任一特征文件存在即认为可用）
        """
        def is_ok(fidx: int) -> bool:
            ex = self._check_exists(vid, fidx)
            return ex["roi"] or ex["pix"]

        # candidates 优先
        if candidates:
            usable = [f for f in candidates if is_ok(f)]
            if usable:
                usable.sort(key=lambda x: abs(x - target))
                return usable[0]

        # ±search_radius
        for d in range(1, self.search_radius + 1):
            for f in (target - d, target + d):
                if f >= 0 and is_ok(f):
                    return f
        return None

    def get_features(self, vid: str, frame_idx: int,
                     candidates: Optional[List[int]] = None
                     ) -> Tuple[torch.Tensor, int, bool, Optional[int]]:
        """
        返回: (feat_530[K,530], used_frame_idx, ok, fallback_from)
          - used_frame_idx: 实际加载的帧号
          - ok: 是否真实读到文件（不含纯0）
          - fallback_from: 若发生替代，记录原始帧号；否则 None
        """
        info = self._check_exists(vid, frame_idx)
        if info["roi"] or info["pix"]:
            feat, ok = self._load_one(vid, frame_idx)
            return feat, frame_idx, ok, None

        # 缺帧：尝试回退
        alt = self._nearest_available(vid, frame_idx, candidates)
        if alt is not None:
            feat, ok = self._load_one(vid, alt)
            if self.logger:
                self.logger.warning(f"[{vid}] frame {frame_idx} missing; fallback -> {alt}")
            return feat, alt, ok, frame_idx

        # 实在找不到：用全 0 token
        if self.logger:
            self.logger.error(f"[{vid}] frame {frame_idx} missing; no fallback found, use zeros")
        feat = torch.zeros(1, 530, dtype=torch.float32)
        return feat, frame_idx, False, frame_idx


# # ===========================================
# # SSG-VQA 单帧数据（静态） 原始
# # ===========================================
# class SSGVQAFrameDataset(Dataset):
#     """
#     读取原项目 SSG-VQA 的逐帧问答：
#       - qa_txt: /path/.../ssg-qa/VID01/1.txt （每行: "What ... ?|liver"）
#       - 视觉特征按 FrameFeatureStore 读取
#     输出 sample:
#       {
#         'vid': str,
#         'frame': int,
#         'question': str,
#         'visual_embeds': (R,530) FloatTensor,
#         'visual_frame_ids': (R,) LongTensor(全0),
#         'task': 'answer_cls',
#         'label': LongTensor/str（建议在 collate 统一转为 id 或 one-hot）,
#         'answer': str,
#         'meta': {...}
#       }
#     """
#     def __init__(self,
#                  qa_root: str,
#                  folder_head: str,
#                  seq: Optional[List[str]] = None,
#                  label_vocab: Optional[List[str]] = None,
#                  logger: Optional[logging.Logger] = None,
#                  *,
#                  multi_frames: bool = False,          # whether also use multi-frames in static qa
#                  frames_per_sample: int = 1):        # window size
#         self.qa_root = qa_root
#         self.folder_head = folder_head
#         self.logger = logger or build_logger()
#         self.store = FrameFeatureStore(folder_head, logger=self.logger)

#         self.multi_frames = bool(multi_frames)
#         self.frames_per_sample = max(1, int(frames_per_sample))

#         # 原 SSG-VQA 的静态答案词表（若不传则使用默认）
#         self.label_vocab = label_vocab or [
#             "0","1","10","2","3","4","5","6","7","8","9",
#             "False","True",
#             "abdominal_wall_cavity","adhesion","anatomy","aspirate","bipolar",
#             "blood_vessel","blue","brown","clip","clipper","coagulate","cut",
#             "cystic_artery","cystic_duct","cystic_pedicle","cystic_plate",
#             "dissect","fluid","gallbladder","grasp","grasper","gut","hook",
#             "instrument","irrigate","irrigator","liver","omentum","pack",
#             "peritoneum","red","retract","scissors","silver","specimen_bag",
#             "specimenbag","white","yellow",
#         ]
#         self.label2id = {s: i for i, s in enumerate(self.label_vocab)}

#         # 扫描视频目录
#         vids = []
#         if seq is None:
#             for p in glob.glob(os.path.join(self.qa_root, "VID*")):
#                 if os.path.isdir(p):
#                     vids.append(os.path.basename(p))
#         else:
#             for s in seq:
#                 vs = s if s.startswith("VID") else f"VID{int(s):02d}"
#                 vdir = os.path.join(self.qa_root, vs)
#                 if os.path.isdir(vdir):
#                     vids.append(vs)

#         # 读取每个视频的 txt 问答
#         self.samples = []
#         for vid in vids:
#             vdir = os.path.join(self.qa_root, vid)
#             for txt in glob.glob(os.path.join(vdir, "*.txt")):
#                 frame_idx = int(os.path.splitext(os.path.basename(txt))[0])
#                 with open(txt, "r", encoding="utf-8") as f:
#                     for ln in f:
#                         s = ln.strip()
#                         if not s or "|" not in s:
#                             continue
#                         q, ans = s.split("|", 1)
#                         ans = ans.strip()
#                         self.samples.append((vid, frame_idx, q, ans))

#         self.logger.info(f"[SSGVQAFrameDataset] videos={len(vids)} total_qa_pairs={len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx: int) -> Dict[str, Any]:
#         vid, frame_idx, question, answer = self.samples[idx]

#         # -------------------------
#         # 1) 视觉特征：单帧 or 多帧
#         # -------------------------
#         if not self.multi_frames or self.frames_per_sample <= 1:
#             # 旧行为：只取当前这一帧
#             feats, used_frame, ok, fallback_from = self.store.get_features(
#                 vid, frame_idx, candidates=None
#             )
#             frame_ids = torch.zeros(feats.size(0), dtype=torch.long)

#             meta = {
#                 "vid": vid, # for visualization
#                 "primary_frame": int(frame_idx),
#                 "used_frame": int(used_frame),
#                 "fallback_from": int(fallback_from) if fallback_from is not None else None,
#                 "ok": bool(ok),
#                 "frames_requested": [int(frame_idx)],
#                 "frames_used": [int(used_frame)],
#                 "substitutions": [],
#                 "category": "answer",             # for visualization      
#                 "has_rationale": False,      # for visualization
#             }

#         else:
#             # 新行为：静态多帧模式
#             K = self.frames_per_sample   # 例如 12

#             # 关键修改：帧号从 0 开始，下限设为 0
#             # 窗口为 [start, ..., frame_idx]，长度可能 < K
#             start = max(0, int(frame_idx) - (K - 1))
#             base_frames = list(range(start, int(frame_idx) + 1))  # [start, ..., frame_idx]

#             # 若不足 K 帧，用最后一帧（frame_idx）复制补齐
#             frames_window = list(base_frames)
#             while len(frames_window) < K:
#                 frames_window.append(int(frame_idx))

#             feats_list = []
#             fids_list = []
#             used_frames = []
#             substitutions = []
#             ok_flags = []

#             # 候选帧集合（用于缺帧 fallback）
#             candidates_for_fallback = sorted(set(frames_window))

#             for pos, f in enumerate(frames_window):
#                 feat, used, ok, fallback_from = self.store.get_features(
#                     vid, f, candidates=candidates_for_fallback
#                 )
#                 feats_list.append(feat)  # (R_i, 530)
#                 # pos 就是时间步 0..K-1，供 temporal position embedding 使用
#                 fids_list.append(torch.full((feat.size(0),), pos, dtype=torch.long))
#                 used_frames.append(int(used))
#                 ok_flags.append(bool(ok))
#                 if fallback_from is not None:
#                     substitutions.append({
#                         "pos": int(pos),
#                         "from": int(fallback_from),
#                         "to": int(used),
#                     })

#             feats = torch.cat(feats_list, dim=0)    # (ΣR_i, 530)
#             frame_ids = torch.cat(fids_list, dim=0) # (ΣR_i,)

#             meta = {
#                 "vid": vid,             # for visualization
#                 "primary_frame": int(frame_idx),        # 当前 QA 真正对应的帧
#                 "used_frame": used_frames[-1],          # 主帧实际使用的特征（最后一个时间步）
#                 "fallback_from": None,                  # 单帧模式用，这里保留字段即可
#                 "ok": all(ok_flags),
#                 "frames_requested": [int(f) for f in frames_window],
#                 "frames_used": used_frames,
#                 "substitutions": substitutions,
#                 "category": "answer",                   # for visualization
#                 "has_rationale": False,                  # for visualization
#             }

#         # -------------------------
#         # 2) 标签（保持原逻辑）
#         # -------------------------
#         if answer in self.label2id:
#             label = torch.tensor(self.label2id[answer], dtype=torch.long)
#         else:
#             # 未登录答案（极少）：返回字符串，交由上层处理
#             label = answer

#         return {
#             "vid": vid,
#             "frame": frame_idx,
#             "question": question,
#             "visual_embeds": feats,               # (R,530) 或 (ΣR_i,530)
#             "visual_frame_ids": frame_ids,        # (R,) 或 (ΣR_i,)
#             "task": "answer_cls",
#             "label": label,
#             "answer": answer,
#             "meta": meta,
#         }


# 为了画出单帧/多帧的注意力图：
class SSGVQAFrameDataset(Dataset):
    """
    读取原项目 SSG-VQA 的逐帧问答：
      - qa_txt: /path/.../ssg-qa/VID01/1.txt （每行: "What ... ?|liver"）
      - 视觉特征按 FrameFeatureStore 读取

    单帧模式（multi_frames=False）：
        一个样本只使用当前帧的视觉特征。

    多帧模式（multi_frames=True, frames_per_sample=K）：
        对于 anchor 帧 frame_idx，构造一个长度为 K 的窗口：
            [max(0, frame_idx-K+1), ..., frame_idx]
        若不足 K 帧，则用 frame_idx 复制补齐。
        每个时间步 t 的所有 token 的 visual_frame_ids = 该时间步在窗口中的相对索引 (0..K-1)。

    输出 sample:
      {
        'vid': str,
        'frame': int,                # anchor frame
        'question': str,
        'visual_embeds': FloatTensor (R, 530)  # 或 ΣR_i
        'visual_frame_ids': LongTensor (R,),   # 相对帧 id，单帧时全 0
        'task': 'answer_cls',
        'label': LongTensor 或 str,
        'answer': str,
        'meta': {
            'vid': str,
            'primary_frame': int,            # 静态 QA 对应的原始帧
            'used_frame': int,               # 主帧实际使用的特征
            'ok': bool,
            'frames_requested': List[int],   # 请求的窗口帧序列（长度 = K 或 1）
            'frames_used': List[int],        # 实际使用的帧序列（长度同上）
            'substitutions': List[...],      # fallback 记录
            'category': 'answer',
            'has_rationale': False,
        }
      }
    """
    def __init__(self,
                 qa_root: str,
                 folder_head: str,
                 seq: Optional[List[str]] = None,
                 label_vocab: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None,
                 *,
                 multi_frames: bool = False,      # 是否使用多帧
                 frames_per_sample: int = 1):     # 窗口长度 K
        self.qa_root = qa_root
        self.folder_head = folder_head
        self.logger = logger or build_logger()
        self.store = FrameFeatureStore(folder_head, logger=self.logger)

        self.multi_frames = bool(multi_frames)
        self.frames_per_sample = max(1, int(frames_per_sample))

        # 静态答案词表
        self.label_vocab = label_vocab or [
            "0","1","10","2","3","4","5","6","7","8","9",
            "False","True",
            "abdominal_wall_cavity","adhesion","anatomy","aspirate","bipolar",
            "blood_vessel","blue","brown","clip","clipper","coagulate","cut",
            "cystic_artery","cystic_duct","cystic_pedicle","cystic_plate",
            "dissect","fluid","gallbladder","grasp","grasper","gut","hook",
            "instrument","irrigate","irrigator","liver","omentum","pack",
            "peritoneum","red","retract","scissors","silver","specimen_bag",
            "specimenbag","white","yellow",
        ]
        self.label2id = {s: i for i, s in enumerate(self.label_vocab)}

        # 扫描视频目录
        vids = []
        if seq is None:
            for p in glob.glob(os.path.join(self.qa_root, "VID*")):
                if os.path.isdir(p):
                    vids.append(os.path.basename(p))
        else:
            for s in seq:
                vs = s if s.startswith("VID") else f"VID{int(s):02d}"
                vdir = os.path.join(self.qa_root, vs)
                if os.path.isdir(vdir):
                    vids.append(vs)

        # 读取每个视频的 txt 问答
        self.samples = []
        for vid in vids:
            vdir = os.path.join(self.qa_root, vid)
            for txt in glob.glob(os.path.join(vdir, "*.txt")):
                frame_idx = int(os.path.splitext(os.path.basename(txt))[0])
                with open(txt, "r", encoding="utf-8") as f:
                    for ln in f:
                        s = ln.strip()
                        if not s or "|" not in s:
                            continue
                        q, ans = s.split("|", 1)
                        ans = ans.strip()
                        self.samples.append((vid, frame_idx, q, ans))

        self.logger.info(f"[SSGVQAFrameDataset] videos={len(vids)} total_qa_pairs={len(self.samples)} "
                         f"(multi_frames={self.multi_frames}, K={self.frames_per_sample})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        vid, frame_idx, question, answer = self.samples[idx]
        frame_idx = int(frame_idx)

        # ------------------------------------------------
        # 1) 视觉特征：单帧 or 多帧
        # ------------------------------------------------
        if (not self.multi_frames) or self.frames_per_sample <= 1:
            # ===== 单帧模式：保持原来的行为 =====
            feats, used_frame, ok, fallback_from = self.store.get_features(
                vid, frame_idx, candidates=None
            )
            # 所有 token 的 frame_id = 0
            frame_ids = torch.zeros(feats.size(0), dtype=torch.long)

            frames_requested = [frame_idx]
            frames_used = [int(used_frame)]
            substitutions = []
            if fallback_from is not None and fallback_from != used_frame:
                substitutions.append({
                    "pos": 0,
                    "from": int(fallback_from),
                    "to": int(used_frame),
                })

            meta = {
                "vid": vid,                        # for visualization
                "primary_frame": frame_idx,        # 静态 QA 对应的帧
                "used_frame": int(used_frame),     # 主帧实际使用的特征
                "fallback_from": int(fallback_from) if fallback_from is not None else None,
                "ok": bool(ok),
                "frames_requested": frames_requested,
                "frames_used": frames_used,
                "substitutions": substitutions,
                "category": "answer",              # for visualization + filtering
                "has_rationale": False,            # for visualization
            }

        else:
            # ===== 多帧模式：静态多帧窗口 =====
            K = self.frames_per_sample

            # 窗口：[start, ..., frame_idx]，不足 K 用 frame_idx 补齐
            start = max(0, frame_idx - (K - 1))
            base_frames = list(range(start, frame_idx + 1))   # 最长到当前帧
            frames_window = list(base_frames)
            while len(frames_window) < K:
                frames_window.append(frame_idx)

            feats_list = []
            fids_list = []
            used_frames = []
            ok_flags = []
            substitutions = []

            # fallback 候选集合：窗口内所有帧
            candidates_for_fallback = sorted(set(frames_window))

            for pos, f in enumerate(frames_window):
                feat, used, ok, fallback_from = self.store.get_features(
                    vid, f, candidates=candidates_for_fallback
                )
                # feat: (R_i, 530)
                feats_list.append(feat)
                # 当前时间步 pos 的所有 token 标记为该 pos
                fids_list.append(torch.full((feat.size(0),), pos, dtype=torch.long))

                used_frames.append(int(used))
                ok_flags.append(bool(ok))

                if fallback_from is not None and fallback_from != used:
                    substitutions.append({
                        "pos": int(pos),
                        "from": int(fallback_from),
                        "to": int(used),
                    })

            feats = torch.cat(feats_list, dim=0)      # (ΣR_i, 530)
            frame_ids = torch.cat(fids_list, dim=0)   # (ΣR_i,)

            meta = {
                "vid": vid,                           # for visualization
                "primary_frame": frame_idx,           # 静态 QA 真实对应的帧
                # 主帧的实际使用帧：窗口最后一个时间步
                "used_frame": used_frames[-1],
                "fallback_from": None,                # 单帧分支用，这里保留字段即可
                "ok": all(ok_flags),
                "frames_requested": [int(f) for f in frames_window],
                "frames_used": used_frames,
                "substitutions": substitutions,
                "category": "answer",
                "has_rationale": False,
            }

        # ------------------------------------------------
        # 2) 标签（保持原逻辑）
        # ------------------------------------------------
        if answer in self.label2id:
            label = torch.tensor(self.label2id[answer], dtype=torch.long)
        else:
            # 未登录答案（极少）：返回字符串，交由上层处理
            label = answer

        return {
            "vid": vid,
            "frame": frame_idx,
            "question": question,
            "visual_embeds": feats,            # (R,530) 或 (ΣR_i,530)
            "visual_frame_ids": frame_ids,     # (R,) 或 (ΣR_i,)
            "task": "answer_cls",
            "label": label,
            "answer": answer,
            "meta": meta,
        }


# STSG temporal QA
class STSGTemporalQADataset(Dataset):
    """
    读取 STSG 生成的 {vid}/temporal_qa.json（顶层 JSON 数组）：
      - 仅接受 allowed_categories 中的样本；
      - 将 evidence.keyframes 或 scope.frames（起止帧）重采样为 <= max_frames；
      - 为每个 keyframe 读取 ROI+全局特征，拼接成 (ΣR_i, 530)；同时给出 visual_frame_ids (ΣR_i,)；
      - 标签路由（优先依据 qa["answer_type"]）：
          * count            : count -> count_cls（20 类，标签 0..19；语义 1..20）
          * duration         : 回归（秒），若答案非数值则尝试 evidence 估计
          * boundary         : boolean -> 二分类；numeric -> 回归；string -> boundary_text（从 fixed_vocabs["boundary"]）
          * ordering         : boolean -> 二分类；numeric -> 计数多分类；string -> ordering_text（从 fixed_vocabs["ordering"]）
          * motion           : string 多分类（从 fixed_vocabs["motion"]）
          * extreme          : string 多分类（从 fixed_vocabs["extreme"]）
          * phase_transition : numeric -> 回归；string -> 多分类（从 fixed_vocabs["phase_transition"]）
          * concurrency      : boolean -> 二分类；numeric -> 回归；其他 -> Fallback 到二分类(0)并告警
      - 不做任何词表自建；固定依赖 fixed_vocabs 进行字符串类映射。
    """

    UNK = "<UNK>"

    def __init__(self,
                 qa_root: str,
                 folder_head: str,
                 vids: Optional[List[str]] = None,
                 allowed_categories: Optional[List[str]] = None,
                 max_frames: int = 16,
                 fps: float = 1.0,
                 logger: Optional[logging.Logger] = None,
                 fixed_vocabs: Optional[Dict[str, List[str]]] = None):
        self.qa_root = qa_root
        self.folder_head = folder_head
        self.max_frames = max_frames
        self.fps = fps
        self.logger = logger or build_logger()
        self.store = FrameFeatureStore(folder_head, logger=self.logger)

        # 1) 类别集合
        self.allowed = set(allowed_categories or [
            "count", "duration", "ordering", "extreme",
            "boundary", "motion", "concurrency", "phase_transition",
        ])

        # 2) 词表：仅来自外部
        if not fixed_vocabs or not isinstance(fixed_vocabs, dict):
            raise ValueError(
                "[STSGTemporalQADataset] 'fixed_vocabs' must be provided and be a dict "
                "with keys: 'extreme','phase_transition','boundary','ordering','motion' "
                f"(UNK at index 0 as '{self.UNK}')"
            )

        self.vocabs: Dict[str, List[str]] = {}
        for key in ["extreme", "phase_transition", "boundary", "ordering", "motion"]:
            lst = list(fixed_vocabs.get(key, []))
            if not lst:
                # 若缺失则最小化补一个 UNK，避免训练时崩溃
                lst = [self.UNK]
                self.logger.warning(f"[STSGTemporalQADataset] fixed_vocabs['{key}'] missing or empty; use ['{self.UNK}']")
            else:
                # 确保 UNK 在第 0 位
                if lst[0] != self.UNK:
                    if self.UNK in lst:
                        lst = [self.UNK] + [x for x in lst if x != self.UNK]
                    else:
                        lst = [self.UNK] + lst
            self.vocabs[key] = lst

        # 3) 构建字符串->id 映射
        self.extreme2id = {s: i for i, s in enumerate(self.vocabs["extreme"])}
        self.phase2id   = {s: i for i, s in enumerate(self.vocabs["phase_transition"])}
        self.boundary2id = {s: i for i, s in enumerate(self.vocabs["boundary"])}
        self.ordering2id = {s: i for i, s in enumerate(self.vocabs["ordering"])}
        self.motion2id   = {s: i for i, s in enumerate(self.vocabs["motion"])}

        # 4) 视频列表
        if vids is None:
            vids = []
            for p in glob.glob(os.path.join(self.qa_root, "VID*")):
                if os.path.isdir(p):
                    vids.append(os.path.basename(p))
        self.vids = vids

        # 5) 逐视频读取 temporal_qa.json（JSON 数组）
        self.samples: List[Dict[str, Any]] = []
        for vid in self.vids:
            jpath = os.path.join(self.qa_root, vid, "temporal_qa.json")
            if not os.path.exists(jpath):
                self.logger.warning(f"[STSGTemporalQADataset] {vid} missing json: {jpath}")
                continue

            try:
                with open(jpath, "r", encoding="utf-8-sig") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    raise RuntimeError("top-level must be a JSON array")
            except Exception as e:
                self.logger.error(f"[STSGTemporalQADataset] bad json in {jpath}: {e}")
                continue

            for obj in data:
                cat = obj.get("category") or (obj.get("evidence") or {}).get("category")
                if cat not in self.allowed:
                    continue
                self.samples.append({"vid": vid, "qa": obj})

        self.logger.info(
            "[STSGTemporalQADataset] videos=%d total_samples=%d | "
            "vocab_sizes: extreme=%d, phase=%d, boundary=%d, ordering=%d, motion=%d",
            len(self.vids), len(self.samples),
            len(self.vocabs["extreme"]), len(self.vocabs["phase_transition"]),
            len(self.vocabs["boundary"]), len(self.vocabs["ordering"]),
            len(self.vocabs["motion"])
        )

    # -------------------- 工具函数 --------------------
    def _resample_to_k(self, frames: List[int], k: int) -> Tuple[List[int], List[int]]:
        """
        返回 (targets, anchors)
        - anchors: 原始关键帧（去重、排序后）
        - targets: 目标采样帧序列，长度尽量 = k
            * 若 len(anchors) >= k: 在 anchors 上均匀选 k 个
            * 若 len(anchors) <  k: 在 [min(anchors), max(anchors)] 上线性插值取 k 个整数
        """
        if not frames:
            return ([0] * k, [])  # 极端防御：没有任何帧时返回 k 个 0

        anchors = sorted(set(int(x) for x in frames))
        if len(anchors) >= k:
            idx = torch.linspace(0, len(anchors) - 1, steps=k).round().long().tolist()
            targets = [anchors[i] for i in idx]
            return targets, anchors

        lo, hi = anchors[0], anchors[-1]
        if lo == hi:
            targets = [lo] * k
        else:
            targets = torch.linspace(lo, hi, steps=k).round().long().tolist()
            # 稳妥处理非递增
            for t in range(1, k):
                if targets[t] < targets[t - 1]:
                    targets[t] = targets[t - 1]
        return targets, anchors

    def _is_bool_like(self, x) -> bool:
        if isinstance(x, bool):
            return True
        if isinstance(x, str) and x.strip().lower() in ("true", "false"):
            return True
        return False

    def _as_bool_id(self, x) -> int:
        if isinstance(x, bool):
            return 1 if x else 0
        return 1 if str(x).strip().lower() == "true" else 0

    def _is_number_like(self, x) -> bool:
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return True
        if isinstance(x, str):
            s = x.strip()
            if s.endswith("+"):  # 例如 "10+"
                return False
            try:
                float(s)
                return True
            except Exception:
                return False
        return False

    def _safe_float(self, x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    def _duration_from_evidence(self, qa: Dict[str, Any]) -> Optional[float]:
        ev = qa.get("evidence", {}) or {}
        # 1) 显式持续时间
        if "duration_seconds" in ev:
            try:
                return float(ev["duration_seconds"])
            except Exception:
                pass
        # 2) overlap_frames / frame_range
        if "overlap_frames" in ev and ev["overlap_frames"]:
            s, e = ev["overlap_frames"]
            return float(max(0, int(e) - int(s) + 1)) / self.fps
        if "frame_range" in ev and ev["frame_range"]:
            s, e = ev["frame_range"]
            return float(max(0, int(e) - int(s) + 1)) / self.fps
        # 3) 最后尝试答案本身
        try:
            return float(qa.get("answer"))
        except Exception:
            return None

    def _count_to_class(self, ans) -> int:
        """
        严格：答案为 1..20 的整数 -> 0..19；其余情况一律映射到 0 并告警
        """
        try:
            iv = int(float(ans))
            if 1 <= iv <= 20:
                return iv - 1
            self.logger.warning(f"[count_to_class] out-of-range {ans} (expect 1..20) -> map to 0")
            return 0
        except Exception:
            self.logger.warning(f"[count_to_class] bad answer={ans} -> map to 0")
            return 0
        
    # def _parse_choice_K(self, question: str) -> Optional[int]:
    #     """
    #     从题目尾部解析 choice 的 K：兼容
    #     - "... Answer with a number 1-{K}"
    #     - "... Reply 1-{K}"
    #     - "... Answer 1-{K}"
    #     忽略大小写，末尾可有 . 或 !
    #     返回 K (int)，若未匹配返回 None；若 K>5 则 clip 到 5 并记日志。
    #     """
    #     if not isinstance(question, str):
    #         return None
    #     pat = re.compile(r'(answer|reply)\s+(?:with a number\s+)?1-(\d+)\s*[.!]?\s*$', re.IGNORECASE)
    #     m = pat.search(question.strip())
    #     if not m:
    #         return None
    #     try:
    #         K = int(m.group(2))
    #         if K > 5:
    #             self.logger.warning(f"[choice] parsed K={K} > 5; clip to 5")
    #             K = 5
    #         if K < 1:
    #             return None
    #         return K
    #     except Exception:
    #         return None
    def _parse_choice_K(self, question: str) -> Optional[int]:
        """
        从题目尾部解析 choice 的 K：兼容
        - "... Answer with a number 1-{K}"
        - "... Reply 1-{K}"
        - "... Answer 1-{K}"
        末尾可有 . 或 !
        """
        if not isinstance(question, str):
            return None
        s = question.strip()
        # --- 新增：把常见破折号统一成 ASCII '-' ---
        s = (s.replace('\u2013', '-')  # EN DASH –
            .replace('\u2014', '-')   # EM DASH —
            .replace('\u2212', '-')   # MINUS SIGN −
            .replace('\u2012', '-'))  # FIGURE DASH ‒

        pat = re.compile(r'(answer|reply)\s+(?:with a number\s+)?1-(\d+)\s*[\.\!\)]?\s*$',
                        re.IGNORECASE)
        m = pat.search(s)
        if not m:
            return None
        try:
            K = int(m.group(2))
            if K > 5:
                self.logger.warning(f"[choice] parsed K={K} > 5; clip to 5")
                K = 5
            if K < 1:
                return None
            return K
        except Exception:
            return None

    # -------------------- Dataset API --------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        vid = item["vid"]
        qa = item["qa"]

        question = qa.get("question", "")
        category = qa.get("category") or (qa.get("evidence") or {}).get("category") or "unknown"
        evidence = qa.get("evidence", {}) or {}
        answer_type = str(qa.get("answer_type", "")).strip().lower()
        aval = qa.get("answer", None)

        # --------- 帧选择：keyframes 或 scope.frames ----------
        keyframes = evidence.get("keyframes") or []
        if not keyframes:
            scope = evidence.get("scope") or {}
            sf = scope.get("frames")
            if isinstance(sf, list) and len(sf) == 2:
                s, e = int(sf[0]), int(sf[1])
                if s > e: s, e = e, s
                keyframes = list(range(s, e + 1))
            elif isinstance(sf, list) and len(sf) > 2:
                keyframes = [int(x) for x in sf]
            else:
                # 兼容旧字段
                for cand in ["scope_frames", "phase_frames", "event_frames", "event_frames_range"]:
                    rng = evidence.get(cand)
                    if isinstance(rng, list) and len(rng) == 2:
                        s, e = int(rng[0]), int(rng[1])
                        if s > e: s, e = e, s
                        keyframes = list(range(s, e + 1))
                        break

        keyframes = sorted(set(int(x) for x in keyframes))
        frames, anchors = self._resample_to_k(keyframes, self.max_frames)
        if not frames:
            frames, anchors = [0], []

        # --------- 读取特征 ----------
        candidates_for_fallback = sorted(set(anchors + frames))
        feats_list, fids_list = [], []
        used_frames, substitutions = [], []
        for i, f in enumerate(frames):
            feat, used, ok, fallback_from = self.store.get_features(
                vid, f, candidates=candidates_for_fallback
            )
            feats_list.append(feat)  # (R_i, 530)
            fids_list.append(torch.full((feat.size(0),), i, dtype=torch.long))
            used_frames.append(int(used))
            if fallback_from is not None:
                substitutions.append({"from": int(fallback_from), "to": int(used)})

        visual_embeds = torch.cat(feats_list, dim=0)  # (ΣR_i, 530)
        visual_frame_ids = torch.cat(fids_list, dim=0)  # (ΣR_i,)

        # --------- 任务与标签路由（优先依据 answer_type） ----------
        def _bool_id(x):      
            return self._as_bool_id(x)
        def _is_bool(x):      
            return self._is_bool_like(x)
        def _is_num(x):       
            return self._is_number_like(x)
        def _str2id(mapping, s): 
            return mapping.get(str(s).strip(), 0)
        
        K = self._parse_choice_K(question)

        meta: Dict[str, Any] = {}

        cat = category  # alias
        task: str
        label: torch.Tensor

        if cat == "count":
            task = "count_cls"
            label = torch.tensor(self._count_to_class(aval), dtype=torch.long)

        elif cat == "duration":
            task = "duration_reg"
            if answer_type == "numeric" or _is_num(aval):
                label = torch.tensor(self._safe_float(aval, 0.0), dtype=torch.float32)
            else:
                dur = self._duration_from_evidence(qa)
                label = torch.tensor(float(dur if dur is not None else 0.0), dtype=torch.float32)

        elif cat == "boundary":
            if answer_type == "numeric" or _is_num(aval):
                task = "boundary_reg"
                label = torch.tensor(self._safe_float(aval, 0.0), dtype=torch.float32)
            elif answer_type == "boolean" or _is_bool(aval):
                task = "boundary_bool"
                label = torch.tensor(_bool_id(aval), dtype=torch.long)
            else:
                task = "boundary_text"
                yid = _str2id(self.boundary2id, aval)
                label = torch.tensor(yid, dtype=torch.long)

        elif cat == "concurrency":
            if K is not None:
                # choice
                y = int(float(aval)) - 1 if _is_num(aval) else 0
                if y < 0 or y >= K:
                    self.logger.warning(f"[concurrency_choice] label {aval} out of 1..{K} -> clip")
                    y = max(0, min(y, K-1))
                task = "concurrency_choice"
                label = torch.tensor(y, dtype=torch.long)
                meta["choice_K"] = K
            elif answer_type == "boolean" or _is_bool(aval):
                task = "concurrency_bool"
                label = torch.tensor(_bool_id(aval), dtype=torch.long)
            elif answer_type == "numeric" or _is_num(aval):
                # 既包含整数也包含一位小数；语义是时长（秒）-> 回归
                task = "concurrency_reg"
                label = torch.tensor(self._safe_float(aval, 0.0), dtype=torch.float32)
            else:
                task = "concurrency_bool"
                label = torch.tensor(0, dtype=torch.long)
                self.logger.warning("[concurrency] unexpected answer; fallback to bool=0")

        elif cat == "ordering":
            if K is not None:
                # choice（K ≤ 5）
                y = int(float(aval)) - 1 if _is_num(aval) else 0
                if y < 0 or y >= K:
                    self.logger.warning(f"[ordering_choice] label {aval} out of 1..{K} -> clip")
                    y = max(0, min(y, K-1))
                task = "ordering_choice"
                label = torch.tensor(y, dtype=torch.long)
                meta["choice_K"] = K
            elif answer_type == "boolean" or _is_bool(aval):
                task = "ordering_bool"
                label = torch.tensor(_bool_id(aval), dtype=torch.long)
            elif answer_type == "string":
                task = "ordering_text"
                yid = _str2id(self.ordering2id, aval)
                label = torch.tensor(yid, dtype=torch.long)
            else:
                # numeric 但没解析到 K -> 当作 choice@5
                if _is_num(aval):
                    y = int(float(aval)) - 1
                    y = 0 if y < 0 else (4 if y > 4 else y)
                    task = "ordering_choice"
                    label = torch.tensor(y, dtype=torch.long)
                    meta["choice_K"] = 5
                else:
                    task = "ordering_bool"
                    label = torch.tensor(0, dtype=torch.long)
                    self.logger.warning("[ordering] unexpected answer; fallback to bool=0")

        elif cat == "motion":
            task = "motion_text"
            yid = _str2id(self.motion2id, aval)
            label = torch.tensor(yid, dtype=torch.long)

        elif cat == "extreme":
            if answer_type == "numeric" or _is_num(aval):
                task = "extreme_reg"
                label = torch.tensor(self._safe_float(aval, 0.0), dtype=torch.float32)
            elif answer_type == "boolean" or _is_bool(aval):
                task = "extreme_bool"
                label = torch.tensor(_bool_id(aval), dtype=torch.long)
            else:
                task = "extreme_text"
                yid = _str2id(self.extreme2id, aval)
                label = torch.tensor(yid, dtype=torch.long)

        elif cat == "phase_transition":
            if K is not None:
                y = int(float(aval)) - 1 if _is_num(aval) else 0
                if y < 0 or y >= K:
                    self.logger.warning(f"[phase_choice] label {aval} out of 1..{K} -> clip")
                    y = max(0, min(y, K-1))
                task = "phase_choice"
                label = torch.tensor(y, dtype=torch.long)
                meta["choice_K"] = K
            elif answer_type == "numeric" or _is_num(aval):
                task = "phase_reg"   # 允许负数，_safe_float 会保留符号
                label = torch.tensor(self._safe_float(aval, 0.0), dtype=torch.float32)
            elif answer_type == "boolean" or _is_bool(aval):
                task = "phase_bool"
                label = torch.tensor(_bool_id(aval), dtype=torch.long)
            else:
                task = "phase_text"
                yid = _str2id(self.phase2id, aval if aval is not None else self.UNK)
                label = torch.tensor(yid, dtype=torch.long)

        else:
            self.logger.warning(f"[temporal] category '{cat}' not handled; skipping sample")
            task = "skip"
            label = torch.tensor(0, dtype=torch.long)

        # --------- rationale（按类别键取） ----------
        rationale = self._pick_rationale_text(qa, category)

        meta.update({
            "vid": vid,
            "frames_requested": [int(f) for f in frames],
            "frames_anchors": anchors,
            "frames_used": used_frames,
            "substitutions": substitutions,   # 原->替
            "category": category,
            "has_rationale": bool(rationale),
        })

        return {
            "vid": vid,
            "question": question,
            "rationale": rationale,
            "visual_embeds": visual_embeds,
            "visual_frame_ids": visual_frame_ids,
            "task": task,
            "label": label,
            "meta": meta,
        }

    # --------- rationale 选择：与生成时的键一致 ----------
    def _rationale_key_for(self, category: str) -> Optional[str]:
        table = {
            "count":             "rationale_count_v1_en",
            "duration":          "rationale_duration_v1_en",
            "motion":            "rationale_motion_v1_en",
            "ordering":          "rationale_ordering_v1_en",
            "extreme":           "rationale_extreme_v1_en",
            "concurrency":       "rationale_concurrency_v1_en",
            "boundary":          "rationale_boundary_v1_en",
            "phase_transition":  "rationale_phase_transition_v1_en",
        }
        return table.get(category)

    def _pick_rationale_text(self, qa: Dict[str, Any], category: str) -> Optional[str]:
        key = self._rationale_key_for(category)
        if not key:
            return None
        val = qa.get(key, None)
        if isinstance(val, str):
            s = val.strip()
            return s if len(s) > 0 else None
        # 容错：若有人误放在 metadata
        md = qa.get("metadata", {})
        val2 = md.get(key, None) if isinstance(md, dict) else None
        if isinstance(val2, str):
            s = val2.strip()
            return s if len(s) > 0 else None
        return None
