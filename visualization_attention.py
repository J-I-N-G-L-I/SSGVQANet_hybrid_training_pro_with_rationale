# -*- coding: utf-8 -*-
import os
import json
import argparse
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import textwrap  
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer
from tqdm import tqdm

# 引入你项目中的模块
from utils.mixed_datasets import STSGTemporalQADataset, SSGVQAFrameDataset
from utils.collate_temporal import make_collate_fn
from models.multitask_ssgvqa import MultiTaskSSGVQA

# --- 原始图像参数 ---
ORIG_WIDTH = 860
ORIG_HEIGHT = 480

def parse_args():
    p = argparse.ArgumentParser()
    # 路径参数
    p.add_argument("--stsg_qa_root", default=None,
                   help="STSG temporal QA root (used in temporal mode)")
    p.add_argument("--ssg_qa_root", default=None,
                   help="SSG-VQA qa_txt root (used in static mode)")
    p.add_argument("--feature_root", required=True)
    p.add_argument("--raw_image_root", required=True,
                   help="原始图片根目录，结构应为 root/VIDxx/xxxxxx.png")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--vocabs_json", default=None,
                   help="fixed vocabs json (only needed in temporal mode)")
    p.add_argument("--output_dir", default="./vis_results_v2")
    
    # 模型参数
    p.add_argument("--count_bins", type=int, default=20)
    p.add_argument("--max_frames", type=int, default=16)
    p.add_argument("--tokenizer_ver", default="v2")
    p.add_argument("--text_max_len", type=int, default=320)
    
    # 模式选择：temporal / static
    p.add_argument("--mode", default="temporal",
                   choices=["temporal", "static"],
                   help="temporal: STSGTemporalQADataset; static: SSGVQAFrameDataset (answer head)")
    # 对 static 模式：静态多帧窗口大小
    p.add_argument("--static_frames_per_sample", type=int, default=12,
                   help="静态 QA 使用的多帧窗口大小，仅在 --mode static 时生效")
    p.add_argument("--static_use_multi_frames", action="store_true",
                   help="仅在 --mode static 时生效。"
                        "若设置，则使用多帧窗口 (frames_per_sample)；"
                        "不设置则退回到原始单帧行为。")

    # 可视化控制
    p.add_argument("--temporal_videos", nargs="+", required=True,
                   help="要可视化的 VID 列表（temporal/static 模式都复用这个参数）")
    p.add_argument("--vis_count", type=int, default=20,
                   help="每个类别可视化多少张")
    p.add_argument("--target_category", default=None,
                   help="只可视化特定类别，例如: motion, extreme, boundary, answer。 不填则全部")
    p.add_argument("--vis_mode", default="global",
                   choices=["global", "local"], 
                   help="global: 跨帧归一化(突显关键帧); local: 帧内归一化(每帧都画出最关注的物体)")
    
    return p.parse_args()

def select_tokenizer(tokenizer_ver: str):
    if tokenizer_ver in ("v2", "v3"):
        return BertTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", do_lower_case=(tokenizer_ver == "v3"))
    else:
        return AutoTokenizer.from_pretrained("bert-base-uncased")

# --- 模型构建辅助函数 ---
def infer_heads_from_state(state):
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

def build_model_from_ckpt(ckpt_path, tokenizer_vocab_size, max_video_frames, default_count_bins):
    print(f"[Model] Loading checkpoint from {ckpt_path}...")
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
    model.load_state_dict(state, strict=False)
    return model

def visualize_single_sample(idx, batch, model, device, args, global_idx: int):
    # 1. 解包数据
    text = {k: v[idx:idx+1].to(device) for k, v in batch["text"].items()}
    vis  = {k: v[idx:idx+1].to(device) for k, v in batch["visual"].items()}
    
    task = batch["tasks"][idx]
    label = batch["labels"][idx]
    meta = batch["meta"][idx]
    vid = meta["vid"]
    
    # [新增] 获取原始问题文本
    # 注意：这需要 main 函数里的 collate_wrapper 支持
    raw_question = batch.get("questions_raw", [""])[idx]
    raw_answer   = batch.get("answers_raw",   [""])[idx]
    qa_source    = batch.get("sources_raw",   [""])[idx]

    # 2. 推理
    with torch.no_grad():
        out = model(text, visual=vis, task=task)
    
    if "attentions" not in out:
        print("Error: Model output does not contain 'attentions'. Please modify multitask_ssgvqa.py.")
        return

    # 3. 提取 Attention
    # 取最后一层: (B, Heads, Seq, Seq) -> 平均 Heads -> (B, Seq, Seq)
    last_layer_attn = out["attentions"][-1].mean(dim=1) 
    
    # 计算 [CLS] 对 Visual Tokens 的关注
    text_len = text["input_ids"].shape[1]
    # attn_map: (Visual_Len,)
    attn_map = last_layer_attn[0, 0, text_len:].cpu().numpy()
    
    # 4. 准备视觉信息
    vis_embeds = vis["visual_embeds"][0].cpu() # (Visual_Len, 530)
    vis_frame_ids = vis["visual_frame_ids"][0].cpu().numpy() # (Visual_Len,)
    
    # 归一化坐标 [x1, y1, x2, y2]
    norm_bboxes = vis_embeds[:, 14:18].numpy()
    
    # 真实帧列表
    frames_used = meta["frames_used"]         # 实际读取的帧 (Fallback后)
    frames_requested = meta["frames_requested"] # 原始请求的帧
    substitutions = meta.get("substitutions", []) # 替换记录
    
    # --- 归一化策略 ---
    # 如果是 global 模式，我们计算全局最大最小值
    global_min, global_max = attn_map.min(), attn_map.max()
    
    # 创建保存目录
    # sample_dir = os.path.join(args.output_dir, f"{task}_{vid}_{idx}")

    sample_dir = os.path.join(
        args.output_dir,
        f"{task}_{vid}_qid{global_idx:05d}"   # 例如 answer_cls_VID22_qid00037
    )
    os.makedirs(sample_dir, exist_ok=True)

    # os.makedirs(sample_dir, exist_ok=True)
    
    unique_rel_ids = sorted(list(set(vis_frame_ids)))
    
    for rel_fidx in unique_rel_ids:
        if rel_fidx == -1: continue # Padding
        if rel_fidx >= len(frames_used): continue

        # DEBUG: 看看这一帧的 token / attention / bbox 情况
        token_indices = np.where(vis_frame_ids == rel_fidx)[0]
        if len(token_indices) == 0:
            print(f"[DEBUG] vid={vid} rel_fidx={rel_fidx}: no tokens")
            continue

        local_attns = attn_map[token_indices]
        print(f"[DEBUG] vid={vid} rel_fidx={rel_fidx}: "
              f"local_attn_min={local_attns.min():.4e}, "
              f"local_attn_max={local_attns.max():.4e}, "
              f"global_min={global_min:.4e}, global_max={global_max:.4e}, "
              f"#tokens={len(token_indices)}")

        # 再看一下 bbox 有没有全 0
        some_bn = norm_bboxes[token_indices[:5]]  # 取前5个
        print(f"[DEBUG]   sample bboxes (first few): {some_bn}")
        
        # 获取真实帧号（这是模型实际看到的图像）
        actual_frame_idx = frames_used[rel_fidx]
        
        # 检查是否是替换帧
        is_fallback = False
        orig_frame_idx = actual_frame_idx
        
        req_frame_idx = frames_requested[rel_fidx]
        if req_frame_idx != actual_frame_idx:
            is_fallback = True
            orig_frame_idx = req_frame_idx

        # 读取图片
        img_name = f"{actual_frame_idx:06d}.png"
        img_path = os.path.join(args.raw_image_root, vid, img_name)
        
        if not os.path.exists(img_path):
            # 尝试找jpg或者其他格式
            img_path = os.path.join(args.raw_image_root, vid, f"{actual_frame_idx:06d}.jpg")
            if not os.path.exists(img_path):
                continue
        
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.resize(img, (ORIG_WIDTH, ORIG_HEIGHT))
        overlay = img.copy()
        
        # 找到当前帧的所有 Token
        token_indices = np.where(vis_frame_ids == rel_fidx)[0]
        if len(token_indices) == 0: continue
        
        # 获取这些 token 的分数
        local_attns = attn_map[token_indices]
        
        # --- 归一化 ---
        if args.vis_mode == "global":
            norm_scores = (local_attns - global_min) / (global_max - global_min + 1e-8)
            draw_threshold = 0.2
        else:
            l_min, l_max = local_attns.min(), local_attns.max()
            norm_scores = (local_attns - l_min) / (l_max - l_min + 1e-8)
            draw_threshold = 0.4 
            
        # 画框逻辑
        sorted_local_idx = np.argsort(norm_scores)
        
        for s_idx in sorted_local_idx:
            score = norm_scores[s_idx]
            
            if score < draw_threshold: continue
            
            g_idx = token_indices[s_idx]
            bn = norm_bboxes[g_idx] # [x1,y1,x2,y2]
            
            if np.sum(bn) < 1e-6: continue
            
            x1 = int(bn[0] * ORIG_WIDTH)
            y1 = int(bn[1] * ORIG_HEIGHT)
            x2 = int(bn[2] * ORIG_WIDTH)
            y2 = int(bn[3] * ORIG_HEIGHT)
            
            heatmap_color = cv2.applyColorMap(np.array([[int(score * 255)]], dtype=np.uint8), cv2.COLORMAP_JET)
            color = heatmap_color[0][0].tolist() # BGR
            
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
        # 融合 Mask
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # # --- 绘制文字信息 ---
        # # 1. 帧号信息
        # info_text = f"Frame: {actual_frame_idx}"
        # if is_fallback:
        #     info_text += f" (Fallback from {orig_frame_idx})"
        #     text_color = (0, 255, 255) 
        # else:
        #     text_color = (0, 255, 0)
            
        # cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # # 2. 任务信息
        # cv2.putText(img, f"Task: {task}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # # 3. 最大权重提示
        # max_s = local_attns.max()
        # cv2.putText(img, f"Max Attn: {max_s:.4f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # # 4. [新增] 绘制问题文本 (Question)
        # if raw_question:
        #     # 参数配置
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     font_scale = 0.5  # 较小字体
        #     font_thickness = 1
        #     line_spacing = 20
        #     start_y = 120  # 从 Max Attn 下方开始
            
        #     # 使用 textwrap 自动换行
        #     # 860宽 / (0.5 scale * 20px/char 估算) ≈ 86个字符，保守设为 70
        #     wrapper = textwrap.TextWrapper(width=70) 
        #     wrapped_lines = wrapper.wrap(text=f"Q: {raw_question}")
            
        #     # 绘制每一行
        #     for i, line in enumerate(wrapped_lines):
        #         y_pos = start_y + i * line_spacing
        #         # 先画第一行 "Q:" 的前缀高亮（可选），这里统一用白色
        #         line_color = (255, 255, 255) # 白色
        #         if i == 0:
        #             # 可以把 Q: 单独画成黄色，但为了简单直接画整行
        #             # 如果想 Q: 黄色，内容白色，需要拆分，这里直接整行
        #             pass
                
        #         # 绘制描边（黑色背景），增加对比度
        #         cv2.putText(img, line, (10, y_pos), font, font_scale, (0, 0, 0), font_thickness + 2)
        #         # 绘制文字
        #         cv2.putText(img, line, (10, y_pos), font, font_scale, line_color, font_thickness)


        # === 文本叠加区域：Question / Answer / TXT 路径 ===
        # 不需要时可以整块注释掉这一段
        if raw_question or raw_answer or qa_source:
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45      # 小字
            font_thickness = 1
            line_spacing = 18      # 行距
            margin_x, margin_y = 10, 20

            lines = []

            # 1) 头部：VID + 帧号 + Fallback 提示
            header = f"{vid}  Frame {actual_frame_idx:06d} QID {global_idx:05d}"
            if is_fallback:
                header += f" (from {orig_frame_idx:06d})"
            lines.append(header)

            # 2) Question：自动换行
            if raw_question:
                q_prefix = "Q: "
                wrapper_q = textwrap.TextWrapper(width=70)  # 控制每行长度
                # 为了不丢掉 Q: 前缀，先拼再 wrap
                for i, line in enumerate(wrapper_q.wrap(text=q_prefix + raw_question)):
                    lines.append(line)

            # 3) Answer：自动换行
            if raw_answer:
                a_prefix = "A: "
                wrapper_a = textwrap.TextWrapper(width=70)
                for i, line in enumerate(wrapper_a.wrap(text=a_prefix + raw_answer)):
                    lines.append(line)

            # 4) TXT 源文件（路径可以稍微裁剪，只保留后半段）
            if qa_source:
                # 只保留末尾一段，防止太长（例如 ".../ssg-qa/VID02/102.txt"）
                short_src = qa_source
                for token in ["qa_txt", "ssg-qa"]:
                    if token in qa_source:
                        short_src = qa_source.split(token, 1)[-1].lstrip("/\\")
                        break
                lines.append(f"src: {short_src}")

            # 实际绘制：每行先画黑边，再画白字，增强对比度
            for i, text_line in enumerate(lines):
                y = margin_y + i * line_spacing

                # 先画“描边”：黑色，粗一点
                cv2.putText(
                    img,
                    text_line,
                    (margin_x, y),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness + 2,
                    cv2.LINE_AA,
                )
                # 再画白色文字
                cv2.putText(
                    img,
                    text_line,
                    (margin_x, y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA,
                )
        # === 文本叠加区域结束 ===

        # 保存
        save_name = f"frame_{rel_fidx:02d}_{actual_frame_idx:06d}.jpg"
        cv2.imwrite(os.path.join(sample_dir, save_name), img)

    print(f"Saved visualization for {vid} to {sample_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Vocabs
    # with open(args.vocabs_json, "r", encoding="utf-8") as f:
    #     fixed_vocabs = json.load(f)
    
    # 2. Tokenizer & Model
    tokenizer = select_tokenizer(args.tokenizer_ver)
    model = build_model_from_ckpt(
        args.checkpoint,
        tokenizer_vocab_size=len(tokenizer),
        max_video_frames=args.max_frames,
        default_count_bins=args.count_bins
    ).to(device)
    model.eval()
    
    if args.mode == "temporal":
        if not args.stsg_qa_root:
            raise ValueError("--mode temporal 需要提供 --stsg_qa_root")
        if not args.vocabs_json:
            raise ValueError("--mode temporal 需要提供 --vocabs_json")
        
        with open(args.vocabs_json, "r", encoding="utf-8") as f:
            fixed_vocabs = json.load(f)
        
        allowed_cats = [
            "count", "duration", "ordering", "extreme",
            "boundary", "motion", "concurrency", "phase_transition"
        ]
        
        ds = STSGTemporalQADataset(
            qa_root=args.stsg_qa_root,
            folder_head=args.feature_root,
            vids=args.temporal_videos,
            allowed_categories=allowed_cats,
            max_frames=args.max_frames,
            fps=1.0,
            fixed_vocabs=fixed_vocabs,
            logger=None 
        )
    else:
        # static 模式：使用 SSGVQAFrameDataset + 多帧窗口
        if not args.ssg_qa_root:
            raise ValueError("--mode static 需要提供 --ssg_qa_root")
        
        ds = SSGVQAFrameDataset(
            qa_root=args.ssg_qa_root,
            folder_head=args.feature_root,
            seq=args.temporal_videos,          # 这里直接复用 temporal_videos 作为 VID 列表
            logger=None,
            # multi_frames=True,                 # 启用多帧静态模式
            multi_frames=args.static_use_multi_frames,
            frames_per_sample=args.static_frames_per_sample
        )
    
    # # [新增] 定义 Wrapper 以保留原始问题文本
    # # 使用原始 Collate 生成 Tensor
    # original_collate_fn = make_collate_fn(tokenizer, max_length=args.text_max_len, use_rationale=True)
    
    # def collate_wrapper(batch):
    #     # 先调用原始 collate 获取 tensor 数据
    #     out = original_collate_fn(batch)
    #     # 手动附加原始 question 文本列表
    #     out["questions_raw"] = [x["qa"]["question"] if "qa" in x else x.get("question", "") for x in batch]
    #     # 注意：STSGTemporalQADataset 的 __getitem__ 返回的字典里直接有 "question" 字段
    #     # 查看 mixed_datasets.py: return { ..., "question": question, ... }
    #     # 所以应该是：
    #     out["questions_raw"] = [x["question"] for x in batch]
    #     return out
        # 3. Collate：static 模式不需要 rationale，temporal 保持原样
    if args.mode == "static":
        original_collate_fn = make_collate_fn(
            tokenizer,
            max_length=args.text_max_len,
            use_rationale=False,       # 静态 QA 没有 rationale
        )
    else:
        original_collate_fn = make_collate_fn(
            tokenizer,
            max_length=args.text_max_len,
            use_rationale=True,        # 仍然把 rationale 拼到文本里
        )
    
    # def collate_wrapper(batch):
    #     out = original_collate_fn(batch)
    #     # 所有 Dataset 的 __getitem__ 都有 "question" 字段
    #     out["questions_raw"] = [x["question"] for x in batch]
    #     return out

    def collate_wrapper(batch):
        """
        在原始 collate 的基础上，额外把：
        - 原始 question 文本
        - 原始 answer 文本（如果有）
        - 对应的 txt 文件路径（如果有）
        放到 batch 里，方便可视化阶段使用。
        """
        out = original_collate_fn(batch)

        # 1) 问题文本（所有 Dataset 已经有 "question" 字段）
        out["questions_raw"] = [x.get("question", "") for x in batch]

        # 2) 答案文本
        #   - SSGVQAFrameDataset 一般会有诸如 "answer", "answer_text" 之类字段
        #   - 这里用 .get 链式兜底，不会因为键名不同而直接报错，最多就是变成空字符串
        out["answers_raw"] = [
            x.get("answer_text", x.get("answer", ""))
            for x in batch
        ]

        # 3) 源 txt 文件路径
        #   - 你可以在 SSGVQAFrameDataset 的 __getitem__ 里，给每个样本加一个字段比如 "txt_path"
        #   - 这里尝试从几种可能的命名中取值（qa_path / txt_path / source_path 等）
        out["sources_raw"] = [
            x.get("qa_path",
                x.get("txt_path",
                        x.get("source_path", "")))
            for x in batch
        ]

        return out
    
    # 手动构建 Loader
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_wrapper)
    
    print(f"Start visualizing... Filter category: {args.target_category}")
    
    # processed_count = 0
    # for i, batch in enumerate(tqdm(loader)):
    #     if processed_count >= args.vis_count:
    #         break
            
    #     # 获取该样本的元数据
    #     meta = batch["meta"][0] # batch_size=1
    #     cat = meta["category"]
        
    #     # 过滤逻辑
    #     if args.target_category and args.target_category not in cat:
    #         continue
        
    #     # 执行可视化
    #     visualize_single_sample(0, batch, model, device, args)
    #     processed_count += 1
    processed_count = 0
    for i, batch in enumerate(tqdm(loader)):
        if processed_count >= args.vis_count:
            break

        meta = batch["meta"][0]
        cat = meta["category"]

        if args.target_category and args.target_category not in cat:
            continue

        # i 就是全局样本索引，把它当作 “问题 ID”
        visualize_single_sample(0, batch, model, device, args, global_idx=i)
        processed_count += 1

if __name__ == "__main__":
    main()