# utils/collate_temporal.py
import torch
from typing import List, Dict, Any, Callable
import random

# def make_collate_fn(tokenizer, pad_token_id=None) -> Callable:
def make_collate_fn(
    tokenizer,
    pad_token_id=None,
    *,
    use_rationale: bool = True,           # whether using rationale
    rationale_mode: str = "append",       # append/prefix: how to concate question and rationale
    rationale_tag: str = "Rationale:",    # explicit label
    rationale_dropout_p: float = 0.2,     # note: using 0.0 for validation
    max_length: int = 320,                # max length of text input
) -> Callable:
    """
    返回一个可直接喂给 DataLoader 的 collate_fn。
    - tokenizer: transformers tokenizer（需 BERT 类）
    - pad_token_id: 若为 None 则从 tokenizer.pad_token_id 取
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id or 0
        
    def _compose_text(question: str, rationale: str | None, training: bool) -> str:
        """把 rationale 按策略合进文本里。"""
        if not use_rationale or not rationale:
            return question
        # 训练随机丢弃
        if training and rationale_dropout_p > 0.0 and random.random() < rationale_dropout_p:
            return question
        # 注入方式
        if rationale_mode == "prefix":
            return f"{rationale_tag} {rationale} [SEP] {question}"
        # 缺省 append，兼容 tokenizer 一次性批量编码
        return f"{question} [SEP] {rationale_tag} {rationale}"

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1) 文本编码（带 rationale）
        # DataLoader 会在 model.train()/eval() 切换时重建？通常不会；
        # 这里用启发式判断：若 labels 中存在浮点（duration）我们也视为训练/验证均可。
        # 更稳妥做法：在构建 DataLoader 时分别传入“训练/验证用的 collate”（下面 main 里会这么做）。
        # 这里默认当作“训练模式使用的 collate”，不做状态推断。
        training_like = True

        questions_raw = [x["question"] for x in batch]
        rationales_raw = [x.get("rationale", None) for x in batch]

        merged_texts = [
            _compose_text(q, r, training_like)
            for q, r in zip(questions_raw, rationales_raw)
        ]

        enc = tokenizer(
            merged_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        text = {
            "input_ids": enc["input_ids"],
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"])),
            "attention_mask": enc["attention_mask"],
        }

        # 2. visual padding with mask
        Vmax = max(x["visual_embeds"].size(0) for x in batch)
        D = batch[0]["visual_embeds"].size(1)
        B = len(batch)

        # 特征仍用 0 pad（数值稳定）
        visual_embeds = torch.zeros(B, Vmax, D, dtype=torch.float32)

        # 帧 id 的 pad 设为 -1，避免与合法时间步 0 混淆
        visual_frame_ids = torch.full((B, Vmax), -1, dtype=torch.long)

        # 显式的视觉 attention mask：1=有效，0=pad
        visual_attention_mask = torch.zeros(B, Vmax, dtype=torch.long)

        for i, x in enumerate(batch):
            v = x["visual_embeds"]
            fids = x["visual_frame_ids"]
            V = v.size(0)
            visual_embeds[i, :V, :] = v
            visual_frame_ids[i, :V] = fids
            visual_attention_mask[i, :V] = 1

        # 3) 任务与标签（原样）
        tasks = [x["task"] for x in batch]
        labels = [x["label"] for x in batch]
        meta = [x.get("meta", {}) for x in batch]
        
        categories = [ (m or {}).get("category", "unknown") for m in meta ]

        return {
            "text": text,
            "visual": {
                "visual_embeds": visual_embeds,
                "visual_frame_ids": visual_frame_ids,          # pad=-1
                "visual_attention_mask": visual_attention_mask # 1/0
            },
            "tasks": tasks,
            "labels": labels,
            "meta": meta,
            "categories": categories,
        }

    return collate
