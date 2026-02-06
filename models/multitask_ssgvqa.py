# models/multitask_ssgvqa.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.VisualBert_ssgqa import VisualBertModel, VisualBertConfig

class MultiTaskSSGVQA(nn.Module):
    """

    """
    def __init__(self,
                 tokenizer_vocab_size,
                 layers=6,
                 n_heads=8,
                 hidden_size=1024,
                 visual_embedding_dim=530,
                 answer_vocab_size=0,
                 count_bins=20,  # e.g., 0..19
                 choice_bins=5,
                 max_video_frames=64,
                 # ---- 新增：各分类 vocab 大小 ----
                 motion_classes: int = 8,
                 boundary_text_size: int = 0,
                 ordering_text_size: int = 0,
                 extreme_vocab_size: int = 0,
                 phase_vocab_size: int = 0,
                 use_localization_regression: bool = False,  # 你准备好 [cx,cy] 后再 True
                 # temporal_vocab_size: Optional[int] = None,
                 ):

        super().__init__()

        cfg = VisualBertConfig(
            vocab_size=tokenizer_vocab_size,
            visual_embedding_dim=visual_embedding_dim,
            num_hidden_layers=layers,
            num_attention_heads=n_heads,
            hidden_size=hidden_size,
        )
        setattr(cfg, "max_video_frames", max_video_frames)
        setattr(cfg, "use_temporal_position_embeddings", True)

        self.base = VisualBertModel(cfg)
        self.dropout = nn.Dropout(cfg.hidden_dropout_prob)
        H = hidden_size

        self.choice_bins = choice_bins
        self.heads = nn.ModuleDict({
            # "answer_cls": nn.Linear(H, answer_vocab_size),
            "count_cls":  nn.Linear(H, count_bins),
            "duration_reg": nn.Linear(H, 1),
            "ordering_bool": nn.Linear(H, 2),
            "ordering_choice": nn.Linear(H, choice_bins),
            "boundary_bool": nn.Linear(H, 2),
            "boundary_reg": nn.Linear(H, 1),
            "concurrency_bool": nn.Linear(H, 2),
            "concurrency_choice": nn.Linear(H, choice_bins),
            "concurrency_reg": nn.Linear(H, 1),
            "extreme_bool": nn.Linear(H, 2),
            "extreme_reg": nn.Linear(H, 1),
            "motion_text": nn.Linear(H, motion_classes),
            "phase_bool": nn.Linear(H, 2),
            "phase_reg": nn.Linear(H, 1),
            "phase_choice": nn.Linear(H, choice_bins),
        })
        # static answer head, only create when > 1 (consider static qa pairs)
        if answer_vocab_size and int(answer_vocab_size) > 1:
            self.heads["answer_cls"] = nn.Linear(H, int(answer_vocab_size))

        if ordering_text_size > 0:
            self.heads["ordering_text"] = nn.Linear(H, ordering_text_size)
        if boundary_text_size > 0:
            self.heads["boundary_text"] = nn.Linear(H, boundary_text_size)
        if extreme_vocab_size > 0:
            self.heads["extreme_text"] = nn.Linear(H, extreme_vocab_size)
        if phase_vocab_size > 0:
            self.heads["phase_text"] = nn.Linear(H, phase_vocab_size)

    def forward(self,
                text_inputs,                 # tokenizer batch dict
                visual=None,
                visual_embeds=None,               # (B, V, D)
                visual_frame_ids=None,       # (B, V)
                visual_attention_mask=None,
                # task="answer",
                task: str = "answer_cls",
                labels=None, 
                meta=None):
        
        # 准备视觉mask & ids
        if isinstance(visual, dict):
            visual_embeds = visual.get("visual_embeds", visual_embeds)
            visual_frame_ids = visual.get("visual_frame_ids", visual_frame_ids)
            visual_attention_mask = visual.get("visual_attention_mask", visual_attention_mask)
        
        B, V, _ = visual_embeds.shape
        device = visual_embeds.device
        visual_token_type_ids = torch.ones((B, V), dtype=torch.long, device=device)

        # 优先使用 collate 提供的 mask；否则按 “非全零向量” 推断
        if visual_attention_mask is None:
            visual_attention_mask = (visual_embeds.abs().sum(dim=-1) > 0).long()
        else:
            visual_attention_mask = visual_attention_mask.long()

        if visual_frame_ids is None:
            visual_frame_ids = torch.zeros((B, V), dtype=torch.long, device=device)

        model_inputs = {
            "input_ids": text_inputs["input_ids"],
            "token_type_ids": text_inputs["token_type_ids"],
            "attention_mask": text_inputs["attention_mask"],
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "visual_frame_ids": visual_frame_ids,
            "output_attentions": True # for visualization
        }

        outputs = self.base(**model_inputs)
        pooled = self.dropout(outputs.pooler_output)  # (B, H)

        if task == "answer_cls" and "answer_cls" not in self.heads:
            raise ValueError(
                "answer_cls head is disabled (no static QA). "
                "Run with --include_static True (and answer_vocab_size>1) to enable."
            )

        if task not in self.heads:
            raise ValueError(f"Unknown task: {task}")

        head = self.heads[task]
        if head is None:
            raise ValueError(f"Head for task '{task}' is not initialized")

        logits = head(pooled)

        # 1) choice 任务：对 >K 的 logit 做 -inf 掩码
        if task.endswith("_choice") and meta is not None and isinstance(meta, list):
            # 取每个样本的 K（默认 choice_bins）
            Ks = [ (m or {}).get("choice_K", self.choice_bins) for m in meta ]
            maxK = logits.size(-1)   # == self.choice_bins
            ar = torch.arange(maxK, device=logits.device).unsqueeze(0).expand(len(Ks), -1)
            Ktensor = torch.tensor(Ks, device=logits.device).unsqueeze(1)
            mask = ar >= Ktensor                    # True 表示需屏蔽
            logits = logits.masked_fill(mask, float("-inf"))

        # 2) 回归任务：输出可能是 (B,1) -> squeeze
        loss = None
        if task.endswith("_reg"):
            logits = logits.squeeze(-1)
            if labels is not None:
                loss = F.smooth_l1_loss(logits, labels)

        # 3) 其余分类任务：用 CE
        else:
            if labels is not None:
                loss = F.cross_entropy(logits, labels)

        # return {"logits": logits, "loss": loss}
        return {
            "logits": logits, 
            "loss": loss, 
            "attentions": outputs.attentions  # for visualization
        }