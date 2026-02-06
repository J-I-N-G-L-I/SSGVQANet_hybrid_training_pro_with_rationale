"""
Project: Advancing Surgical VQA with Scene Graph Knowledge
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
"""
import torch
from torch import nn
from models.VisualBert_ssgqa import VisualBertModel, VisualBertConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import AutoModel

"""
VisualBert Classification Model
"""


# class VisualBertClassification(nn.Module):
#     """
#     VisualBert Classification Model
#     vocab_size    = tokenizer length
#     encoder_layer = 6
#     n_heads       = 8
#     num_class     = number of class in dataset
#     """
#
#     def __init__(self, vocab_size, layers, n_heads, num_class=10):
#         super(VisualBertClassification, self).__init__()
#         VBconfig = VisualBertConfig(
#             vocab_size=vocab_size,
#             visual_embedding_dim=530,
#             num_hidden_layers=layers,
#             num_attention_heads=n_heads,
#             hidden_size=1024,
#         )
#         self.VisualBertEncoder = VisualBertModel(VBconfig)
#         self.classifier = nn.Linear(VBconfig.hidden_size, num_class)
#
#     def forward(self, inputs, visual_embeds):
#         # prepare visual embedding
#         visual_token_type_ids = torch.ones(
#             visual_embeds.shape[:-1], dtype=torch.long
#         ).to(device)
#         visual_attention_mask = torch.ones(
#             visual_embeds.shape[:-1], dtype=torch.float
#         ).to(device)
#         # append visual features to text
#         inputs.update(
#             {
#                 "visual_embeds": visual_embeds,
#                 # "visual_token_type_ids": visual_token_type_ids,
#                 # "visual_attention_mask": visual_attention_mask,
#                 "output_attentions": True,
#             }
#         )
#
#         inputs["input_ids"] = inputs["input_ids"].to(device)
#         inputs["token_type_ids"] = inputs["token_type_ids"].to(device)
#         inputs["attention_mask"] = inputs["attention_mask"].to(device)
#         # inputs['visual_token_type_ids'] = inputs['visual_token_type_ids'].to(device)
#         # inputs['visual_attention_mask'] = inputs['visual_attention_mask'].to(device)
#
#         # Encoder output
#         outputs = self.VisualBertEncoder(**inputs)
#         # classification layer
#         outputs = self.classifier(outputs["pooler_output"])
#         return outputs

class VisualBertClassification(nn.Module):
    """
    VisualBert Classification Model
    vocab_size    = tokenizer length
    encoder_layer = 6
    n_heads       = 8
    num_class     = number of class in dataset
    """

    def __init__(self, vocab_size, layers, n_heads, num_class=10):
        super(VisualBertClassification, self).__init__()
        VBconfig = VisualBertConfig(
            vocab_size=vocab_size,
            visual_embedding_dim=530,
            num_hidden_layers=layers,
            num_attention_heads=n_heads,
            hidden_size=1024,
        )
        # [PATCH C-1] 配置里设默认上限（也可在外面传入后 setattr）
        setattr(VBconfig, "max_video_frames", 64)
        setattr(VBconfig, "use_temporal_position_embeddings", True)

        self.VisualBertEncoder = VisualBertModel(VBconfig)
        self.classifier = nn.Linear(VBconfig.hidden_size, num_class)

    def forward(self, inputs, visual_embeds, visual_frame_ids=None):
        """
        inputs: tokenizer 的输出字典（包含 input_ids/token_type_ids/attention_mask）
        visual_embeds: (B, V, D) 视觉token序列（多帧已flatten）
        visual_frame_ids: (B, V) 每个视觉token属于第几帧(0..F-1)，单帧可传 None 或 0
        """
        # ---- 准备视觉 mask ----
        B, V, _ = visual_embeds.shape
        visual_embeds = visual_embeds.to(device)

        if visual_frame_ids is None:
            visual_frame_ids = torch.zeros((B, V), dtype=torch.long, device=visual_embeds.device)
        else:
            visual_frame_ids = visual_frame_ids.to(device).long()

        visual_token_type_ids = torch.ones((B, V), dtype=torch.long, device=visual_embeds.device)
        # 有效token为1，padding为0（若你的collate已pad，下面这一行就能正确反映）
        visual_attention_mask = (visual_embeds.abs().sum(dim=-1) > 0).float()

        # ---- 把视觉字段拼入 inputs ----
        model_inputs = {
            "input_ids": inputs["input_ids"].to(device),
            "token_type_ids": inputs["token_type_ids"].to(device),
            "attention_mask": inputs["attention_mask"].to(device),
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "visual_frame_ids": visual_frame_ids,
            "output_attentions": inputs.get("output_attentions", False),
        }

        outputs = self.VisualBertEncoder(**model_inputs)
        logits = self.classifier(outputs["pooler_output"])
        return logits
