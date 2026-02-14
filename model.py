import torch
import torch.nn as nn
from transformers import AutoModel


class TrafficResBlock(nn.Module):
    """带残差连接的流量编码块"""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        return x + self.block(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, traffic_dim, text_dim, embed_dim=256, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.traffic_proj = nn.Linear(traffic_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, traffic_feat, text_feat):
        query = self.traffic_proj(traffic_feat).unsqueeze(1)
        key_value = self.text_proj(text_feat)
        attn_output, _ = self.mha(query, key_value, key_value)
        x = self.norm1(query + attn_output)
        output = self.norm2(x + self.ffn(x))
        return output.squeeze(1)


class AdvancedMultiModalNet(nn.Module):
    def __init__(self, config):
        super(AdvancedMultiModalNet, self).__init__()

        # --- 模态 A: 文本塔 ---
        self.bert = AutoModel.from_pretrained(config.TEXT_MODEL)
        n_unfreeze = getattr(config, 'UNFREEZE_BERT_LAYERS', 0)
        if n_unfreeze > 0:
            total_layers = self.bert.config.num_hidden_layers
            # BERT 用 encoder.layer，DistilBERT 用 transformer.layer
            layer_module = getattr(self.bert, 'encoder', None) or getattr(self.bert, 'transformer', None)
            for i, layer in enumerate(layer_module.layer):
                layer.requires_grad_(i >= total_layers - n_unfreeze)
        else:
            for param in self.bert.parameters():
                param.requires_grad = False
        text_hidden_dim = self.bert.config.hidden_size

        # --- 模态 B: 流量塔 (加深 + 残差) ---
        dims = [config.TRAFFIC_INPUT_DIM] + getattr(config, 'TRAFFIC_HIDDEN_DIMS', [64, 128, 256])
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.LeakyReLU(0.1),
            ])
            if i < len(dims) - 2:
                layers.append(TrafficResBlock(dims[i + 1]))
        self.traffic_encoder = nn.Sequential(*layers)
        traffic_dim = dims[-1]

        embed_dim = getattr(config, 'FUSION_EMBED_DIM', 256)
        num_heads = getattr(config, 'FUSION_HEADS', 8)
        self.fusion = CrossAttentionFusion(
            traffic_dim=traffic_dim, text_dim=text_hidden_dim,
            embed_dim=embed_dim, num_heads=num_heads
        )

        dropout = getattr(config, 'DROPOUT', 0.4)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, config.NUM_CLASSES)
        )

    def forward(self, traffic_data, input_ids, attention_mask):
        text_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_seq_feat = text_out.last_hidden_state
        traffic_feat = self.traffic_encoder(traffic_data)
        fused_feat = self.fusion(traffic_feat, text_seq_feat)
        logits = self.classifier(fused_feat)
        return logits