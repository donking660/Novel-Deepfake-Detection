"""
Hierarchical Decoupling Module (HDM)
层次化解耦模块: 将 3 个区域向量通过组合扩展为 7 个层次化特征向量

借鉴 DeMo 的 HDM 设计，但将 RGB/NI/TI 三模态替换为 face/boundary/background 三区域:

  单区域 (3个):   face, boundary, background
  双区域 (3个):   face+boundary, face+background, boundary+background
  全区域 (1个):   face+boundary+background

每个组合使用:
  - 一个可学习 query token (nn.Parameter)
  - 一个 MultiheadAttention 模块
  - K/V = [CLS token, 对应区域向量] 的拼接序列

Cross-attention: query = learnable_token, K = V = [cls, region_vectors...]
→ 输出一个聚合了全局信息和区域信息的特征向量

数据流:
  cls_token (B, D) + face_vec (B, D) + boundary_vec (B, D) + bg_vec (B, D)
  → 7 个 cross-attention
  → 7 个特征向量, 各 (B, D)
"""

import torch
import torch.nn as nn


class HierarchicalDecouplingModule(nn.Module):
    """
    层次化解耦模块 (HDM)

    Args:
        feat_dim: 特征维度 (默认 1024, CLIP ViT-L/14)
        num_heads: MultiheadAttention 的头数 (默认 16, head_dim=64)
        dropout: Attention dropout 概率
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        scale = feat_dim ** -0.5

        # =====================================================================
        # 7 个可学习 query token (形状与 DeMo 一致: (1, 1, D))
        # =====================================================================
        # 单区域
        self.face_token = nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        self.boundary_token = nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        self.bg_token = nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        # 双区域
        self.face_boundary_token = nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        self.face_bg_token = nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        self.boundary_bg_token = nn.Parameter(scale * torch.randn(1, 1, feat_dim))
        # 全区域
        self.all_token = nn.Parameter(scale * torch.randn(1, 1, feat_dim))

        # =====================================================================
        # 7 个 MultiheadAttention (query=learnable_token, K=V=region_sequence)
        # nn.MultiheadAttention 期望输入格式 (L, B, D)
        # =====================================================================
        # 单区域
        self.attn_face = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.attn_boundary = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.attn_bg = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        # 双区域
        self.attn_face_boundary = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.attn_face_bg = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        self.attn_boundary_bg = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )
        # 全区域
        self.attn_all = nn.MultiheadAttention(
            embed_dim=feat_dim, num_heads=num_heads, dropout=dropout, batch_first=False
        )

    def forward(
        self,
        cls_token: torch.Tensor,
        face_vec: torch.Tensor,
        boundary_vec: torch.Tensor,
        bg_vec: torch.Tensor,
    ) -> tuple:
        """
        前向传播

        Args:
            cls_token:    (B, D) — CLIP CLS token (全局特征)
            face_vec:     (B, D) — 人脸区域特征向量
            boundary_vec: (B, D) — 边界区域特征向量
            bg_vec:       (B, D) — 背景区域特征向量

        Returns:
            7 个特征向量的元组, 各 (B, D):
            (face_feat, boundary_feat, bg_feat,
             face_boundary_feat, face_bg_feat, boundary_bg_feat,
             all_feat)
        """
        B = cls_token.shape[0]

        # 将向量转为 (1, B, D) 格式用于 nn.MultiheadAttention
        cls_seq = cls_token.unsqueeze(0)       # (1, B, D)
        face_seq = face_vec.unsqueeze(0)       # (1, B, D)
        boundary_seq = boundary_vec.unsqueeze(0)  # (1, B, D)
        bg_seq = bg_vec.unsqueeze(0)           # (1, B, D)

        # =================================================================
        # 构建每个组合的 K/V 序列 (DeMo 风格: 全局 + 区域)
        # =================================================================
        # 单区域: KV = [cls, region_i]
        kv_face = torch.cat([cls_seq, face_seq], dim=0)           # (2, B, D)
        kv_boundary = torch.cat([cls_seq, boundary_seq], dim=0)   # (2, B, D)
        kv_bg = torch.cat([cls_seq, bg_seq], dim=0)               # (2, B, D)

        # 双区域: KV = [cls, region_i, region_j]
        kv_face_boundary = torch.cat([cls_seq, face_seq, boundary_seq], dim=0)  # (3, B, D)
        kv_face_bg = torch.cat([cls_seq, face_seq, bg_seq], dim=0)              # (3, B, D)
        kv_boundary_bg = torch.cat([cls_seq, boundary_seq, bg_seq], dim=0)      # (3, B, D)

        # 全区域: KV = [cls, face, boundary, bg]
        kv_all = torch.cat([cls_seq, face_seq, boundary_seq, bg_seq], dim=0)  # (4, B, D)

        # =================================================================
        # 扩展 learnable token: (1, 1, D) -> (1, B, D)
        # =================================================================
        q_face = self.face_token.expand(-1, B, -1)
        q_boundary = self.boundary_token.expand(-1, B, -1)
        q_bg = self.bg_token.expand(-1, B, -1)
        q_face_boundary = self.face_boundary_token.expand(-1, B, -1)
        q_face_bg = self.face_bg_token.expand(-1, B, -1)
        q_boundary_bg = self.boundary_bg_token.expand(-1, B, -1)
        q_all = self.all_token.expand(-1, B, -1)

        # =================================================================
        # Cross-Attention: query=learnable_token, key=value=region_sequence
        # 输出 (1, B, D) → squeeze → (B, D)
        # =================================================================
        # 单区域
        face_feat = self.attn_face(q_face, kv_face, kv_face)[0].squeeze(0)              # (B, D)
        boundary_feat = self.attn_boundary(q_boundary, kv_boundary, kv_boundary)[0].squeeze(0)
        bg_feat = self.attn_bg(q_bg, kv_bg, kv_bg)[0].squeeze(0)

        # 双区域
        face_boundary_feat = self.attn_face_boundary(
            q_face_boundary, kv_face_boundary, kv_face_boundary
        )[0].squeeze(0)
        face_bg_feat = self.attn_face_bg(
            q_face_bg, kv_face_bg, kv_face_bg
        )[0].squeeze(0)
        boundary_bg_feat = self.attn_boundary_bg(
            q_boundary_bg, kv_boundary_bg, kv_boundary_bg
        )[0].squeeze(0)

        # 全区域
        all_feat = self.attn_all(q_all, kv_all, kv_all)[0].squeeze(0)

        return (
            face_feat, boundary_feat, bg_feat,
            face_boundary_feat, face_bg_feat, boundary_bg_feat,
            all_feat,
        )
