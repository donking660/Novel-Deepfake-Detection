"""
Region Feature Extractor
区域特征提取模块: 通过掩膜池化从空间特征图中提取人脸/边界/背景三个区域的特征向量

数据流:
  spatial_tokens (B, 256, 1024) + region_masks (B, 3, 16, 16)
  → 掩膜 reshape 为 (B, 3, 256)
  → 对每个区域做 masked average pooling
  → face_vec (B, D), boundary_vec (B, D), bg_vec (B, D)

注意:
  spatial_tokens 来自 CLIP ViT 最后一层的 patch token (未经 ln_post),
  因此本模块先做 LayerNorm 再池化, 以提高数值稳定性.
"""

import torch
import torch.nn as nn


class RegionFeatureExtractor(nn.Module):
    """
    区域特征提取器

    将 CLIP ViT 最后一层的空间特征 (patch tokens) 按三个区域掩膜做加权平均池化,
    生成 face / boundary / background 三个区域特征向量.

    Args:
        feat_dim: 特征维度 (CLIP ViT-L/14 = 1024)
        grid_size: 特征图空间分辨率 (224/14 = 16)
    """

    def __init__(self, feat_dim: int = 1024, grid_size: int = 16):
        super().__init__()
        self.feat_dim = feat_dim
        self.grid_size = grid_size
        self.num_patches = grid_size * grid_size  # 256

        # 对空间特征做 LayerNorm (CLIP 最后一层 patch tokens 未经 ln_post)
        self.ln = nn.LayerNorm(feat_dim)

    def forward(
        self,
        spatial_tokens: torch.Tensor,
        region_masks: torch.Tensor,
    ) -> tuple:
        """
        前向传播

        Args:
            spatial_tokens: CLIP 最后一层 patch token 序列
                形状 (B, num_patches, feat_dim) = (B, 256, 1024)
            region_masks: 三区域掩膜
                形状 (B, 3, grid_size, grid_size) = (B, 3, 16, 16)
                channel 0: face, channel 1: boundary, channel 2: background
                约束: sum over channels ≈ 1

        Returns:
            face_vec:     (B, feat_dim) — 人脸区域特征向量
            boundary_vec: (B, feat_dim) — 边界区域特征向量
            bg_vec:       (B, feat_dim) — 背景区域特征向量
        """
        B = spatial_tokens.shape[0]

        # 1. LayerNorm
        spatial_tokens = self.ln(spatial_tokens.float())  # (B, 256, D)

        # 2. 将掩膜展平: (B, 3, 16, 16) -> (B, 3, 256)
        masks_flat = region_masks.reshape(B, 3, self.num_patches).float()

        # 3. 分离三个区域掩膜: 各 (B, 256)
        m_face = masks_flat[:, 0, :]      # (B, 256)
        m_boundary = masks_flat[:, 1, :]  # (B, 256)
        m_bg = masks_flat[:, 2, :]        # (B, 256)

        # 4. Masked average pooling
        # spatial_tokens: (B, 256, D), mask: (B, 256) -> (B, 256, 1)
        face_vec = self._masked_avg_pool(spatial_tokens, m_face)          # (B, D)
        boundary_vec = self._masked_avg_pool(spatial_tokens, m_boundary)  # (B, D)
        bg_vec = self._masked_avg_pool(spatial_tokens, m_bg)              # (B, D)

        return face_vec, boundary_vec, bg_vec

    @staticmethod
    def _masked_avg_pool(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        掩膜加权平均池化

        Args:
            tokens: (B, N, D) 特征序列
            mask: (B, N) 权重掩膜 (非负, 不必归一化)

        Returns:
            pooled: (B, D) 池化后的特征向量
        """
        # mask: (B, N) -> (B, N, 1)
        mask_expanded = mask.unsqueeze(-1)  # (B, N, 1)

        # 加权求和
        weighted_sum = (tokens * mask_expanded).sum(dim=1)  # (B, D)

        # 归一化 (避免除零)
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1e-8)  # (B, 1)

        return weighted_sum / mask_sum  # (B, D)
