"""
Region-Decoupled MoE Detector (RegionDeMo)
区域解耦混合专家检测器 — 消融实验 3

在消融实验 2 (Region-Aware Adapter) 基础上增加:
  1. 从 CLIP 最后一层提取空间特征图 (patch tokens)
  2. 通过掩膜池化获得 face/boundary/background 三个区域特征向量
  3. HDM (层次化解耦模块): 3 向量 → 7 向量 (借鉴 DeMo)
  4. CLS-Guided MoE: CLS token 引导的混合专家路由 → 融合特征
  5. BN + Linear → 真假二分类

完整数据流:
  image (B,3,224,224) + mask (B,3,16,16)
  │
  ▼ CLIPViTBackbone (+ RegionAdapter at layers [2,5,8,...,23])
  ├─ cls_token (B, 1024)           ← CLS 全局特征
  └─ spatial_tokens (B, 256, 1024) ← patch token 序列
  │
  ▼ RegionFeatureExtractor (掩膜池化)
  ├─ face_vec     (B, 1024)
  ├─ boundary_vec (B, 1024)
  └─ bg_vec       (B, 1024)
  │
  ▼ HierarchicalDecouplingModule (3→7)
  7 个特征向量, 各 (B, 1024)
  │
  ▼ CLSGuidedMoE (CLS 引导路由 + Multi-Head Expert)
  moe_feat (B, 7*1024) = (B, 7168)
  │
  ▼ BatchNorm1d(7168) → Linear(7168, 2)
  logits (B, 2)
"""

import torch
import torch.nn as nn
from typing import Optional, List

from lib.models.backbones.clip_vit import CLIPViTBackbone
from lib.models.modules.region_feature_extractor import RegionFeatureExtractor
from lib.models.modules.hierarchical_decoupling import HierarchicalDecouplingModule
from lib.models.modules.cls_guided_moe import CLSGuidedMoE


def _weights_init_classifier(m):
    """分类器权重初始化 (DeMo 风格)"""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def _weights_init_kaiming(m):
    """BN 权重初始化 (DeMo 风格)"""
    if isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)


class RegionDeMoDetector(nn.Module):
    """
    Region-Decoupled MoE Deepfake 检测器

    Args:
        device: 设备
        adapter_layers: RegionAdapter 插入的层索引列表
        adapter_type: 适配器类型 (默认 'region_adapter')
        adapter_kwargs: 适配器额外参数
        feat_dim: 特征维度 (CLIP ViT-L/14 = 1024)
        grid_size: 特征图空间分辨率 (224/14 = 16)
        hdm_num_heads: HDM cross-attention 头数
        hdm_dropout: HDM attention dropout
        moe_num_experts: MoE 每个 head 的专家数 (= 7)
        moe_num_heads: MoE multi-head 数量
        num_classes: 分类类别数 (默认 2: real/fake)
    """

    def __init__(
        self,
        device: str = "cuda",
        adapter_layers: Optional[List[int]] = None,
        adapter_type: str = "region_adapter",
        adapter_kwargs: dict = None,
        feat_dim: int = 1024,
        grid_size: int = 16,
        hdm_num_heads: int = 16,
        hdm_dropout: float = 0.1,
        moe_num_experts: int = 7,
        moe_num_heads: int = 16,
        num_classes: int = 2,
    ):
        super().__init__()

        self.feat_dim = feat_dim
        self.adapter_type = adapter_type

        # =====================================================================
        # 1. Backbone: CLIP ViT-L/14 (冻结) + RegionAdapter (可训练)
        # =====================================================================
        self.backbone = CLIPViTBackbone(
            device=device,
            adapter_layers=adapter_layers,
            adapter_type=adapter_type,
            adapter_kwargs=adapter_kwargs,
        )

        # =====================================================================
        # 2. 区域特征提取器: spatial tokens + masks → 3 vectors
        # =====================================================================
        self.region_extractor = RegionFeatureExtractor(
            feat_dim=feat_dim,
            grid_size=grid_size,
        )

        # =====================================================================
        # 3. HDM: 3 vectors + CLS → 7 vectors
        # =====================================================================
        self.hdm = HierarchicalDecouplingModule(
            feat_dim=feat_dim,
            num_heads=hdm_num_heads,
            dropout=hdm_dropout,
        )

        # =====================================================================
        # 4. CLS-Guided MoE: CLS + 7 vectors → fused feature
        # =====================================================================
        self.moe = CLSGuidedMoE(
            feat_dim=feat_dim,
            num_experts=moe_num_experts,
            num_heads=moe_num_heads,
        )

        # =====================================================================
        # 5. 分类头: BN + Linear (DeMo 风格)
        # =====================================================================
        moe_output_dim = moe_num_experts * feat_dim  # 7 * 1024 = 7168
        self.bottleneck = nn.BatchNorm1d(moe_output_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(_weights_init_kaiming)

        self.classifier = nn.Linear(moe_output_dim, num_classes, bias=False)
        self.classifier.apply(_weights_init_classifier)

        # 移到指定设备
        # (backbone 已在 CLIPViTBackbone.__init__ 中处理, 其余模块需要显式移动)
        self.region_extractor = self.region_extractor.to(device)
        self.hdm = self.hdm.to(device)
        self.moe = self.moe.to(device)
        self.bottleneck = self.bottleneck.to(device)
        self.classifier = self.classifier.to(device)

        # 配置信息
        self._config = {
            'adapter_layers': adapter_layers or [2, 5, 8, 11, 14, 17, 20, 23],
            'adapter_type': adapter_type,
            'feat_dim': feat_dim,
            'grid_size': grid_size,
            'hdm_num_heads': hdm_num_heads,
            'hdm_dropout': hdm_dropout,
            'moe_num_experts': moe_num_experts,
            'moe_num_heads': moe_num_heads,
            'num_classes': num_classes,
        }

    @property
    def config(self) -> dict:
        return self._config.copy()

    def forward(
        self,
        x: torch.Tensor,
        region_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入图像 (B, 3, 224, 224)
            region_masks: 三区域掩膜 (B, 3, 16, 16)

        Returns:
            logits: 分类 logits (B, 2)
        """
        # 1. Backbone → CLS token + spatial tokens
        cls_token, spatial_tokens = self.backbone.forward_dual(x, region_masks)
        # cls_token: (B, 1024), spatial_tokens: (B, 256, 1024)

        # 2. 区域特征提取
        face_vec, boundary_vec, bg_vec = self.region_extractor(
            spatial_tokens, region_masks
        )
        # 各 (B, 1024)

        # 3. HDM: 3 → 7 向量
        (face_feat, boundary_feat, bg_feat,
         face_boundary_feat, face_bg_feat, boundary_bg_feat,
         all_feat) = self.hdm(
            cls_token.float(), face_vec, boundary_vec, bg_vec
        )
        # 7 个向量, 各 (B, 1024)

        # 4. CLS-Guided MoE
        moe_feat = self.moe(
            cls_token.float(),
            face_feat, boundary_feat, bg_feat,
            face_boundary_feat, face_bg_feat, boundary_bg_feat,
            all_feat,
        )
        # (B, 7*1024) = (B, 7168)

        # 5. 分类
        feat = self.bottleneck(moe_feat)
        logits = self.classifier(feat)
        # (B, 2)

        return logits

    # ------------------------------------------------------------------
    # 参数管理
    # ------------------------------------------------------------------
    def get_trainable_parameters(self):
        """
        获取所有可训练参数

        可训练部分:
          - Backbone 中的 RegionAdapter
          - RegionFeatureExtractor (LayerNorm)
          - HDM (learnable tokens + MultiheadAttention)
          - CLSGuidedMoE (gating network + expert heads)
          - Bottleneck (BatchNorm)
          - Classifier (Linear)
        """
        # Backbone adapters
        for param in self.backbone.get_trainable_parameters():
            yield param
        # RegionFeatureExtractor
        for param in self.region_extractor.parameters():
            yield param
        # HDM
        for param in self.hdm.parameters():
            yield param
        # MoE
        for param in self.moe.parameters():
            yield param
        # Bottleneck + Classifier
        for param in self.bottleneck.parameters():
            yield param
        for param in self.classifier.parameters():
            yield param

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.get_trainable_parameters())

    def count_total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_parameter_breakdown(self) -> dict:
        """获取各模块可训练参数量的详细分解"""
        adapter_params = sum(
            p.numel() for p in self.backbone.get_trainable_parameters()
        )
        extractor_params = sum(
            p.numel() for p in self.region_extractor.parameters()
        )
        hdm_params = sum(p.numel() for p in self.hdm.parameters())
        moe_params = sum(p.numel() for p in self.moe.parameters())
        bn_params = sum(p.numel() for p in self.bottleneck.parameters())
        cls_params = sum(p.numel() for p in self.classifier.parameters())

        return {
            'backbone_adapters': adapter_params,
            'region_extractor': extractor_params,
            'hdm': hdm_params,
            'cls_guided_moe': moe_params,
            'bottleneck': bn_params,
            'classifier': cls_params,
            'total_trainable': (adapter_params + extractor_params + hdm_params
                                + moe_params + bn_params + cls_params),
        }
