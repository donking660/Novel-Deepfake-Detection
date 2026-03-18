"""
Region-Aware Adapter Module
区域感知适配器模块，将特征图分为人脸内部、边界、背景三个区域，分别进行独立卷积

设计:
  1. 通过掩膜图分别提取人脸内部、人脸与背景边界、背景三个区域的特征
  2. 对每个区域使用独立的瓶颈卷积分支进行处理
  3. 卷积后重新施加掩膜恢复原始空间布局
  4. 将三个区域的特征图相加，得到完整的适配特征图

前向: x + (scale_face * branch_face(x * m1) * m1 +
           scale_boundary * branch_boundary(x * m2) * m2 +
           scale_bg * branch_bg(x * m3) * m3)

参数量 (dim=1024, bottleneck_dim=128): ~2.0M/适配器 (3 个分支),
    8 个适配器共 ~16.1M
"""

import torch
import torch.nn as nn


def _make_branch(dim: int, bottleneck_dim: int, kernel_size: int, dropout: float) -> nn.Sequential:
    """
    创建一个瓶颈卷积分支（与 StandardAdapter 结构一致）
    
    结构:
      1. Conv2d 1x1 (降维)
      2. Conv2d 3x3 (空间特征提取)
      3. InstanceNorm2d + ReLU
      4. Dropout2d
      5. Conv2d 1x1 (升维)
    
    Args:
        dim: 输入/输出通道数
        bottleneck_dim: 瓶颈层通道数
        kernel_size: 空间卷积核大小
        dropout: Dropout 概率
    
    Returns:
        nn.Sequential 分支网络
    """
    padding = kernel_size // 2
    return nn.Sequential(
        # 降维: dim -> bottleneck_dim
        nn.Conv2d(dim, bottleneck_dim, kernel_size=1, bias=False),
        # 空间卷积
        nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=kernel_size,
                  stride=1, padding=padding, bias=False),
        nn.InstanceNorm2d(bottleneck_dim),
        nn.ReLU(inplace=True),
        # Dropout
        nn.Dropout2d(p=dropout),
        # 升维: bottleneck_dim -> dim
        nn.Conv2d(bottleneck_dim, dim, kernel_size=1, bias=False),
    )


class RegionAdapter(nn.Module):
    """
    区域感知适配器 (Region-Aware Adapter)
    
    将特征图按掩膜分为三个区域，每个区域使用独立的瓶颈卷积分支处理:
      - 人脸内部区域 (不含头发): 捕获面部纹理/颜色伪造痕迹
      - 边界区域 (人脸与背景过渡带): 捕获融合边界伪影
      - 背景区域 (含头发): 捕获背景不一致性
    
    每个分支有独立的可学习缩放因子 (scale)。
    
    Args:
        dim: 输入输出通道数，默认 1024 (ViT-L/14 hidden dim)
        bottleneck_dim: 瓶颈层通道数，默认 128
        kernel_size: 空间卷积核大小，默认 3
        dropout: Dropout 概率，默认 0.2
        init_scale: 每个分支的初始缩放因子，默认 0.01
        scale_trainable: scale 是否可训练，默认 True
    """
    
    def __init__(
        self,
        dim: int = 1024,
        bottleneck_dim: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.2,
        init_scale: float = 0.01,
        scale_trainable: bool = True,
    ):
        super().__init__()
        
        # 三个独立的瓶颈卷积分支
        self.branch_face = _make_branch(dim, bottleneck_dim, kernel_size, dropout)
        self.branch_boundary = _make_branch(dim, bottleneck_dim, kernel_size, dropout)
        self.branch_background = _make_branch(dim, bottleneck_dim, kernel_size, dropout)
        
        # 每个分支独立的缩放因子
        if scale_trainable:
            # 可学习 scale (nn.Parameter, requires_grad=True)
            self.scale_face = nn.Parameter(torch.ones(1) * init_scale)
            self.scale_boundary = nn.Parameter(torch.ones(1) * init_scale)
            self.scale_background = nn.Parameter(torch.ones(1) * init_scale)
        else:
            # 固定 scale (register_buffer, 不参与梯度计算)
            self.register_buffer('scale_face', torch.ones(1) * init_scale)
            self.register_buffer('scale_boundary', torch.ones(1) * init_scale)
            self.register_buffer('scale_background', torch.ones(1) * init_scale)
    
    def forward(self, x: torch.Tensor, region_masks: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征图，形状 (B, C, H, W) = (B, 1024, 16, 16)
            region_masks: 区域掩膜，形状 (B, 3, H, W) = (B, 3, 16, 16)
                channel 0: 人脸内部掩膜 (M_face)
                channel 1: 边界掩膜 (M_boundary)
                channel 2: 背景掩膜 (M_background)
                约束: M_face + M_boundary + M_background = 1 (像素级)
        
        Returns:
            适配后的特征图，形状 (B, C, H, W)
        """
        # 分离三个区域掩膜: (B, 1, H, W)
        m_face = region_masks[:, 0:1, :, :]
        m_boundary = region_masks[:, 1:2, :, :]
        m_bg = region_masks[:, 2:3, :, :]
        
        # 分支 1: 人脸内部 — 提取区域特征 → 卷积 → 重新掩膜
        feat_face = self.branch_face(x * m_face) * m_face
        
        # 分支 2: 边界 — 提取区域特征 → 卷积 → 重新掩膜
        feat_boundary = self.branch_boundary(x * m_boundary) * m_boundary
        
        # 分支 3: 背景 — 提取区域特征 → 卷积 → 重新掩膜
        feat_bg = self.branch_background(x * m_bg) * m_bg
        
        # 加权求和 + 残差连接
        adapted = (self.scale_face * feat_face
                   + self.scale_boundary * feat_boundary
                   + self.scale_background * feat_bg)
        
        return x + adapted
