"""
Standard Adapter Module (Bottleneck Version)
瓶颈结构适配器模块，用于CLIP视觉编码器的参数高效微调
"""

import torch
import torch.nn as nn


class StandardAdapter(nn.Module):
    """
    改进版适配器：瓶颈结构 (Bottleneck)
    
    结构: 
      1. Conv2d 1x1 (降维)
      2. Conv2d 3x3 (空间特征提取)
      3. InstanceNorm2d + GELU (归一化 + 非线性)
      4. Dropout2d (正则化)
      5. Conv2d 1x1 (升维)
    
    前向: x + scale * adapter(x)
    
    参数量 (dim=1024, bottleneck_dim=128): ~0.67M/适配器, 8个适配器共 ~5.4M
    
    Args:
        dim: 输入输出通道数，默认1024（对应ViT-L/14的hidden dim）
        bottleneck_dim: 瓶颈层通道数，默认128（增加容量以捕获更复杂的伪造特征）
        kernel_size: 空间卷积核大小，默认3
        dropout: Dropout概率，默认0.2
    """
    
    def __init__(self, dim: int = 1024, bottleneck_dim: int = 128, 
                 kernel_size: int = 3, dropout: float = 0.2,
                 init_scale: float = 0.05):
        super().__init__()
        
        padding = kernel_size // 2  # k=3 -> p=1
        
        self.adapter = nn.Sequential(
            # 1. 降维: 1024 -> 128
            nn.Conv2d(dim, bottleneck_dim, kernel_size=1, bias=False),
            
            # 2. 空间卷积: 在低维空间进行 3x3 卷积，捕获纹理/边缘
            nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=kernel_size, 
                      stride=1, padding=padding, bias=False),
            nn.InstanceNorm2d(bottleneck_dim),
            nn.GELU(),
            
            # 3. Dropout: 增加正则化，防止过拟合
            nn.Dropout2d(p=dropout),
            
            # 4. 升维: 128 -> 1024
            nn.Conv2d(bottleneck_dim, dim, kernel_size=1, bias=False)
        )
        
        # 可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1) * init_scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征，形状为 (B, C, H, W)
        Returns:
            输出特征，形状为 (B, C, H, W)
        """
        return x + self.scale * self.adapter(x)
