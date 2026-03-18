"""
Deepfake Detector Module
Deepfake 检测器，结合 CLIP ViT Backbone 和线性分类头 (DF40 风格)

设计:
  - 不使用 CLIP 投影层，直接使用 1024 维 CLS token
  - 使用简单的线性分类头 Linear(1024, 2)
  - 支持标准适配器和区域感知适配器
"""

import torch
import torch.nn as nn
from typing import Optional, List

from lib.models.backbones.clip_vit import CLIPViTBackbone


class DeepfakeDetector(nn.Module):
    """
    Deepfake 检测器 (DF40 风格)
    结合 CLIPViTBackbone 特征提取器和线性分类头
    
    - Backbone 输出 1024 维 CLS token (不使用 CLIP 投影层)
    - 分类头为简单的 Linear(1024, 2)
    - 支持两种适配器: standard (标准) 和 region_adapter (区域感知)
    
    Args:
        device: 设备，默认"cuda"
        adapter_layers: 插入适配器的层索引列表
        adapter_type: 适配器类型，'standard' 或 'region_adapter'
        adapter_kwargs: 传递给适配器构造函数的额外参数
    """
    
    def __init__(
        self,
        device: str = "cuda",
        adapter_layers: Optional[List[int]] = None,
        adapter_type: str = "standard",
        adapter_kwargs: dict = None,
    ):
        super().__init__()
        
        self.adapter_type = adapter_type
        
        # 特征提取器 (不使用 CLIP 投影层，输出 1024 维)
        self.backbone = CLIPViTBackbone(
            device=device,
            adapter_layers=adapter_layers,
            adapter_type=adapter_type,
            adapter_kwargs=adapter_kwargs,
        )
        
        # 获取 backbone 输出维度 (1024)
        input_dim = self.backbone.output_dim
        
        # DF40 风格：简单线性分类头
        self.classifier = nn.Linear(input_dim, 2)
        
        # 将分类头移到正确的设备
        self.classifier = self.classifier.to(device)
        
        # 保存配置信息
        self._config = {
            'adapter_layers': adapter_layers or [2, 5, 8, 11, 14, 17, 20, 23],
            'adapter_type': adapter_type,
            'input_dim': input_dim
        }
    
    @property
    def config(self) -> dict:
        """返回模型配置"""
        return self._config.copy()
    
    def forward(self, x: torch.Tensor, region_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像，形状为 (B, 3, 224, 224)
            region_masks: 区域掩膜 (仅 region_adapter 需要)，形状为 (B, 3, 16, 16)
        Returns:
            分类 logits，形状为 (B, 2)
        """
        # 提取特征
        features = self.backbone(x, region_masks)
        # 分类
        logits = self.classifier(features.float())
        return logits
    
    def get_trainable_parameters(self):
        """
        获取所有可训练参数
        
        Returns:
            可训练参数的生成器
        """
        # 适配器参数
        for param in self.backbone.get_trainable_parameters():
            yield param
        # 分类头参数
        for param in self.classifier.parameters():
            yield param
    
    def count_trainable_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def count_total_parameters(self) -> int:
        """统计总参数数量"""
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Testing DeepfakeDetector (DF40 Style)")
    print("=" * 60)
    
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    # 创建模型
    model = DeepfakeDetector(device=device)
    model = model.to(device)
    
    # 打印配置
    print(f"\n模型配置: {model.config}")
    
    # 统计参数
    total_params = model.count_total_parameters()
    trainable_params = model.count_trainable_parameters()
    
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  可训练比例: {trainable_params / total_params * 100:.2f}%")
    
    # 测试前向传播
    print(f"\n测试前向传播:")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    
    assert output.shape == (2, 2), f"输出维度不匹配: {output.shape}"
    
    print(f"  输出维度: {model.backbone.output_dim} -> 2")
    print(f"  ✅ 测试通过!")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)
