"""
CLIP ViT Backbone
基于CLIP ViT-L/14的视觉backbone，支持适配器插入

支持两种适配器类型:
  - standard: 标准瓶颈适配器 (StandardAdapter)
  - region_adapter: 区域感知适配器 (RegionAdapter)，需要传入区域掩膜
"""

import torch
import torch.nn as nn
import clip
from typing import Optional

from lib.models.adapters.standard_adapter import StandardAdapter
from lib.models.adapters.region_adapter import RegionAdapter


class CLIPViTBackbone(nn.Module):
    """
    带适配器的CLIP ViT-L/14模型
    
    在ViT-L的第3, 6, 9, 12, 15, 18, 21, 24层（block index 2, 5, 8, 11, 14, 17, 20, 23）
    输出位置插入适配器，实现均匀密集的特征适配
    
    Args:
        pretrained: 预训练模型名称，默认"ViT-L/14"
        device: 设备，默认"cuda"
        adapter_layers: 插入适配器的层索引列表，默认每隔3层插入一个
        adapter_type: 适配器类型，'standard' 或 'region_adapter'
        adapter_kwargs: 传递给适配器构造函数的额外参数
    
    输出:
        不使用 CLIP 投影层，直接输出 1024 维 CLS token (与 DF40/HuggingFace 一致)
    """
    
    def __init__(
        self,
        pretrained: str = "ViT-L/14",
        device: str = "cuda",
        adapter_layers: list = None,
        adapter_type: str = "standard",
        adapter_kwargs: dict = None,
    ):
        super().__init__()
        
        # 默认在第3, 6, 9, 12, 15, 18, 21, 24层插入适配器 (索引从0开始)
        # 每隔3层插入一个，共8个适配器，覆盖浅层、中层、深层特征
        if adapter_layers is None:
            adapter_layers = [2, 5, 8, 11, 14, 17, 20, 23]
        self.adapter_layers = adapter_layers
        self.adapter_type = adapter_type
        
        # 加载CLIP预训练模型
        self.clip_model, self.preprocess = clip.load(pretrained, device=device)
        
        # 获取视觉编码器
        self.visual = self.clip_model.visual
        
        # 冻结所有原始CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # ViT-L/14 的配置
        self.hidden_dim = 1024  # ViT-L/14 hidden dimension
        self.patch_size = 14
        self.grid_size = 16  # 224 / 14 = 16
        
        # 投影后的维度 (ViT-L/14 proj 输出为 768)
        self.proj_dim = self.visual.proj.shape[1] if self.visual.proj is not None else self.hidden_dim
        
        # 适配器的额外参数
        extra_kwargs = adapter_kwargs or {}
        
        # 为每个指定层创建适配器
        if adapter_type == 'region_adapter':
            self.adapters = nn.ModuleDict({
                str(layer_idx): RegionAdapter(dim=self.hidden_dim, **extra_kwargs)
                for layer_idx in self.adapter_layers
            })
        else:
            # 默认: standard adapter
            self.adapters = nn.ModuleDict({
                str(layer_idx): StandardAdapter(dim=self.hidden_dim, **extra_kwargs)
                for layer_idx in self.adapter_layers
            })
    
    @property
    def output_dim(self) -> int:
        """
        返回输出特征维度 (1024)
        """
        return self.hidden_dim  # 1024
    
    def forward_visual(self, x: torch.Tensor, region_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        视觉编码器前向传播，在指定层插入适配器
        
        Args:
            x: 输入图像，形状为 (B, 3, 224, 224)
            region_masks: 区域掩膜 (仅 region_adapter 需要)，形状为 (B, 3, 16, 16)
                          None 表示使用标准适配器
        Returns:
            图像特征: (B, 1024)
        """
        # 获取视觉Transformer的组件
        vit = self.visual
        
        # Patch Embedding: (B, 3, 224, 224) -> (B, hidden_dim, grid, grid)
        x = vit.conv1(x)  # (B, 1024, 16, 16)
        
        # Reshape: (B, hidden_dim, grid, grid) -> (B, hidden_dim, grid*grid)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, 1024, 256)
        
        # Permute: (B, hidden_dim, grid*grid) -> (B, grid*grid, hidden_dim)
        x = x.permute(0, 2, 1)  # (B, 256, 1024)
        
        # 添加class token: (B, 256, 1024) -> (B, 257, 1024)
        class_token = vit.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)  # (B, 257, 1024)
        
        # 添加位置编码
        x = x + vit.positional_embedding.to(x.dtype)
        
        # Layer Norm (pre)
        x = vit.ln_pre(x)
        
        # Permute for transformer: (B, L, D) -> (L, B, D)
        x = x.permute(1, 0, 2)  # (257, B, 1024)
        
        # 逐层通过Transformer blocks
        for layer_idx, block in enumerate(vit.transformer.resblocks):
            # 通过Transformer block
            x = block(x)
            
            # 如果当前层需要插入适配器
            if layer_idx in self.adapter_layers:
                x = self._apply_adapter(x, layer_idx, region_masks)
        
        # Permute back: (L, B, D) -> (B, L, D)
        x = x.permute(1, 0, 2)  # (B, 257, 1024)
        
        # Layer Norm (post) - 只取 CLS token
        x = vit.ln_post(x[:, 0, :])  # (B, 1024)
        
        return x
    
    def _apply_adapter(self, x: torch.Tensor, layer_idx: int, 
                       region_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        在指定层应用适配器
        
        Args:
            x: 特征序列，形状为 (L, B, D) = (257, B, 1024)
            layer_idx: 当前层索引
            region_masks: 区域掩膜 (仅 region_adapter 需要)，形状为 (B, 3, 16, 16)
        Returns:
            处理后的特征序列，形状为 (L, B, D)
        """
        # 保存原始数据类型（CLIP使用float16）
        original_dtype = x.dtype
        
        # 分离class token和spatial tokens
        # x: (257, B, 1024)
        class_token = x[0:1, :, :]  # (1, B, 1024)
        spatial_tokens = x[1:, :, :]  # (256, B, 1024)
        
        # 获取batch size
        L, B, D = spatial_tokens.shape  # L=256, D=1024
        
        # Reshape spatial tokens: (256, B, 1024) -> (B, 1024, 16, 16)
        spatial_tokens = spatial_tokens.permute(1, 2, 0)  # (B, 1024, 256)
        spatial_tokens = spatial_tokens.reshape(B, D, self.grid_size, self.grid_size)  # (B, 1024, 16, 16)
        
        # 转换为float32以匹配适配器权重
        spatial_tokens = spatial_tokens.float()
        
        # 应用适配器
        adapter = self.adapters[str(layer_idx)]
        if self.adapter_type == 'region_adapter' and region_masks is not None:
            # 区域感知适配器: 需要传入区域掩膜
            spatial_tokens = adapter(spatial_tokens, region_masks.float())
        else:
            # 标准适配器
            spatial_tokens = adapter(spatial_tokens)
        
        # 转换回原始数据类型
        spatial_tokens = spatial_tokens.to(original_dtype)
        
        # Reshape back: (B, 1024, 16, 16) -> (256, B, 1024)
        spatial_tokens = spatial_tokens.reshape(B, D, -1)  # (B, 1024, 256)
        spatial_tokens = spatial_tokens.permute(2, 0, 1)  # (256, B, 1024)
        
        # 拼接回class token
        x = torch.cat([class_token, spatial_tokens], dim=0)  # (257, B, 1024)
        
        return x
    
    def forward_with_spatial(self, x: torch.Tensor, region_masks: Optional[torch.Tensor] = None):
        """
        视觉编码器前向传播，返回 CLS token 和空间特征
        
        与 forward_visual 逻辑完全一致，但额外返回最后一层的空间特征 (patch tokens)。
        CLS token 经过 ln_post，空间特征保持原样 (与 DeMo 的 global/cash 对应)。
        
        Args:
            x: 输入图像，形状为 (B, 3, 224, 224)，已转换为 CLIP dtype
            region_masks: 区域掩膜 (仅 region_adapter 需要)，形状为 (B, 3, 16, 16)
        Returns:
            cls_token: (B, 1024) — 经过 ln_post 的 CLS token
            spatial_tokens: (B, 256, 1024) — 最后一层的 patch token 序列 (未经 ln_post)
        """
        vit = self.visual
        
        # Patch Embedding
        x = vit.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # 添加 class token
        class_token = vit.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)
        
        # 位置编码
        x = x + vit.positional_embedding.to(x.dtype)
        
        # Layer Norm (pre)
        x = vit.ln_pre(x)
        
        # Permute for transformer: (B, L, D) -> (L, B, D)
        x = x.permute(1, 0, 2)
        
        # 逐层通过 Transformer blocks
        for layer_idx, block in enumerate(vit.transformer.resblocks):
            x = block(x)
            if layer_idx in self.adapter_layers:
                x = self._apply_adapter(x, layer_idx, region_masks)
        
        # Permute back: (L, B, D) -> (B, L, D)
        x = x.permute(1, 0, 2)  # (B, 257, 1024)
        
        # 分离 CLS token 和空间特征
        spatial_tokens = x[:, 1:, :]        # (B, 256, 1024) — 不经过 ln_post
        cls_token = vit.ln_post(x[:, 0, :])  # (B, 1024) — 经过 ln_post
        
        return cls_token, spatial_tokens
    
    def forward(self, images: torch.Tensor, region_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            images: 输入图像，形状为 (B, 3, 224, 224)
            region_masks: 区域掩膜 (仅 region_adapter 需要)，形状为 (B, 3, 16, 16)
        Returns:
            图像特征: (B, 1024)
        """
        return self.forward_visual(images.type(self.clip_model.dtype), region_masks)
    
    def forward_dual(self, images: torch.Tensor, region_masks: Optional[torch.Tensor] = None):
        """
        前向传播，同时返回 CLS token 和空间特征
        
        Args:
            images: 输入图像，形状为 (B, 3, 224, 224)
            region_masks: 区域掩膜，形状为 (B, 3, 16, 16)
        Returns:
            cls_token: (B, 1024)
            spatial_tokens: (B, 256, 1024)
        """
        return self.forward_with_spatial(images.type(self.clip_model.dtype), region_masks)
    
    def get_trainable_parameters(self):
        """
        获取所有可训练参数（仅适配器参数）
        
        Returns:
            可训练参数的生成器
        """
        for adapter in self.adapters.values():
            for param in adapter.parameters():
                yield param
    
    def count_trainable_parameters(self) -> int:
        """
        统计可训练参数数量
        
        Returns:
            可训练参数总数
        """
        return sum(p.numel() for p in self.get_trainable_parameters())
    
    def count_total_parameters(self) -> int:
        """
        统计总参数数量
        
        Returns:
            总参数数
        """
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("Testing CLIPViTBackbone with StandardAdapter")
    print("=" * 60)
    
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    # 创建模型
    model = CLIPViTBackbone(device=device)
    model = model.to(device)
    
    # 统计参数
    total_params = model.count_total_parameters()
    trainable_params = model.count_trainable_parameters()
    
    print(f"\n模型参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  可训练比例: {trainable_params / total_params * 100:.2f}%")
    print(f"  输出维度: {model.output_dim}")
    
    # 测试前向传播
    print(f"\n测试前向传播:")
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"  输入形状: {dummy_input.shape}")
    print(f"  输出形状: {output.shape}")
    
    assert output.shape == (2, 1024), f"输出维度不匹配: {output.shape}"
    
    print(f"  ✅ 测试通过!")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试通过!")
    print("=" * 60)
