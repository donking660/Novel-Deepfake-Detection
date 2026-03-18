"""
Exponential Moving Average (EMA) Module
指数移动平均模块，用于平滑模型权重
"""

import copy
from typing import Optional

import torch
import torch.nn as nn


class EMA:
    """
    Exponential Moving Average (EMA)
    
    EMA 通过对模型参数进行指数加权平均，可以：
    1. 平滑训练过程中的参数波动
    2. 提高模型在验证/测试时的稳定性和性能
    3. 起到类似于集成学习的效果
    
    公式: ema_param = decay * ema_param + (1 - decay) * model_param
    
    Args:
        model: 要进行 EMA 的模型
        decay: 衰减系数，通常设置为 0.999 或 0.9999
        device: 设备
    
    Usage:
        >>> model = MyModel()
        >>> ema = EMA(model, decay=0.999)
        >>> 
        >>> # 训练循环
        >>> for epoch in range(epochs):
        >>>     for batch in dataloader:
        >>>         loss = train_step(model, batch)
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         ema.update()  # 每次梯度更新后调用
        >>>     
        >>>     # 验证时使用 EMA 模型
        >>>     ema.apply_shadow()
        >>>     validate(model)  # 此时 model 使用 EMA 权重
        >>>     ema.restore()    # 恢复原始权重继续训练
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.decay = decay
        self.device = device
        
        # 存储 EMA 参数（shadow）
        self.shadow = {}
        # 存储原始参数（用于恢复）
        self.backup = {}
        
        # 初始化 EMA 参数为模型当前参数
        self._register()
    
    def _register(self):
        """注册模型参数到 shadow"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """
        更新 EMA 参数
        
        公式: shadow = decay * shadow + (1 - decay) * param
        应在每次优化器 step 后调用
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (
                    self.decay * self.shadow[name] + 
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """
        将 EMA 参数应用到模型（用于验证/测试）
        
        调用后模型使用 EMA 权重
        记得之后调用 restore() 恢复原始权重
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """
        恢复模型的原始参数（从 backup）
        
        在验证完成后调用，以便继续训练
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}
    
    def state_dict(self) -> dict:
        """返回 EMA 的状态字典（用于保存）"""
        return {
            'decay': self.decay,
            'shadow': copy.deepcopy(self.shadow)
        }
    
    def load_state_dict(self, state_dict: dict):
        """加载 EMA 的状态字典（用于恢复）"""
        self.decay = state_dict['decay']
        self.shadow = copy.deepcopy(state_dict['shadow'])
    
    def get_ema_model_state_dict(self) -> dict:
        """
        获取 EMA 模型的 state_dict（用于保存最佳模型）
        
        Returns:
            包含 EMA 参数的模型 state_dict
        """
        # 创建一个完整的 state_dict
        model_state = self.model.state_dict()
        
        # 用 shadow 参数替换可训练参数
        for name in self.shadow:
            if name in model_state:
                model_state[name] = self.shadow[name].clone()
        
        return model_state


class ModelEMA:
    """
    另一种 EMA 实现：维护一个完整的模型副本
    
    这种实现更简洁，但内存占用更大
    适用于需要直接使用 EMA 模型进行推理的场景
    
    Args:
        model: 要进行 EMA 的模型
        decay: 衰减系数
        device: 设备（可选，默认与模型相同）
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        # 创建模型的深拷贝作为 EMA 模型
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()  # EMA 模型始终处于 eval 模式
        
        # 禁用 EMA 模型的梯度
        for param in self.ema_model.parameters():
            param.requires_grad = False
        
        self.decay = decay
        self.device = device
        
        if device is not None:
            self.ema_model.to(device)
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        更新 EMA 模型参数
        
        Args:
            model: 主模型（训练中的模型）
        """
        for ema_param, model_param in zip(
            self.ema_model.parameters(), 
            model.parameters()
        ):
            ema_param.data.mul_(self.decay).add_(
                model_param.data, alpha=1.0 - self.decay
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """使用 EMA 模型进行前向传播"""
        return self.ema_model(x)
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def state_dict(self) -> dict:
        """返回 EMA 模型的 state_dict"""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict: dict):
        """加载 EMA 模型的 state_dict"""
        self.ema_model.load_state_dict(state_dict)
