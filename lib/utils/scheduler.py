"""
Learning Rate Scheduler with Warmup
带 Warmup 的学习率调度器
"""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    带 Linear Warmup 的 Cosine Annealing 学习率调度器
    
    学习率变化过程：
    1. Warmup 阶段 (0 ~ warmup_epochs): 
       lr 从 0 线性增加到 base_lr
    2. Cosine 阶段 (warmup_epochs ~ total_epochs):
       lr 从 base_lr 按余弦曲线衰减到 min_lr
    
    Args:
        optimizer: 优化器
        warmup_epochs: Warmup 的 epoch 数
        total_epochs: 总 epoch 数
        min_lr: 最小学习率，默认 0
        last_epoch: 上一个 epoch 编号，用于恢复训练
    
    Usage:
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> scheduler = WarmupCosineAnnealingLR(
        ...     optimizer, 
        ...     warmup_epochs=5, 
        ...     total_epochs=100,
        ...     min_lr=1e-6
        ... )
        >>> 
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     scheduler.step()
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        
        # 保存初始学习率
        self.base_lrs_copy = None
        
        super().__init__(optimizer, last_epoch)
        
        # 在初始化后保存 base_lrs
        self.base_lrs_copy = list(self.base_lrs)
    
    def get_lr(self):
        """计算当前 epoch 的学习率"""
        if self.base_lrs_copy is None:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        current_epoch = self.last_epoch
        
        if current_epoch < self.warmup_epochs:
            # Warmup 阶段：线性增加
            # lr = base_lr * (epoch / warmup_epochs)
            warmup_factor = (current_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs_copy]
        else:
            # Cosine 阶段
            # 计算 cosine 进度
            progress = (current_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            # cosine 衰减公式
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs_copy
            ]


class WarmupLinearDecayLR(_LRScheduler):
    """
    带 Linear Warmup 的线性衰减学习率调度器
    
    Args:
        optimizer: 优化器
        warmup_epochs: Warmup 的 epoch 数
        total_epochs: 总 epoch 数
        min_lr: 最小学习率
        last_epoch: 上一个 epoch 编号
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs_copy = None
        
        super().__init__(optimizer, last_epoch)
        self.base_lrs_copy = list(self.base_lrs)
    
    def get_lr(self):
        if self.base_lrs_copy is None:
            return [group['lr'] for group in self.optimizer.param_groups]
        
        current_epoch = self.last_epoch
        
        if current_epoch < self.warmup_epochs:
            # Warmup 阶段
            warmup_factor = (current_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs_copy]
        else:
            # 线性衰减阶段
            progress = (current_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            linear_factor = 1.0 - progress
            
            return [
                self.min_lr + (base_lr - self.min_lr) * linear_factor
                for base_lr in self.base_lrs_copy
            ]


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    num_cycles: float = 0.5,
    min_lr: float = 0.0,
    last_epoch: int = -1
) -> _LRScheduler:
    """
    创建带 warmup 的 cosine 学习率调度器（按 step 计算）
    
    适用于需要按 step 而非 epoch 调度学习率的场景
    
    Args:
        optimizer: 优化器
        warmup_steps: Warmup 的 step 数
        total_steps: 总 step 数
        num_cycles: cosine 周期数，默认 0.5（半个周期）
        min_lr: 最小学习率
        last_epoch: 上一个 step 编号
    
    Returns:
        LambdaLR 调度器
    """
    from torch.optim.lr_scheduler import LambdaLR
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Warmup 阶段
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine 阶段
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(
            min_lr,
            0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class EarlyStopping:
    """
    早停机制
    
    监控指定指标，如果连续 patience 个 epoch 没有改善，则触发早停
    
    Args:
        patience: 容忍的 epoch 数
        min_delta: 最小改善量，小于此值不算改善
        mode: 'max' 或 'min'，指标是越大越好还是越小越好
        verbose: 是否打印信息
    
    Usage:
        >>> early_stopping = EarlyStopping(patience=10, mode='max')
        >>> 
        >>> for epoch in range(epochs):
        >>>     val_acc = validate(...)
        >>>     
        >>>     if early_stopping(val_acc):
        >>>         print("Early stopping triggered!")
        >>>         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int = 0) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前指标值
            epoch: 当前 epoch（用于记录最佳 epoch）
        
        Returns:
            True 如果应该早停，否则 False
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} "
                      f"(best={self.best_score:.4f} at epoch {self.best_epoch})")
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def reset(self):
        """重置早停计数器"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
