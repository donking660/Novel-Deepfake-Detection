#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deepfake Detection 训练脚本 (带面部掩膜支持)

在 tools/train.py 基础上增加:
  - 支持 DF40DatasetWithMask 数据集 (加载面部掩膜 + 预计算三区域掩膜)
  - 支持 RegionAdapter (区域感知适配器)
  - 训练循环传递 region_masks 到模型

通过配置文件中的 model.adapter.type 和 mask.enabled 控制:
  - adapter.type == 'standard': 使用标准训练流程 (兼容 train.py)
  - adapter.type == 'region_adapter' + mask.enabled == true: 使用掩膜训练流程

使用方法:
    python tools/train_with_mask.py --config configs/baseline_region_adapter.yaml
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# sklearn 指标
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    roc_curve
)

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入自定义模块
from lib.data.dataset_with_mask import DF40DatasetWithMask
from lib.data.transfroms import get_train_transforms, get_val_transforms
from lib.models.detector import DeepfakeDetector
from lib.utils.ema import ModelEMA
from lib.utils.scheduler import WarmupCosineAnnealingLR, EarlyStopping
from lib.utils.config import get_config, Config, save_config


# ============================================================================
# 日志配置 (与 train.py 一致)
# ============================================================================
class TeeLogger:
    """将输出同时写入控制台和文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log_file = open(log_file, 'a', buffering=1)
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()


def setup_logging(log_dir: str, experiment_name: str = None):
    """设置日志系统"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_file = log_dir / f"{experiment_name}_{timestamp}.log"
    else:
        log_file = log_dir / f"train_{timestamp}.log"
    
    tee = TeeLogger(str(log_file))
    sys.stdout = tee
    sys.stderr = tee
    
    print(f"=" * 70)
    print(f"日志文件: {log_file}")
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 70)
    
    return str(log_file)


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> float:
    """计算 EER (Equal Error Rate)"""
    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        return float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    except:
        return 0.0


# ============================================================================
# 训练和验证函数
# ============================================================================
def train_one_epoch(
    model: nn.Module,
    ema: 'ModelEMA',
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    wandb_run=None
) -> float:
    """训练一个 epoch (带掩膜支持)"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (images, labels, region_masks) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        region_masks = region_masks.to(device)
        
        optimizer.zero_grad()
        logits = model(images, region_masks)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        # 更新 EMA 模型
        ema.update(model)
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if wandb_run is not None and batch_idx % 100 == 0:
            wandb_run.log({'train/batch_loss': loss.item()})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def validate(
    ema: 'ModelEMA',
    dataloader: DataLoader,
    device: torch.device,
    epoch: int
) -> dict:
    """使用 EMA 模型进行验证 (带掩膜支持)"""
    ema.ema_model.eval()
    
    tp = tn = fp = fn = 0
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val-EMA]")
    
    for images, labels, region_masks in pbar:
        images = images.to(device)
        labels_tensor = labels.to(device)
        region_masks = region_masks.to(device)
        
        logits = ema.ema_model(images, region_masks)
        probs = torch.softmax(logits, dim=1)[:, 1]
        predictions = torch.argmax(logits, dim=1)
        
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        
        tp += ((predictions == 1) & (labels_tensor == 1)).sum().item()
        tn += ((predictions == 0) & (labels_tensor == 0)).sum().item()
        fp += ((predictions == 1) & (labels_tensor == 0)).sum().item()
        fn += ((predictions == 0) & (labels_tensor == 1)).sum().item()
        
        total = tp + tn + fp + fn
        acc = (tp + tn) / total if total > 0 else 0
        pbar.set_postfix({'acc': f'{acc*100:.2f}%'})
    
    total = tp + tn + fp + fn
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    auc = ap = eer = 0.0
    if len(np.unique(all_labels)) > 1:
        try:
            auc = roc_auc_score(all_labels, all_probs)
            ap = average_precision_score(all_labels, all_probs)
            eer = compute_eer(all_labels, all_probs)
        except Exception as e:
            print(f"指标计算失败: {e}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'ap': ap,
        'eer': eer,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'all_probs': all_probs,
        'all_labels': all_labels
    }


# ============================================================================
# WandB 日志 (与 train.py 一致)
# ============================================================================
def log_to_wandb(wandb_run, epoch: int, train_loss: float, val_metrics: dict, 
                 lr: float, epoch_stats: dict = None):
    """记录指标到 wandb"""
    import wandb
    
    log_dict = {
        'epoch': epoch,
        'train/loss': train_loss,
        'val/accuracy': val_metrics['accuracy'],
        'val/precision': val_metrics['precision'],
        'val/recall': val_metrics['recall'],
        'val/f1': val_metrics['f1'],
        'val/auc': val_metrics['auc'],
        'val/ap': val_metrics['ap'],
        'val/eer': val_metrics['eer'],
        'learning_rate': lr
    }
    
    if epoch_stats:
        log_dict['train/epoch_samples'] = epoch_stats.get('total', 0)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        cm = np.array([
            [val_metrics['tn'], val_metrics['fp']],
            [val_metrics['fn'], val_metrics['tp']]
        ], dtype=int)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Real (Pred)', 'Fake (Pred)'],
                    yticklabels=['Real (True)', 'Fake (True)'])
        ax.set_title(f'Confusion Matrix (EMA) - Epoch {epoch}')
        plt.tight_layout()
        log_dict['val/confusion_matrix'] = wandb.Image(fig)
        plt.close(fig)
    except:
        pass
    
    if epoch % 10 == 0 or epoch == 1:
        try:
            if len(np.unique(val_metrics['all_labels'])) > 1:
                fpr, tpr, _ = roc_curve(val_metrics['all_labels'], val_metrics['all_probs'])
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(fpr, tpr, 'b-', linewidth=2, 
                       label=f'ROC (AUC = {val_metrics["auc"]:.4f})')
                ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'ROC Curve (EMA) - Epoch {epoch}')
                ax.legend(loc='lower right')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                log_dict['val/roc_curve'] = wandb.Image(fig)
                plt.close(fig)
        except:
            pass
    
    wandb_run.log(log_dict)


def init_wandb(cfg: Config, model, total_params: int, trainable_params: int):
    """初始化 wandb"""
    import wandb
    
    run_name = f"{cfg.experiment.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    run = wandb.init(
        project=cfg.wandb.project,
        name=run_name,
        config=cfg.to_dict(),
        tags=['deepfake-detection', 'clip-vit', 'region-adapter'],
        reinit=True
    )
    
    return run


# ============================================================================
# 主训练函数
# ============================================================================
def train(cfg: Config, args):
    """主训练函数 (带掩膜支持)"""
    device = torch.device(f'cuda:{cfg.device}' if torch.cuda.is_available() else 'cpu')
    
    # 获取掩膜配置
    mask_cfg = cfg.get('mask', None)
    use_mask = mask_cfg is not None and getattr(mask_cfg, 'enabled', False)
    erosion_kernel_size = getattr(mask_cfg, 'erosion_kernel_size', 19) if mask_cfg else 19
    feature_map_size = getattr(mask_cfg, 'feature_map_size', 16) if mask_cfg else 16
    
    # 获取适配器类型
    adapter_type = getattr(cfg.model.adapter, 'type', 'standard')
    
    print("=" * 70)
    print("Deepfake Detection Training (with Mask Support)")
    print("=" * 70)
    
    print(f"\n实验信息:")
    print(f"  名称: {cfg.experiment.name}")
    print(f"  描述: {cfg.experiment.description}")
    
    print(f"\n配置:")
    print(f"  Device: {device}")
    print(f"  Batch Size: {cfg.train.batch_size}")
    print(f"  Learning Rate: {cfg.optimizer.lr}")
    print(f"  Epochs: {cfg.train.epochs}")
    print(f"  Warmup Epochs: {cfg.train.warmup_epochs}")
    print(f"  EMA Decay: {cfg.ema.decay}")
    print(f"  Early Stopping Patience: {cfg.train.patience}")
    print(f"  Adapter Type: {adapter_type}")
    print(f"  Use Mask: {use_mask}")
    if use_mask:
        print(f"  Erosion Kernel Size: {erosion_kernel_size}")
        print(f"  Feature Map Size: {feature_map_size}")
    
    # =========================================================================
    # 1. 数据加载
    # =========================================================================
    print("\n" + "=" * 70)
    print("加载数据集 (带掩膜)...")
    print("=" * 70)
    
    # 数据增强配置
    aug_config = {
        'resolution': cfg.data.resolution,
        'flip_prob': cfg.data_aug.flip_prob,
        'rotate_prob': cfg.data_aug.rotate_prob,
        'rotate_limit': cfg.data_aug.rotate_limit,
        'blur_prob': cfg.data_aug.blur_prob,
        'blur_limit': cfg.data_aug.blur_limit,
        'brightness_limit': cfg.data_aug.brightness_limit,
        'contrast_limit': cfg.data_aug.contrast_limit,
        'quality_lower': cfg.data_aug.quality_lower,
        'quality_upper': cfg.data_aug.quality_upper,
    }
    
    train_transform = get_train_transforms(config=aug_config)
    val_transform = get_val_transforms(config=aug_config)
    
    print(f"  分辨率: {cfg.data.resolution}x{cfg.data.resolution}")
    
    # 训练集 (带掩膜)
    train_dataset = DF40DatasetWithMask(
        json_path=str(cfg.data.train_json),
        data_root=str(cfg.data.data_root),
        transform=train_transform,
        balance_mode=cfg.data.balance_mode,
        seed=cfg.train.seed,
        erosion_kernel_size=erosion_kernel_size,
        feature_map_size=feature_map_size
    )
    
    # 验证集 (带掩膜)
    val_json = cfg.data.val_json if cfg.data.val_json else cfg.data.train_json
    val_dataset = DF40DatasetWithMask(
        json_path=str(val_json),
        data_root=str(cfg.data.data_root),
        transform=val_transform,
        balance_mode='static',
        seed=cfg.train.seed + 10000,
        erosion_kernel_size=erosion_kernel_size,
        feature_map_size=feature_map_size
    )
    
    train_stats = train_dataset.get_epoch_stats()
    val_stats = val_dataset.get_epoch_stats()
    
    print(f"\n训练集 ({cfg.data.balance_mode}):")
    print(f"  原始: {train_stats['original_real']} 真实 + {train_stats['original_fake']} 伪造")
    print(f"  每 Epoch: {train_stats['total']} 样本")
    
    print(f"\n验证集 (静态):")
    print(f"  固定: {val_stats['real']} 真实 + {val_stats['fake']} 伪造 = {val_stats['total']} 样本")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )
    
    # =========================================================================
    # 2. 模型初始化
    # =========================================================================
    print("\n" + "=" * 70)
    print("初始化模型...")
    print("=" * 70)
    
    # 构建适配器额外参数
    adapter_kwargs = {}
    if hasattr(cfg.model.adapter, 'bottleneck_dim'):
        adapter_kwargs['bottleneck_dim'] = cfg.model.adapter.bottleneck_dim
    if hasattr(cfg.model.adapter, 'dropout'):
        adapter_kwargs['dropout'] = cfg.model.adapter.dropout
    if hasattr(cfg.model.adapter, 'init_scale'):
        adapter_kwargs['init_scale'] = cfg.model.adapter.init_scale
    if hasattr(cfg.model.adapter, 'scale_trainable'):
        adapter_kwargs['scale_trainable'] = cfg.model.adapter.scale_trainable
    
    # 创建模型
    model = DeepfakeDetector(
        device=str(device),
        adapter_layers=cfg.model.adapter.layers,
        adapter_type=adapter_type,
        adapter_kwargs=adapter_kwargs,
    )
    model = model.to(device)
    
    # 创建 EMA 模型
    ema = ModelEMA(model, decay=cfg.ema.decay, device=device) if cfg.ema.enabled else None
    
    total_params = model.count_total_parameters()
    trainable_params = model.count_trainable_parameters()
    
    print(f"\n模型: CLIP ViT-L/14 + {adapter_type}")
    print(f"  输出维度: {model.backbone.output_dim}")
    print(f"  分类头: Linear({model.backbone.output_dim}, 2)")
    print(f"  适配器层: {cfg.model.adapter.layers}")
    print(f"  适配器类型: {adapter_type}")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数量: {trainable_params:,}")
    print(f"  可训练比例: {trainable_params / total_params * 100:.4f}%")
    if cfg.ema.enabled:
        print(f"  EMA Decay: {cfg.ema.decay}")
    
    # =========================================================================
    # 3. 优化器 + 调度器
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )
    
    scheduler = WarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=cfg.train.warmup_epochs,
        total_epochs=cfg.train.epochs,
        min_lr=cfg.optimizer.lr * 0.01
    )
    
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.optimizer.label_smoothing)
    
    print(f"\n优化器: AdamW (lr={cfg.optimizer.lr}, weight_decay={cfg.optimizer.weight_decay})")
    print(f"损失函数: CrossEntropyLoss (label_smoothing={cfg.optimizer.label_smoothing})")
    print(f"调度器: Warmup ({cfg.train.warmup_epochs} epochs) + CosineAnnealing")
    
    # =========================================================================
    # 4. Early Stopping
    # =========================================================================
    early_stopping = EarlyStopping(
        patience=cfg.train.patience,
        mode='max',
        verbose=True
    )
    
    print(f"早停: 监控 Val AUC, patience={cfg.train.patience}")
    
    # =========================================================================
    # 5. WandB
    # =========================================================================
    wandb_run = None
    use_wandb = args.use_wandb or cfg.wandb.enabled
    if use_wandb:
        try:
            wandb_run = init_wandb(cfg, model, total_params, trainable_params)
            print(f"\n✓ WandB: {wandb_run.url}")
        except Exception as e:
            print(f"\n✗ WandB 初始化失败: {e}")
            use_wandb = False
    
    # =========================================================================
    # 6. 断点续训
    # =========================================================================
    start_epoch = 1
    best_auc = 0.0
    best_auc_epoch = 0
    best_acc = 0.0
    best_acc_epoch = 0
    save_dir = Path(cfg.output.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存配置文件到输出目录
    config_save_path = save_dir / "config.yaml"
    save_config(cfg, str(config_save_path))
    print(f"\n配置已保存: {config_save_path}")
    
    if args.resume is not None:
        if Path(args.resume).exists():
            print("\n" + "=" * 70)
            print(f"从 checkpoint 恢复训练: {args.resume}")
            print("=" * 70)
            
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  ✓ 恢复模型权重")
            
            if cfg.ema.enabled and 'ema_state_dict' in checkpoint:
                ema.load_state_dict(checkpoint['ema_state_dict'])
                print("  ✓ 恢复 EMA 权重")
            
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("  ✓ 恢复优化器状态")
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("  ✓ 恢复调度器状态")
            
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                print(f"  ✓ 从 Epoch {start_epoch} 继续训练")
            
            if 'best_auc' in checkpoint:
                best_auc = checkpoint['best_auc']
                best_auc_epoch = checkpoint.get('best_auc_epoch', 0)
                print(f"  ✓ 恢复最佳 AUC: {best_auc:.4f} (Epoch {best_auc_epoch})")
            
            if 'best_acc' in checkpoint:
                best_acc = checkpoint['best_acc']
                best_acc_epoch = checkpoint.get('best_acc_epoch', 0)
                print(f"  ✓ 恢复最佳 ACC: {best_acc*100:.2f}% (Epoch {best_acc_epoch})")
            
            if 'early_stopping_counter' in checkpoint:
                early_stopping.counter = checkpoint['early_stopping_counter']
                early_stopping.best_score = checkpoint.get('early_stopping_best_score', None)
                print(f"  ✓ 恢复 Early Stopping 状态: {early_stopping.counter}/{cfg.train.patience}")
            
            print("=" * 70)
        else:
            print(f"警告: checkpoint 文件不存在: {args.resume}")
    
    # =========================================================================
    # 7. 训练循环
    # =========================================================================
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    
    for epoch in range(start_epoch, cfg.train.epochs + 1):
        print(f"\n{'='*20} Epoch {epoch}/{cfg.train.epochs} {'='*20}")
        
        current_lr = optimizer.param_groups[0]['lr']
        is_warmup = epoch <= cfg.train.warmup_epochs
        print(f"Learning Rate: {current_lr:.6f} {'(Warmup)' if is_warmup else ''}")
        
        # 重新采样训练数据
        train_dataset.resample_for_epoch(epoch)
        epoch_stats = train_dataset.get_epoch_stats()
        print(f"训练采样: {epoch_stats['real']} 真实 + {epoch_stats['fake']} 伪造")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            num_workers=cfg.data.num_workers,
            pin_memory=True
        )
        
        # 训练
        if cfg.ema.enabled:
            train_loss = train_one_epoch(
                model, ema, train_loader, criterion, optimizer, device, epoch, wandb_run
            )
            val_metrics = validate(ema, val_loader, device, epoch)
        else:
            # 如果没有 EMA，创建一个简单的包装器
            class SimpleWrapper:
                def __init__(self, m):
                    self.ema_model = m
                def update(self, m):
                    pass
                def state_dict(self):
                    return {}
            
            wrapper = SimpleWrapper(model)
            train_loss = train_one_epoch(
                model, wrapper, train_loader, criterion, optimizer, device, epoch, wandb_run
            )
            val_metrics = validate(wrapper, val_loader, device, epoch)
        
        scheduler.step()
        
        # 打印结果
        print(f"\n结果:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Metrics:")
        print(f"    Accuracy:  {val_metrics['accuracy'] * 100:.2f}%")
        print(f"    Precision: {val_metrics['precision'] * 100:.2f}%")
        print(f"    Recall:    {val_metrics['recall'] * 100:.2f}%")
        print(f"    F1 Score:  {val_metrics['f1'] * 100:.2f}%")
        print(f"    AUC:       {val_metrics['auc']:.4f}")
        print(f"    AP:        {val_metrics['ap']:.4f}")
        print(f"    EER:       {val_metrics['eer']:.4f}")
        print(f"  混淆矩阵: TP={int(val_metrics['tp'])}, TN={int(val_metrics['tn'])}, "
              f"FP={int(val_metrics['fp'])}, FN={int(val_metrics['fn'])}")
        
        if wandb_run is not None:
            log_to_wandb(wandb_run, epoch, train_loss, val_metrics, current_lr, epoch_stats)
        
        # 保存最佳模型
        val_auc = val_metrics['auc']
        val_acc = val_metrics['accuracy']
        
        # 构建保存的配置字典
        save_config_dict = cfg.to_dict()
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_auc_epoch = epoch
            
            save_path = save_dir / "best_auc_model.pth"
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': {k: v for k, v in val_metrics.items() 
                               if k not in ['all_probs', 'all_labels']},
                'train_loss': train_loss,
                'config': save_config_dict,
                'best_auc': best_auc,
                'best_auc_epoch': best_auc_epoch,
                'best_acc': best_acc,
                'best_acc_epoch': best_acc_epoch,
                'early_stopping_counter': early_stopping.counter,
                'early_stopping_best_score': early_stopping.best_score,
            }
            if cfg.ema.enabled:
                save_dict['ema_state_dict'] = ema.state_dict()
            
            torch.save(save_dict, save_path)
            print(f"  ✓ 保存最佳 AUC 模型 (AUC: {val_auc:.4f})")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_acc_epoch = epoch
            
            save_path = save_dir / "best_acc_model.pth"
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': {k: v for k, v in val_metrics.items() 
                               if k not in ['all_probs', 'all_labels']},
                'train_loss': train_loss,
                'config': save_config_dict,
                'best_auc': best_auc,
                'best_auc_epoch': best_auc_epoch,
                'best_acc': best_acc,
                'best_acc_epoch': best_acc_epoch,
                'early_stopping_counter': early_stopping.counter,
                'early_stopping_best_score': early_stopping.best_score,
            }
            if cfg.ema.enabled:
                save_dict['ema_state_dict'] = ema.state_dict()
            
            torch.save(save_dict, save_path)
            print(f"  ✓ 保存最佳 ACC 模型 (ACC: {val_acc*100:.2f}%)")
        
        # Early Stopping 检查
        if early_stopping(val_auc, epoch):
            print(f"\n{'='*70}")
            print(f"Early Stopping 触发!")
            print(f"最佳 AUC: {best_auc:.4f} (Epoch {best_auc_epoch})")
            print(f"最佳 ACC: {best_acc*100:.2f}% (Epoch {best_acc_epoch})")
            print(f"{'='*70}")
            break
        
        # 定期保存 checkpoint
        if epoch % cfg.train.save_freq == 0:
            ckpt_path = save_dir / f"checkpoint_epoch{epoch:03d}.pth"
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': {k: v for k, v in val_metrics.items() 
                               if k not in ['all_probs', 'all_labels']},
                'config': save_config_dict,
                'best_auc': best_auc,
                'best_auc_epoch': best_auc_epoch,
                'best_acc': best_acc,
                'best_acc_epoch': best_acc_epoch,
                'early_stopping_counter': early_stopping.counter,
                'early_stopping_best_score': early_stopping.best_score,
            }
            if cfg.ema.enabled:
                save_dict['ema_state_dict'] = ema.state_dict()
            
            torch.save(save_dict, ckpt_path)
            print(f"  ✓ 保存 Checkpoint: {ckpt_path.name}")
    
    # 关闭 wandb
    if wandb_run is not None:
        wandb_run.summary['best_auc'] = best_auc
        wandb_run.summary['best_auc_epoch'] = best_auc_epoch
        wandb_run.summary['best_acc'] = best_acc
        wandb_run.summary['best_acc_epoch'] = best_acc_epoch
        wandb_run.finish()
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print(f"最佳 AUC 模型:")
    print(f"  Epoch: {best_auc_epoch}")
    print(f"  AUC: {best_auc:.4f}")
    print(f"  保存位置: {save_dir / 'best_auc_model.pth'}")
    print(f"最佳 ACC 模型:")
    print(f"  Epoch: {best_acc_epoch}")
    print(f"  ACC: {best_acc*100:.2f}%")
    print(f"  保存位置: {save_dir / 'best_acc_model.pth'}")
    print("=" * 70)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Deepfake Detection Training with Mask Support',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 主要参数：YAML 配置文件
    parser.add_argument('--config', type=str, required=True,
                        help='YAML 配置文件路径')
    
    # 可选：命令行覆盖
    parser.add_argument('--device', type=int, default=None,
                        help='覆盖配置中的 GPU ID')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖配置中的 batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='覆盖配置中的学习率')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖配置中的 epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='从 checkpoint 恢复训练')
    parser.add_argument('--use_wandb', action='store_true',
                        help='启用 wandb (覆盖配置)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 加载 YAML 配置
    if not Path(args.config).exists():
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)
    
    cfg = get_config(args.config)
    
    # 命令行参数覆盖配置文件
    if args.device is not None:
        cfg.device = args.device
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.optimizer.lr = args.lr
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    
    # 设置日志
    log_file = setup_logging(cfg.output.log_dir, cfg.experiment.name)
    print(f"PID: {os.getpid()}")
    print(f"配置文件: {args.config}")
    print(f"命令: {' '.join(sys.argv)}")
    
    # 打印完整配置
    print("\n完整配置:")
    print("-" * 40)
    print(cfg)
    print("-" * 40)
    
    # 设置随机种子
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    
    # 检查数据目录
    if not Path(cfg.data.data_root).exists():
        print(f"错误: 数据目录不存在: {cfg.data.data_root}")
        sys.exit(1)
    
    try:
        train(cfg, args)
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("训练被用户中断 (Ctrl+C)")
        print("=" * 70)
    except Exception as e:
        print("\n" + "=" * 70)
        print(f"训练出错: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        raise


if __name__ == "__main__":
    main()
