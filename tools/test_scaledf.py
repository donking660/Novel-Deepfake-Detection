#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ScaleDF Deepfake Detection 测试脚本

在 ScaleDF 独立测试集上评估模型性能

测试集结构:
    ScaleDF/val/
        000000Xxx/   -> 真实人脸 (label=0)
        OtherXxx/    -> 伪造人脸 (label=1)

评估指标: AUC, ACC, AP, EER, Precision, Recall, F1, 混淆矩阵

使用方法:
    python tools/test_scaledf.py --config configs/scaledf_baseline.yaml

    # 指定权重 / GPU / batch_size
    python tools/test_scaledf.py --config configs/scaledf_baseline.yaml \
        --checkpoint checkpoints/scaledf_baseline/best_auc_model.pth \
        --device 0 --batch_size 128
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.data.transfroms import get_scaledf_val_transforms
from lib.models.detector import DeepfakeDetector
from lib.utils.config import get_config, Config

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}


# ============================================================================
# 测试数据集
# ============================================================================
class ScaleDFTestDataset(Dataset):
    """
    ScaleDF 独立测试集加载器

    扫描 data_root 下全部子文件夹, 000000 开头为真实 (label=0), 其余为伪造 (label=1).
    加载所有图像, 无 train/val 拆分, 无子采样.

    Args:
        data_root: 测试集根目录 (ScaleDF/val/)
        transform: albumentations 变换
    """

    def __init__(self, data_root: str, transform=None):
        super().__init__()
        self.data_root = str(data_root)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.real_count = 0
        self.fake_count = 0
        self.method_stats: Dict[str, Dict] = {}

        self._collect_samples()

    def _collect_samples(self):
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"测试集目录不存在: {self.data_root}")

        subdirs = sorted(
            d for d in os.listdir(self.data_root)
            if os.path.isdir(os.path.join(self.data_root, d))
        )

        print(f"  扫描测试集目录: {self.data_root}")
        print(f"  子文件夹数量: {len(subdirs)}")

        for i, folder_name in enumerate(subdirs):
            folder_path = os.path.join(self.data_root, folder_name)
            label = 0 if folder_name.startswith('000000') else 1

            filenames = sorted(
                fn for fn in os.listdir(folder_path)
                if os.path.splitext(fn)[1].lower() in VALID_EXTENSIONS
            )

            for fn in filenames:
                self.samples.append((os.path.join(folder_path, fn), label))

            count = len(filenames)
            if label == 0:
                self.real_count += count
            else:
                self.fake_count += count

            self.method_stats[folder_name] = {
                'label': 'real' if label == 0 else 'fake',
                'count': count,
            }

            if (i + 1) % 50 == 0 or (i + 1) == len(subdirs):
                print(f"    [{i + 1}/{len(subdirs)}] 已收集 {len(self.samples)} 样本 "
                      f"(真实: {self.real_count}, 伪造: {self.fake_count})")

        print(f"\n  测试集加载完成:")
        print(f"    真实: {self.real_count}")
        print(f"    伪造: {self.fake_count}")
        print(f"    总计: {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total': len(self.samples),
            'real': self.real_count,
            'fake': self.fake_count,
        }


# ============================================================================
# 模型加载
# ============================================================================
def remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith('module.') else key
        new_state_dict[new_key] = value
    return new_state_dict


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    config: Optional[Config] = None,
) -> nn.Module:
    """
    从 checkpoint 加载模型, 优先使用 EMA 权重.
    """
    print(f"加载模型权重: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    adapter_layers = None
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        saved_cfg = checkpoint['config']
        if isinstance(saved_cfg, dict):
            adapter_layers = saved_cfg.get('model', {}).get('adapter', {}).get('layers')
            print(f"  从 checkpoint 读取模型配置")

    if config is not None:
        adapter_layers = config.model.adapter.layers
        print(f"  使用 YAML 配置覆盖")

    print(f"  adapter_layers: {adapter_layers}")

    model = DeepfakeDetector(device=device, adapter_layers=adapter_layers)

    # 优先使用 EMA 权重
    if isinstance(checkpoint, dict) and 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        print(f"  使用 EMA 权重")
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"  使用原始模型权重 (无 EMA)")
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = remove_module_prefix(state_dict)
        print(f"  已移除 'module.' 前缀")

    model.load_state_dict(state_dict, strict=True)
    print(f"  模型权重加载成功")

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get('epoch', '?')
        val_metrics = checkpoint.get('val_metrics', {})
        best_auc = checkpoint.get('best_auc', val_metrics.get('auc', '?'))
        print(f"  来自 Epoch {epoch}, 训练时最佳 AUC: {best_auc}")

    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# 指标计算
# ============================================================================
def compute_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, Any]:
    """
    计算全部评估指标.

    Args:
        labels: 真实标签 (0=Real, 1=Fake)
        probs: Fake 类预测概率

    Returns:
        指标字典
    """
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)

    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fpr - fnr))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)

    pred_labels = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, pred_labels)
    prec = precision_score(labels, pred_labels, pos_label=1, zero_division=0)
    rec = recall_score(labels, pred_labels, pos_label=1, zero_division=0)
    f1 = f1_score(labels, pred_labels, pos_label=1, zero_division=0)

    cm = confusion_matrix(labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        'AUC': float(auc),
        'ACC': float(acc),
        'AP': float(ap),
        'EER': float(eer),
        'Precision': float(prec),
        'Recall': float(rec),
        'F1': float(f1),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
    }


def compute_per_method_metrics(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    all_methods: List[str],
) -> Dict[str, Dict[str, float]]:
    """按伪造方法分别计算指标 (仅 Fake vs Real 的 AUC/ACC)"""
    method_set = sorted(set(all_methods))
    real_mask = (all_labels == 0)
    real_labels = all_labels[real_mask]
    real_probs = all_probs[real_mask]

    results = {}
    for method in method_set:
        if method.startswith('000000'):
            continue
        method_mask = np.array([m == method for m in all_methods])
        fake_labels = all_labels[method_mask]
        fake_probs = all_probs[method_mask]

        combined_labels = np.concatenate([real_labels, fake_labels])
        combined_probs = np.concatenate([real_probs, fake_probs])

        if len(np.unique(combined_labels)) < 2:
            continue

        try:
            auc = roc_auc_score(combined_labels, combined_probs)
            pred = (combined_probs >= 0.5).astype(int)
            acc = accuracy_score(combined_labels, pred)
            results[method] = {'AUC': float(auc), 'ACC': float(acc), 'count': int(fake_labels.sum() + (1 - fake_labels).sum())}
        except Exception:
            pass

    return results


# ============================================================================
# 评估
# ============================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Testing", ncols=100)
    for images, labels in pbar:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, 1]

        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())

        current_total = len(all_labels)
        pbar.set_postfix({'samples': current_total})

    return np.array(all_labels), np.array(all_probs)


# ============================================================================
# 混淆矩阵可视化
# ============================================================================
def plot_confusion_matrix(metrics: Dict, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = np.array([
        [metrics['TN'], metrics['FP']],
        [metrics['FN'], metrics['TP']],
    ], dtype=int)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Real (Pred)', 'Fake (Pred)'],
        yticklabels=['Real (True)', 'Fake (True)'],
        annot_kws={'size': 16},
    )
    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title('Confusion Matrix', fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  混淆矩阵已保存: {save_path}")


def plot_roc_curve(labels: np.ndarray, probs: np.ndarray, auc_val: float, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_val:.4f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title('ROC Curve', fontsize=15)
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ROC 曲线已保存: {save_path}")


# ============================================================================
# 打印结果
# ============================================================================
def print_results(metrics: Dict, stats: Dict):
    print(f"\n{'=' * 70}")
    print(f"ScaleDF 测试集评估结果")
    print(f"{'=' * 70}")
    print(f"数据集: Real={stats['real']:,}, Fake={stats['fake']:,}, Total={stats['total']:,}")
    print(f"{'-' * 70}")
    print(f"分类指标 (阈值=0.5):")
    print(f"  ACC:        {metrics['ACC'] * 100:.2f}%")
    print(f"  Precision:  {metrics['Precision'] * 100:.2f}%")
    print(f"  Recall:     {metrics['Recall'] * 100:.2f}%")
    print(f"  F1 Score:   {metrics['F1'] * 100:.2f}%")
    print(f"{'-' * 70}")
    print(f"概率指标:")
    print(f"  AUC:        {metrics['AUC']:.4f}")
    print(f"  AP:         {metrics['AP']:.4f}")
    print(f"  EER:        {metrics['EER']:.4f} ({metrics['EER'] * 100:.2f}%)")
    print(f"{'-' * 70}")
    print(f"混淆矩阵 (阈值=0.5):")
    print(f"                Pred Real    Pred Fake")
    print(f"  True Real     {metrics['TN']:>8,}    {metrics['FP']:>8,}")
    print(f"  True Fake     {metrics['FN']:>8,}    {metrics['TP']:>8,}")
    print(f"{'=' * 70}")


# ============================================================================
# 主函数
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='ScaleDF Deepfake Detection 测试脚本',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径 (如 configs/scaledf_baseline.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型权重路径 (默认从配置推断)')
    parser.add_argument('--test_root', type=str,
                        default='/data/zhanzhijie/DATASETS_POOL/deepfakes_detection_datasets/ScaleDF/val',
                        help='ScaleDF 测试集根目录')
    parser.add_argument('--device', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录 (默认: checkpoints/<exp>/test_results)')
    parser.add_argument('--no_ema', action='store_true',
                        help='强制使用原始模型权重而非 EMA')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置
    cfg = None
    if args.config is not None:
        if not Path(args.config).exists():
            print(f"错误: 配置文件不存在: {args.config}")
            sys.exit(1)
        cfg = get_config(args.config)

    # 确定权重路径
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    elif cfg is not None:
        checkpoint_path = str(Path(cfg.output.save_dir) / 'best_auc_model.pth')
    else:
        print("错误: 请通过 --checkpoint 或 --config 指定权重路径")
        sys.exit(1)

    if not Path(checkpoint_path).exists():
        print(f"错误: 权重文件不存在: {checkpoint_path}")
        sys.exit(1)

    # 设备
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # 输出目录
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif cfg is not None:
        output_dir = Path(cfg.output.save_dir) / 'test_results_scaledf'
    else:
        output_dir = Path('test_results_scaledf')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 分辨率
    resolution = 224
    if cfg is not None and hasattr(cfg, 'data') and hasattr(cfg.data, 'resolution'):
        resolution = cfg.data.resolution

    # 打印实验信息
    print("=" * 70)
    print("ScaleDF Deepfake Detection - 测试脚本")
    print("=" * 70)
    if cfg is not None:
        print(f"  实验名称:   {cfg.experiment.name}")
    print(f"  权重文件:   {checkpoint_path}")
    print(f"  测试集目录: {args.test_root}")
    print(f"  设备:       {device}")
    print(f"  批次大小:   {args.batch_size}")
    print(f"  分辨率:     {resolution}")
    print(f"  输出目录:   {output_dir}")
    print("=" * 70)

    # =========================================================================
    # 1. 加载模型
    # =========================================================================
    print("\n加载模型...")

    if args.no_ema:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        adapter_layers = None
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            adapter_layers = checkpoint['config'].get('model', {}).get('adapter', {}).get('layers')
        if cfg is not None:
            adapter_layers = cfg.model.adapter.layers

        model = DeepfakeDetector(device=device, adapter_layers=adapter_layers)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = remove_module_prefix(state_dict)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(device)
        model.eval()
        print("  使用原始模型权重 (--no_ema)")
    else:
        model = load_model(checkpoint_path, device, config=cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {total_params:,}")

    # =========================================================================
    # 2. 加载测试集
    # =========================================================================
    print("\n加载测试集...")
    transform = get_scaledf_val_transforms(config={'resolution': resolution})

    test_dataset = ScaleDFTestDataset(
        data_root=args.test_root,
        transform=transform,
    )

    stats = test_dataset.get_stats()
    if stats['total'] == 0:
        print("错误: 测试集为空!")
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if 'cuda' in device else False,
    )

    # =========================================================================
    # 3. 推理
    # =========================================================================
    print(f"\n开始评估 ({stats['total']:,} 样本)...")
    all_labels, all_probs = evaluate(model, test_loader, device)

    # =========================================================================
    # 4. 计算指标
    # =========================================================================
    metrics = compute_metrics(all_labels, all_probs)
    print_results(metrics, stats)

    # =========================================================================
    # 5. 按伪造方法分析
    # =========================================================================
    method_list = []
    idx = 0
    for folder_name in sorted(test_dataset.method_stats.keys()):
        info = test_dataset.method_stats[folder_name]
        method_list.extend([folder_name] * info['count'])

    if len(method_list) == len(all_labels):
        per_method = compute_per_method_metrics(all_labels, all_probs, method_list)
        if per_method:
            print(f"\n{'=' * 70}")
            print(f"各伪造方法性能:")
            print(f"{'=' * 70}")
            print(f"{'Method':<30} {'AUC':>8} {'ACC':>8} {'Samples':>8}")
            print(f"{'-' * 70}")
            for method_name in sorted(per_method.keys()):
                m = per_method[method_name]
                print(f"{method_name:<30} {m['AUC']:>8.4f} {m['ACC']*100:>7.2f}% {m['count']:>8,}")
            print(f"{'=' * 70}")

    # =========================================================================
    # 6. 保存结果
    # =========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        'experiment': cfg.experiment.name if cfg else 'unknown',
        'checkpoint': checkpoint_path,
        'test_root': args.test_root,
        'timestamp': timestamp,
        'device': device,
        'stats': stats,
        'metrics': metrics,
    }
    if len(method_list) == len(all_labels) and 'per_method' in dir():
        result_data['per_method'] = per_method

    result_file = output_dir / f'scaledf_test_results.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {result_file}")

    # 混淆矩阵图
    try:
        cm_path = str(output_dir / 'confusion_matrix.png')
        plot_confusion_matrix(metrics, cm_path)
    except Exception as e:
        print(f"  混淆矩阵绘图失败: {e}")

    # ROC 曲线图
    try:
        roc_path = str(output_dir / 'roc_curve.png')
        plot_roc_curve(all_labels, all_probs, metrics['AUC'], roc_path)
    except Exception as e:
        print(f"  ROC 曲线绘图失败: {e}")

    # 保存预测结果 (可选, 用于后续分析)
    np.savez_compressed(
        str(output_dir / 'predictions.npz'),
        labels=all_labels,
        probs=all_probs,
    )
    print(f"  预测结果已保存: {output_dir / 'predictions.npz'}")

    print(f"\n全部结果已保存到: {output_dir}")

    if 'cuda' in device:
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
