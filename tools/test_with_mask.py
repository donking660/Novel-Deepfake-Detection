#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Deepfake Detection 统一测试脚本 (带面部掩膜支持)

在 tools/test.py 基础上增加:
  - TestDatasetWithMask: 加载面部掩膜并预计算三区域掩膜
  - 评估时传递 region_masks 到模型

支持 5 个测试集: closed_set, openset1, openset2, openset3, openset4

使用方法:
    python tools/test_with_mask.py --config configs/baseline_region_adapter.yaml --checkpoint path/to/model.pth
    python tools/test_with_mask.py --config configs/baseline_region_adapter.yaml --checkpoint path/to/model.pth --test_set all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from collections import OrderedDict

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
    average_precision_score
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入重构后的模块
from lib.data.transfroms import get_val_transforms, CLIP_MEAN, CLIP_STD
from lib.models.detector import DeepfakeDetector
from lib.utils.config import get_config, Config


# ============================================================================
# 测试集配置
# ============================================================================
TEST_CONFIGS = {
    'closed_set': 'test_closed_set_config.json',
    'openset1': 'test_openset1_config.json',
    'openset2': 'test_openset2_config.json',
    'openset3': 'test_openset3_config.json',
    'openset4': 'test_openset4_config.json',
}


# ============================================================================
# 数据集定义 (带掩膜)
# ============================================================================
class TestDatasetWithMask(Dataset):
    """
    通用测试数据集 (带面部掩膜支持)
    
    在 TestDataset 基础上增加掩膜加载和三区域掩膜预计算。
    
    Args:
        config_path: JSON 配置文件路径
        data_root: 数据集根目录
        transform: 图像预处理变换
        erosion_kernel_size: 腐蚀/膨胀核大小
        feature_map_size: 特征图空间分辨率
    """
    
    def __init__(
        self,
        config_path: str,
        data_root: str,
        transform=None,
        erosion_kernel_size: int = 19,
        feature_map_size: int = 16
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.transform = transform
        self.erosion_kernel_size = erosion_kernel_size
        self.feature_map_size = feature_map_size
        
        # 形态学操作的核
        self.morph_kernel = np.ones(
            (erosion_kernel_size, erosion_kernel_size), dtype=np.uint8
        )
        
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 收集所有样本
        self.samples: List[Tuple[str, int]] = []
        self._collect_real_samples()
        self._collect_fake_samples()
    
    def _collect_real_samples(self):
        """收集真实图像样本 (Label 0)"""
        real_data = self.config.get('real_data', {})
        
        if 'source' in real_data and ('video_ids' in real_data or 'folders' in real_data):
            self._process_source(real_data, label=0)
        else:
            for source_name, source_data in real_data.items():
                if isinstance(source_data, dict) and 'source' in source_data:
                    self._process_source(source_data, label=0)
    
    def _collect_fake_samples(self):
        """收集假图像样本 (Label 1)"""
        fake_data = self.config.get('fake_data', {})
        self._traverse_fake_data(fake_data)
    
    def _traverse_fake_data(self, data: Any):
        """递归遍历假数据配置"""
        if not isinstance(data, dict):
            return
        
        if 'source' in data and ('folders' in data or 'video_ids' in data):
            self._process_source(data, label=1)
        else:
            for key, value in data.items():
                if isinstance(value, dict):
                    self._traverse_fake_data(value)
    
    def _process_source(self, source_data: Dict[str, Any], label: int):
        """处理单个数据来源"""
        source = source_data['source']
        folders = source_data.get('folders', source_data.get('video_ids', []))
        
        for folder_name in folders:
            folder_path = self.data_root / source / folder_name
            if folder_path.exists():
                self._add_images_from_folder(folder_path, label)
    
    def _add_images_from_folder(self, folder_path: Path, label: int):
        """从文件夹中添加所有有效图像"""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
        
        if not folder_path.is_dir():
            return
        
        for img_file in folder_path.iterdir():
            if img_file.suffix.lower() in valid_extensions and img_file.is_file():
                self.samples.append((str(img_file), label))
    
    # =========================================================================
    # 掩膜相关方法 (与 DF40DatasetWithMask 一致)
    # =========================================================================
    def _get_mask_path(self, image_path: str) -> Optional[str]:
        """根据图片路径获取对应的面部掩膜路径"""
        if '/frames/' in image_path:
            mask_path = image_path.replace('/frames/', '/masks/')
            if Path(mask_path).exists():
                return mask_path
        
        path_parts = Path(image_path)
        parent_dir = path_parts.parent
        grandparent_dir = parent_dir.parent
        filename = path_parts.name
        
        mask_folder_name = parent_dir.name + '_masks'
        mask_path = str(grandparent_dir / mask_folder_name / filename)
        if Path(mask_path).exists():
            return mask_path
        
        return None
    
    def _load_face_mask(self, mask_path: Optional[str], target_size: int = 224) -> np.ndarray:
        """加载面部区域二值掩膜"""
        if mask_path is None or not Path(mask_path).exists():
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _compute_region_masks(self, face_mask: np.ndarray) -> np.ndarray:
        """从面部二值掩膜计算三个区域掩膜并下采样到特征图尺寸"""
        mask_eroded = cv2.erode(face_mask, self.morph_kernel, iterations=1)
        mask_dilated = cv2.dilate(face_mask, self.morph_kernel, iterations=1)
        
        m_face = mask_eroded.astype(np.float32) / 255.0
        m_dilated = mask_dilated.astype(np.float32) / 255.0
        m_boundary = m_dilated - m_face
        m_boundary = np.clip(m_boundary, 0, 1)
        m_background = 1.0 - m_dilated
        m_background = np.clip(m_background, 0, 1)
        
        size = self.feature_map_size
        m_face_16 = cv2.resize(m_face, (size, size), interpolation=cv2.INTER_LINEAR)
        m_boundary_16 = cv2.resize(m_boundary, (size, size), interpolation=cv2.INTER_LINEAR)
        m_background_16 = cv2.resize(m_background, (size, size), interpolation=cv2.INTER_LINEAR)
        
        total = m_face_16 + m_boundary_16 + m_background_16 + 1e-8
        m_face_16 = m_face_16 / total
        m_boundary_16 = m_boundary_16 / total
        m_background_16 = m_background_16 / total
        
        region_masks = np.stack([m_face_16, m_boundary_16, m_background_16], axis=0)
        return region_masks.astype(np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        img_path, label = self.samples[idx]
        
        # 加载图像
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"无法读取图像: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # 加载面部掩膜
        mask_path = self._get_mask_path(img_path)
        face_mask = self._load_face_mask(mask_path, target_size=image.shape[0])
        
        # 应用变换 (测试时无数据增强，但仍需 resize + normalize)
        if self.transform:
            augmented = self.transform(image=image, mask=face_mask)
            image = augmented['image']
            face_mask_aug = augmented['mask']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            face_mask_aug = face_mask
        
        # 确保二值化
        if isinstance(face_mask_aug, np.ndarray):
            _, face_mask_aug = cv2.threshold(
                face_mask_aug.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
            )
        else:
            face_mask_aug = face_mask_aug.numpy()
            _, face_mask_aug = cv2.threshold(
                face_mask_aug.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
            )
        
        # 预计算三个区域掩膜
        region_masks = self._compute_region_masks(face_mask_aug)
        region_masks = torch.from_numpy(region_masks)
        
        return image, label, region_masks
    
    def get_stats(self) -> Dict[str, int]:
        """获取数据集统计信息"""
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        return {
            'total': len(self.samples),
            'real': real_count,
            'fake': fake_count
        }


# ============================================================================
# 模型加载
# ============================================================================
def remove_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """移除 DDP 训练产生的 'module.' 前缀"""
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    config: Optional[Config] = None
) -> nn.Module:
    """加载模型权重 (支持区域适配器)"""
    print(f"加载模型权重: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 确定模型配置
    adapter_layers = None
    adapter_type = 'standard'
    adapter_kwargs = {}
    
    # 1. 从 checkpoint 中读取配置
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        saved_config = checkpoint['config']
        if isinstance(saved_config, dict):
            model_config = saved_config.get('model', {})
            adapter_config = model_config.get('adapter', {})
            adapter_layers = adapter_config.get('layers', None)
            adapter_type = adapter_config.get('type', 'standard')
            
            # 读取适配器额外参数
            for key in ['bottleneck_dim', 'dropout', 'init_scale', 'scale_trainable']:
                if key in adapter_config:
                    adapter_kwargs[key] = adapter_config[key]
            
            print(f"  从 checkpoint 读取配置")
    
    # 2. 如果提供了 YAML 配置，覆盖
    if config is not None:
        adapter_layers = config.model.adapter.layers
        adapter_type = getattr(config.model.adapter, 'type', 'standard')
        
        adapter_kwargs = {}
        if hasattr(config.model.adapter, 'bottleneck_dim'):
            adapter_kwargs['bottleneck_dim'] = config.model.adapter.bottleneck_dim
        if hasattr(config.model.adapter, 'dropout'):
            adapter_kwargs['dropout'] = config.model.adapter.dropout
        if hasattr(config.model.adapter, 'init_scale'):
            adapter_kwargs['init_scale'] = config.model.adapter.init_scale
        if hasattr(config.model.adapter, 'scale_trainable'):
            adapter_kwargs['scale_trainable'] = config.model.adapter.scale_trainable
        
        print(f"  使用 YAML 配置")
    
    print(f"  模型配置:")
    print(f"    adapter_layers: {adapter_layers}")
    print(f"    adapter_type: {adapter_type}")
    
    # 创建模型
    model = DeepfakeDetector(
        device=device,
        adapter_layers=adapter_layers,
        adapter_type=adapter_type,
        adapter_kwargs=adapter_kwargs if adapter_kwargs else None,
    )
    
    # 获取 state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    has_module_prefix = any(key.startswith('module.') for key in state_dict.keys())
    if has_module_prefix:
        print("  检测到 DDP 权重，正在移除 'module.' 前缀...")
        state_dict = remove_module_prefix(state_dict)
    
    model.load_state_dict(state_dict, strict=True)
    print("  ✓ 模型权重加载成功！")
    
    model = model.to(device)
    model.eval()
    
    return model


# ============================================================================
# 评估指标计算 (与 test.py 一致)
# ============================================================================
def calculate_metrics(labels: np.ndarray, predictions: np.ndarray) -> Dict[str, Any]:
    """计算评估指标"""
    labels = np.array(labels)
    predictions = np.array(predictions)
    
    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    auc = roc_auc_score(labels, predictions)
    ap = average_precision_score(labels, predictions)
    
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = float(interp1d(fpr, thresholds)(eer))
    except:
        eer = 0.5
        eer_threshold = 0.5
    
    pred_labels_05 = (predictions >= 0.5).astype(int)
    
    acc = accuracy_score(labels, pred_labels_05)
    precision = precision_score(labels, pred_labels_05, pos_label=1, zero_division=0)
    recall = recall_score(labels, pred_labels_05, pos_label=1, zero_division=0)
    f1 = f1_score(labels, pred_labels_05, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(labels, pred_labels_05, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    pred_labels_eer = (predictions >= eer_threshold).astype(int)
    tn_eer, fp_eer, fn_eer, tp_eer = confusion_matrix(labels, pred_labels_eer, labels=[0, 1]).ravel()
    
    far = fn_eer / (fn_eer + tp_eer) if (fn_eer + tp_eer) > 0 else 0.0
    frr = fp_eer / (fp_eer + tn_eer) if (fp_eer + tn_eer) > 0 else 0.0
    hter = (far + frr) / 2
    
    target_fpr = 0.01
    try:
        tpr_at_fpr = float(interp1d(fpr, tpr, kind='linear', fill_value='extrapolate')(target_fpr))
        tpr_at_fpr = np.clip(tpr_at_fpr, 0.0, 1.0)
    except:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at_fpr = tpr[idx]
    
    return {
        'ACC': acc,
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'AUC': auc,
        'AP': ap,
        'EER': eer,
        'HTER': hter,
        'TPR@FPR=1%': tpr_at_fpr,
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
    }


# ============================================================================
# 评估函数
# ============================================================================
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """在测试集上评估模型 (带掩膜支持)"""
    model.eval()
    
    all_labels = []
    all_probs = []
    
    for images, labels, region_masks in tqdm(dataloader, desc="评估进度", ncols=80):
        images = images.to(device)
        region_masks = region_masks.to(device)
        
        logits = model(images, region_masks)
        probs = torch.softmax(logits, dim=1)
        
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs[:, 1].cpu().numpy().tolist())
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    return metrics


def print_metrics(test_name: str, metrics: Dict[str, Any], stats: Dict[str, int]):
    """打印评估结果"""
    print(f"\n{'='*60}")
    print(f"Results for {test_name}")
    print(f"{'='*60}")
    print(f"数据集统计: Real={stats['real']:,}, Fake={stats['fake']:,}, Total={stats['total']:,}")
    print(f"{'-'*60}")
    print(f"主要指标 (阈值=0.5):")
    print(f"  ACC:        {metrics['ACC']*100:.2f}%")
    print(f"  Precision:  {metrics['Precision']*100:.2f}%")
    print(f"  Recall:     {metrics['Recall']*100:.2f}%")
    print(f"  F1 Score:   {metrics['F1']*100:.2f}%")
    print(f"{'-'*60}")
    print(f"概率指标:")
    print(f"  AUC:        {metrics['AUC']:.4f}")
    print(f"  AP:         {metrics['AP']:.4f}")
    print(f"  EER:        {metrics['EER']:.4f} ({metrics['EER']*100:.2f}%)")
    print(f"{'-'*60}")
    print(f"其他指标:")
    print(f"  HTER:       {metrics['HTER']:.4f} ({metrics['HTER']*100:.2f}%)")
    print(f"  TPR@FPR=1%: {metrics['TPR@FPR=1%']:.4f} ({metrics['TPR@FPR=1%']*100:.2f}%)")
    print(f"{'-'*60}")
    print(f"混淆矩阵 (阈值=0.5):")
    print(f"  TP={metrics['TP']:,}, TN={metrics['TN']:,}, FP={metrics['FP']:,}, FN={metrics['FN']:,}")
    print(f"{'='*60}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Deepfake Detection 测试脚本 (带掩膜支持)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型权重文件路径')
    parser.add_argument('--config_dir', type=str,
                        default='/data/zhanzhijie/WORKSPACE_CODE',
                        help='测试配置文件所在目录')
    parser.add_argument('--data_root', type=str,
                        default='/data/zhanzhijie/DATASETS_POOL/deepfakes_detection_datasets',
                        help='数据集根目录')
    parser.add_argument('--test_set', type=str, default='all',
                        choices=['all', 'closed_set', 'openset1', 'openset2', 'openset3', 'openset4'],
                        help='选择测试集')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU 设备 ID')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录')
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = None
    if args.config is not None:
        if not Path(args.config).exists():
            print(f"错误: 配置文件不存在: {args.config}")
            sys.exit(1)
        cfg = get_config(args.config)
        print(f"已加载配置文件: {args.config}")
    
    # 设备
    if torch.cuda.is_available():
        device = f'cuda:{args.device}'
        torch.cuda.set_device(args.device)
    else:
        device = 'cpu'
        print("警告: CUDA 不可用，使用 CPU")
    
    # 输出目录
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    elif cfg is not None:
        output_dir = Path(cfg.output.save_dir) / 'test_results'
    else:
        output_dir = Path('test_results')
    
    # 数据根目录
    data_root = args.data_root
    if cfg is not None and hasattr(cfg, 'data') and hasattr(cfg.data, 'data_root'):
        data_root = cfg.data.data_root
    
    # 分辨率
    resolution = 224
    if cfg is not None and hasattr(cfg, 'data') and hasattr(cfg.data, 'resolution'):
        resolution = cfg.data.resolution
    
    # 掩膜配置
    erosion_kernel_size = 19
    feature_map_size = 16
    if cfg is not None:
        mask_cfg = cfg.get('mask', None)
        if mask_cfg is not None:
            erosion_kernel_size = getattr(mask_cfg, 'erosion_kernel_size', 19)
            feature_map_size = getattr(mask_cfg, 'feature_map_size', 16)
    
    print("\n" + "=" * 60)
    print("Deepfake Detection - 测试脚本 (带掩膜)")
    print("=" * 60)
    if cfg is not None:
        print(f"实验名称: {cfg.experiment.name}")
    print(f"模型权重: {args.checkpoint}")
    print(f"配置目录: {args.config_dir}")
    print(f"数据根目录: {data_root}")
    print(f"测试集: {args.test_set}")
    print(f"设备: {device}")
    print(f"批次大小: {args.batch_size}")
    print(f"腐蚀核大小: {erosion_kernel_size}")
    print(f"特征图尺寸: {feature_map_size}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 检查文件
    if not os.path.exists(args.checkpoint):
        print(f"错误: 模型权重不存在: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(data_root):
        print(f"错误: 数据根目录不存在: {data_root}")
        sys.exit(1)
    
    # 加载模型
    model = load_model(args.checkpoint, device, config=cfg)
    
    # 预处理
    transform = get_val_transforms(config={'resolution': resolution})
    
    # 测试集
    if args.test_set == 'all':
        test_sets = TEST_CONFIGS
    else:
        test_sets = {args.test_set: TEST_CONFIGS[args.test_set]}
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for test_name, config_file in test_sets.items():
        config_path = os.path.join(args.config_dir, config_file)
        
        if not os.path.exists(config_path):
            print(f"\n[{test_name}] 配置文件不存在: {config_path}, 跳过...")
            continue
        
        print(f"\n{'='*60}")
        print(f"正在评估: {test_name}")
        print(f"{'='*60}")
        print(f"配置文件: {config_path}")
        
        # 创建数据集 (带掩膜)
        dataset = TestDatasetWithMask(
            config_path=config_path,
            data_root=data_root,
            transform=transform,
            erosion_kernel_size=erosion_kernel_size,
            feature_map_size=feature_map_size
        )
        
        stats = dataset.get_stats()
        print(f"数据集统计: Real={stats['real']:,}, Fake={stats['fake']:,}, Total={stats['total']:,}")
        
        if stats['total'] == 0:
            print(f"警告: 数据集为空，跳过...")
            continue
        
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if 'cuda' in device else False
        )
        
        # 评估
        metrics = evaluate(model, dataloader, device)
        
        # 打印结果
        print_metrics(test_name, metrics, stats)
        
        # 保存结果
        all_results[test_name] = {
            'stats': stats,
            'metrics': metrics
        }
        
        result_file = output_dir / f'{test_name}_results.json'
        result_data = {
            'test_set': test_name,
            'config_path': config_path,
            'checkpoint': args.checkpoint,
            'stats': stats,
            'metrics': metrics
        }
        if cfg is not None:
            result_data['experiment'] = cfg.experiment.name
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        if 'cuda' in device:
            torch.cuda.empty_cache()
    
    # 汇总结果
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("汇总结果")
        print("=" * 80)
        print(f"{'Test Set':<15} {'ACC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8} {'AUC':>8} {'AP':>8} {'EER':>8}")
        print("-" * 80)
        
        for test_name, result in all_results.items():
            m = result['metrics']
            print(f"{test_name:<15} {m['ACC']*100:>7.2f}% {m['Precision']*100:>7.2f}% {m['Recall']*100:>7.2f}% {m['F1']*100:>7.2f}% {m['AUC']:>8.4f} {m['AP']:>8.4f} {m['EER']*100:>7.2f}%")
        
        print("-" * 80)
        
        avg_metrics = {}
        for key in ['ACC', 'Precision', 'Recall', 'F1', 'AUC', 'AP', 'EER', 'HTER', 'TPR@FPR=1%']:
            avg_metrics[key] = np.mean([r['metrics'][key] for r in all_results.values()])
        
        print(f"{'Average':<15} {avg_metrics['ACC']*100:>7.2f}% {avg_metrics['Precision']*100:>7.2f}% {avg_metrics['Recall']*100:>7.2f}% {avg_metrics['F1']*100:>7.2f}% {avg_metrics['AUC']:>8.4f} {avg_metrics['AP']:>8.4f} {avg_metrics['EER']*100:>7.2f}%")
        print("=" * 80)
    
    # 保存汇总
    summary_file = output_dir / 'summary_results.json'
    summary_data = {
        'checkpoint': args.checkpoint,
        'results': all_results
    }
    if cfg is not None:
        summary_data['experiment'] = cfg.experiment.name
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
