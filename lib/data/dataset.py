"""
Dataset Module (Albumentations 版本)
数据集模块，包含 DF40 数据集加载器

支持：
  - 动态采样训练集（每 epoch 重新采样）
  - 静态验证集（固定 1:1 比例，训练期间不变）
  - 使用 albumentations 进行数据增强 (OpenCV/Numpy 格式)
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.data.transfroms import get_train_transforms, get_val_transforms


class DF40Dataset(Dataset):
    """
    DF40 数据集加载器 (Albumentations 版本)
    
    支持三种平衡模式：
      - 'dynamic': 每个 epoch 动态欠采样伪造样本（训练用）
      - 'static': 固定 1:1 采样，一次性采样后永不改变（验证用）
      - 'none': 不平衡，使用原始数据
    
    图片加载使用 OpenCV (cv2)，与 albumentations 兼容
    
    Args:
        json_path: 配置文件路径
        data_root: 数据集根目录
        transform: albumentations 变换 (A.Compose 对象)
        balance_mode: 平衡模式 ('dynamic', 'static', 'none')
        seed: 随机种子
    """
    
    def __init__(
        self,
        json_path: str,
        data_root: str = "../../DATASETS_POOL/deepfakes_detection_datasets",
        transform: Callable = None,
        balance_mode: str = 'dynamic',
        seed: int = 42
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.transform = transform
        self.balance_mode = balance_mode
        self.seed = seed
        self.rng = random.Random(seed)
        
        # 加载配置文件
        with open(json_path, 'r') as f:
            self.config = json.load(f)
        
        # 分别收集真实和伪造样本
        self.real_samples: List[Tuple[str, int]] = []
        self.fake_samples: List[Tuple[str, int]] = []
        self.num_fake_methods = 0
        
        # 收集样本
        self._collect_real_samples()
        self._collect_fake_samples()
        
        # 当前使用的样本列表
        self.samples: List[Tuple[str, int]] = []
        
        # 根据平衡模式构建数据集
        self._build_dataset()
        
        # 打印统计信息
        self._print_stats()
    
    def _collect_real_samples(self):
        """收集真实图像样本"""
        real_data = self.config['real_data']
        source = real_data['source']
        video_ids = real_data['video_ids']
        
        for video_id in video_ids:
            folder_path = self.data_root / source / video_id
            if folder_path.exists():
                self._add_images_from_folder(folder_path, label=0)
    
    def _collect_fake_samples(self):
        """收集伪造图像样本"""
        fake_data = self.config['fake_data']
        self._traverse_fake_data(fake_data)
    
    def _traverse_fake_data(self, data: Dict[str, Any]):
        """递归遍历 fake_data 字典"""
        if isinstance(data, dict):
            if 'source' in data and 'folders' in data:
                source = data['source']
                folders = data['folders']
                self.num_fake_methods += 1
                
                for folder_name in folders:
                    folder_path = self.data_root / source / folder_name
                    if folder_path.exists():
                        self._add_images_from_folder(folder_path, label=1)
            else:
                for key, value in data.items():
                    self._traverse_fake_data(value)
    
    def _add_images_from_folder(self, folder_path: Path, label: int):
        """从文件夹中添加图像到样本列表"""
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        
        if folder_path.is_dir():
            for img_file in folder_path.iterdir():
                if img_file.suffix.lower() in valid_extensions:
                    sample = (str(img_file), label)
                    if label == 0:
                        self.real_samples.append(sample)
                    else:
                        self.fake_samples.append(sample)
    
    def _build_dataset(self):
        """根据平衡模式构建数据集"""
        if self.balance_mode == 'dynamic':
            # 动态采样模式：初始化时先构建一次
            self.resample_for_epoch(epoch=0)
            
        elif self.balance_mode == 'static':
            # 静态采样模式：一次性 1:1 采样，永不改变
            self._build_static_balanced_dataset()
            
        else:  # 'none'
            # 不平衡模式：直接合并
            self.samples = self.real_samples + self.fake_samples
            self.rng.shuffle(self.samples)
    
    def _build_static_balanced_dataset(self):
        """
        构建静态平衡数据集（验证集专用）
        
        策略：
        1. 使用所有真实样本
        2. 从伪造样本中随机抽取等量样本
        3. 永久固定，训练期间不再改变
        """
        # 使用固定种子确保可复现
        static_rng = random.Random(self.seed)
        
        num_real = len(self.real_samples)
        
        # 从伪造样本中随机抽取等量样本
        if len(self.fake_samples) >= num_real:
            sampled_fake = static_rng.sample(self.fake_samples, num_real)
        else:
            sampled_fake = self.fake_samples.copy()
        
        # 合并并打乱
        self.samples = self.real_samples.copy() + sampled_fake
        static_rng.shuffle(self.samples)
        
        # 标记为已固定
        self._is_static_fixed = True
    
    def resample_for_epoch(self, epoch: int):
        """
        为新的 epoch 重新采样数据（仅 dynamic 模式有效）
        
        Args:
            epoch: 当前 epoch 编号
        """
        if self.balance_mode != 'dynamic':
            return
        
        # 设置随机种子
        epoch_seed = self.seed + epoch
        self.rng = random.Random(epoch_seed)
        
        num_real = len(self.real_samples)
        
        # 从伪造样本中随机抽取等量样本
        if len(self.fake_samples) >= num_real:
            sampled_fake = self.rng.sample(self.fake_samples, num_real)
        else:
            sampled_fake = self.fake_samples.copy()
        
        # 合并并打乱
        self.samples = self.real_samples.copy() + sampled_fake
        self.rng.shuffle(self.samples)
    
    def _print_stats(self):
        """打印数据集统计信息"""
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        
        print(f"数据集加载完成:")
        print(f"  原始真实样本: {len(self.real_samples)}")
        print(f"  原始伪造样本: {len(self.fake_samples)}")
        print(f"  伪造方法数量: {self.num_fake_methods}")
        print(f"  平衡模式: {self.balance_mode}")
        print(f"  当前样本:")
        print(f"    真实: {real_count}")
        print(f"    伪造: {fake_count}")
        print(f"    总计: {len(self.samples)}")
        if real_count > 0:
            print(f"    比例: 1:{fake_count / real_count:.2f}")
    
    def get_epoch_stats(self) -> Dict[str, int]:
        """获取当前样本统计"""
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        return {
            'total': len(self.samples),
            'real': real_count,
            'fake': fake_count,
            'original_real': len(self.real_samples),
            'original_fake': len(self.fake_samples)
        }
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        使用 OpenCV 加载图像并转换为 RGB 格式
        
        Args:
            img_path: 图像路径
            
        Returns:
            RGB 格式的 numpy 数组 (H, W, 3)
        """
        # 使用 OpenCV 读取图像 (默认 BGR 格式)
        image = cv2.imread(img_path)
        
        if image is None:
            # 图像加载失败，返回黑色图像 (CLIP ViT-L/14 需要 224x224)
            print(f"警告: 无法加载图像: {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取数据样本
        
        使用 OpenCV 加载图像，应用 albumentations 变换
        
        Args:
            idx: 样本索引
            
        Returns:
            (image_tensor, label) 元组
            - image_tensor: (C, H, W) 格式的 Tensor
            - label: 0 表示真实，1 表示伪造
        """
        img_path, label = self.samples[idx]
        
        # 使用 OpenCV 加载图像 (RGB 格式的 numpy 数组)
        image = self._load_image(img_path)
        
        # 应用 albumentations 变换
        if self.transform is not None:
            # albumentations 返回字典格式
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            # 如果没有 transform，手动转换为 Tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        return image, label


def create_train_val_datasets(
    json_path: str,
    data_root: str,
    train_transform: Callable = None,
    val_transform: Callable = None,
    seed: int = 42
) -> Tuple[DF40Dataset, DF40Dataset]:
    """
    创建训练集和验证集
    
    训练集：动态采样模式（每 epoch 重新采样）
    验证集：静态采样模式（1:1 固定，永不改变）
    
    Args:
        json_path: 配置文件路径
        data_root: 数据集根目录
        train_transform: 训练数据增强 (albumentations)
        val_transform: 验证数据预处理 (albumentations)
        seed: 随机种子
    
    Returns:
        (train_dataset, val_dataset) 元组
    """
    # 训练集：动态采样
    train_dataset = DF40Dataset(
        json_path=json_path,
        data_root=data_root,
        transform=train_transform,
        balance_mode='dynamic',
        seed=seed
    )
    
    # 验证集：静态平衡（使用不同的种子以获得不同的样本）
    val_dataset = DF40Dataset(
        json_path=json_path,
        data_root=data_root,
        transform=val_transform,
        balance_mode='static',
        seed=seed + 10000  # 使用不同种子确保验证集样本与训练集有所不同
    )
    
    return train_dataset, val_dataset
