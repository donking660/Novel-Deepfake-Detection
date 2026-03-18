"""
Dataset Module with Face Mask Support (Albumentations 版本)
数据集模块，支持面部区域掩膜加载和预计算三区域掩膜

在原有 DF40Dataset 基础上增加:
  - 加载面部区域二值掩膜 (face region mask)
  - 通过腐蚀/膨胀预计算三个区域掩膜:
    1. 人脸内部 (face): 腐蚀后的面部核心区域
    2. 边界 (boundary): 膨胀 - 腐蚀 的过渡带
    3. 背景 (background): 膨胀后的外部区域
  - 下采样到 16x16 并归一化，确保 M1 + M2 + M3 = 1
  - 掩膜与图像同步数据增强
"""

import json
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from lib.data.transfroms import get_train_transforms, get_val_transforms


class DF40DatasetWithMask(Dataset):
    """
    DF40 数据集加载器 (带面部掩膜支持)
    
    在 DF40Dataset 基础上，额外加载面部区域掩膜并预计算三个区域掩膜。
    
    掩膜路径规则 (与 ForensicsAdapter 一致):
      1. face_swapping 格式: /frames/ -> /masks/
      2. entire_face_synthesis 格式: folder -> folder_masks
      3. 真实图片: 同样有对应的面部掩膜
    
    三个区域掩膜的计算:
      - M_eroded = erode(face_mask, kernel=19x19)  → 人脸内部
      - M_dilated = dilate(face_mask, kernel=19x19) → 扩展面部区域
      - face = M_eroded
      - boundary = M_dilated - M_eroded
      - background = 1 - M_dilated
      
    下采样到 16x16 后归一化确保 M1 + M2 + M3 = 1
    
    Args:
        json_path: 配置文件路径
        data_root: 数据集根目录
        transform: albumentations 变换 (A.Compose 对象)
        balance_mode: 平衡模式 ('dynamic', 'static', 'none')
        seed: 随机种子
        erosion_kernel_size: 腐蚀/膨胀核大小，默认 19 (每侧 9 像素)
        feature_map_size: 特征图空间分辨率，默认 16 (ViT-L/14: 224/14=16)
    """
    
    def __init__(
        self,
        json_path: str,
        data_root: str = "../../DATASETS_POOL/deepfakes_detection_datasets",
        transform: Callable = None,
        balance_mode: str = 'dynamic',
        seed: int = 42,
        erosion_kernel_size: int = 19,
        feature_map_size: int = 16
    ):
        super().__init__()
        
        self.data_root = Path(data_root)
        self.transform = transform
        self.balance_mode = balance_mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.erosion_kernel_size = erosion_kernel_size
        self.feature_map_size = feature_map_size
        
        # 形态学操作的核
        self.morph_kernel = np.ones(
            (erosion_kernel_size, erosion_kernel_size), dtype=np.uint8
        )
        
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
    
    # =========================================================================
    # 样本收集 (与 DF40Dataset 一致)
    # =========================================================================
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
    
    # =========================================================================
    # 数据集构建 (与 DF40Dataset 一致)
    # =========================================================================
    def _build_dataset(self):
        """根据平衡模式构建数据集"""
        if self.balance_mode == 'dynamic':
            self.resample_for_epoch(epoch=0)
        elif self.balance_mode == 'static':
            self._build_static_balanced_dataset()
        else:  # 'none'
            self.samples = self.real_samples + self.fake_samples
            self.rng.shuffle(self.samples)
    
    def _build_static_balanced_dataset(self):
        """构建静态平衡数据集（验证集专用）"""
        static_rng = random.Random(self.seed)
        num_real = len(self.real_samples)
        
        if len(self.fake_samples) >= num_real:
            sampled_fake = static_rng.sample(self.fake_samples, num_real)
        else:
            sampled_fake = self.fake_samples.copy()
        
        self.samples = self.real_samples.copy() + sampled_fake
        static_rng.shuffle(self.samples)
        self._is_static_fixed = True
    
    def resample_for_epoch(self, epoch: int):
        """为新的 epoch 重新采样数据（仅 dynamic 模式有效）"""
        if self.balance_mode != 'dynamic':
            return
        
        epoch_seed = self.seed + epoch
        self.rng = random.Random(epoch_seed)
        
        num_real = len(self.real_samples)
        
        if len(self.fake_samples) >= num_real:
            sampled_fake = self.rng.sample(self.fake_samples, num_real)
        else:
            sampled_fake = self.fake_samples.copy()
        
        self.samples = self.real_samples.copy() + sampled_fake
        self.rng.shuffle(self.samples)
    
    def _print_stats(self):
        """打印数据集统计信息"""
        real_count = sum(1 for _, label in self.samples if label == 0)
        fake_count = sum(1 for _, label in self.samples if label == 1)
        
        print(f"数据集加载完成 (带掩膜):")
        print(f"  原始真实样本: {len(self.real_samples)}")
        print(f"  原始伪造样本: {len(self.fake_samples)}")
        print(f"  伪造方法数量: {self.num_fake_methods}")
        print(f"  平衡模式: {self.balance_mode}")
        print(f"  腐蚀/膨胀核大小: {self.erosion_kernel_size}")
        print(f"  特征图尺寸: {self.feature_map_size}x{self.feature_map_size}")
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
    
    # =========================================================================
    # 掩膜相关方法
    # =========================================================================
    def _get_mask_path(self, image_path: str) -> Optional[str]:
        """
        根据图片路径获取对应的面部掩膜路径
        
        路径规则与 ForensicsAdapter 一致:
          1. face_swapping 格式: /frames/ -> /masks/
          2. entire_face_synthesis 格式: folder -> folder_masks
          3. 真实图片也有对应掩膜
        
        Args:
            image_path: 图片路径
            
        Returns:
            mask 路径，如果找不到返回 None
        """
        # 方式1: face_swapping 格式 (frames -> masks)
        if '/frames/' in image_path:
            mask_path = image_path.replace('/frames/', '/masks/')
            if Path(mask_path).exists():
                return mask_path
        
        # 方式2: entire_face_synthesis 格式 (folder -> folder_masks)
        path_parts = Path(image_path)
        parent_dir = path_parts.parent
        grandparent_dir = parent_dir.parent
        filename = path_parts.name
        
        mask_folder_name = parent_dir.name + '_masks'
        mask_path = str(grandparent_dir / mask_folder_name / filename)
        if Path(mask_path).exists():
            return mask_path
        
        # 找不到掩膜
        return None
    
    def _load_face_mask(self, mask_path: Optional[str], target_size: int = 224) -> np.ndarray:
        """
        加载面部区域二值掩膜
        
        Args:
            mask_path: 掩膜路径，None 则返回全零掩膜
            target_size: 目标尺寸 (与图像一致)
        
        Returns:
            二值掩膜 (H, W)，值为 0 或 255，dtype=uint8
        """
        if mask_path is None or not Path(mask_path).exists():
            # 没有掩膜时返回全零 (全背景)
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((target_size, target_size), dtype=np.uint8)
        
        # resize 到目标尺寸
        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        
        # 确保二值化
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        return mask
    
    def _compute_region_masks(self, face_mask: np.ndarray) -> np.ndarray:
        """
        从面部二值掩膜计算三个区域掩膜，并下采样到特征图尺寸
        
        计算步骤:
          1. 在 224x224 分辨率下腐蚀/膨胀
          2. 得到 face, boundary, background 三个区域
          3. 下采样到 16x16
          4. 归一化确保 M1 + M2 + M3 = 1
        
        Args:
            face_mask: 面部二值掩膜 (H, W)，值为 0 或 255，dtype=uint8
        
        Returns:
            region_masks: 三个区域掩膜 (3, feature_map_size, feature_map_size)，
                          float32，归一化后满足 M1+M2+M3=1
                          channel 0: face (人脸内部)
                          channel 1: boundary (边界)
                          channel 2: background (背景)
        """
        # 1. 腐蚀和膨胀 (在 224x224 分辨率下)
        mask_eroded = cv2.erode(face_mask, self.morph_kernel, iterations=1)
        mask_dilated = cv2.dilate(face_mask, self.morph_kernel, iterations=1)
        
        # 2. 计算三个区域 (float32, 0~1)
        m_face = mask_eroded.astype(np.float32) / 255.0        # 人脸内部
        m_dilated = mask_dilated.astype(np.float32) / 255.0
        m_eroded = mask_eroded.astype(np.float32) / 255.0
        m_boundary = m_dilated - m_eroded                       # 边界区域
        m_boundary = np.clip(m_boundary, 0, 1)                  # 防止数值误差
        m_background = 1.0 - m_dilated                          # 背景区域
        m_background = np.clip(m_background, 0, 1)
        
        # 3. 下采样到 feature_map_size x feature_map_size
        size = self.feature_map_size
        m_face_16 = cv2.resize(m_face, (size, size), interpolation=cv2.INTER_LINEAR)
        m_boundary_16 = cv2.resize(m_boundary, (size, size), interpolation=cv2.INTER_LINEAR)
        m_background_16 = cv2.resize(m_background, (size, size), interpolation=cv2.INTER_LINEAR)
        
        # 4. 归一化确保 M1 + M2 + M3 = 1
        total = m_face_16 + m_boundary_16 + m_background_16 + 1e-8
        m_face_16 = m_face_16 / total
        m_boundary_16 = m_boundary_16 / total
        m_background_16 = m_background_16 / total
        
        # 5. 组合成 (3, H, W)
        region_masks = np.stack([m_face_16, m_boundary_16, m_background_16], axis=0)
        
        return region_masks.astype(np.float32)
    
    # =========================================================================
    # 图像加载
    # =========================================================================
    def _load_image(self, img_path: str) -> np.ndarray:
        """
        使用 OpenCV 加载图像并转换为 RGB 格式
        
        Args:
            img_path: 图像路径
            
        Returns:
            RGB 格式的 numpy 数组 (H, W, 3)
        """
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"警告: 无法加载图像: {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    
    # =========================================================================
    # __getitem__
    # =========================================================================
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        获取数据样本
        
        Returns:
            (image_tensor, label, region_masks) 元组
            - image_tensor: (C, H, W) 格式的 Tensor
            - label: 0 表示真实，1 表示伪造
            - region_masks: (3, 16, 16) 的 Tensor，三个区域掩膜
        """
        img_path, label = self.samples[idx]
        
        # 加载图像
        image = self._load_image(img_path)
        
        # 加载面部掩膜
        mask_path = self._get_mask_path(img_path)
        face_mask = self._load_face_mask(mask_path, target_size=image.shape[0])
        
        # 应用数据增强（图像和掩膜同步变换）
        if self.transform is not None:
            # albumentations 支持同步变换 image 和 mask
            augmented = self.transform(image=image, mask=face_mask)
            image = augmented['image']             # 已经是 Tensor (C, H, W)
            face_mask_aug = augmented['mask']       # numpy (H, W)，经过同步空间变换
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            face_mask_aug = face_mask
        
        # 确保增强后的掩膜仍然是二值的
        # (旋转等操作可能引入插值中间值)
        if isinstance(face_mask_aug, np.ndarray):
            _, face_mask_aug = cv2.threshold(
                face_mask_aug.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
            )
        else:
            # 如果是 Tensor，转回 numpy
            face_mask_aug = face_mask_aug.numpy()
            _, face_mask_aug = cv2.threshold(
                face_mask_aug.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
            )
        
        # 预计算三个区域掩膜 (在 CPU 上，使用 numpy/OpenCV)
        region_masks = self._compute_region_masks(face_mask_aug)
        
        # 转换为 Tensor
        region_masks = torch.from_numpy(region_masks)  # (3, 16, 16)
        
        return image, label, region_masks


def create_train_val_datasets_with_mask(
    json_path: str,
    data_root: str,
    train_transform: Callable = None,
    val_transform: Callable = None,
    seed: int = 42,
    erosion_kernel_size: int = 19,
    feature_map_size: int = 16
) -> Tuple[DF40DatasetWithMask, DF40DatasetWithMask]:
    """
    创建带掩膜的训练集和验证集
    
    Args:
        json_path: 配置文件路径
        data_root: 数据集根目录
        train_transform: 训练数据增强
        val_transform: 验证数据预处理
        seed: 随机种子
        erosion_kernel_size: 腐蚀/膨胀核大小
        feature_map_size: 特征图空间分辨率
    
    Returns:
        (train_dataset, val_dataset) 元组
    """
    train_dataset = DF40DatasetWithMask(
        json_path=json_path,
        data_root=data_root,
        transform=train_transform,
        balance_mode='dynamic',
        seed=seed,
        erosion_kernel_size=erosion_kernel_size,
        feature_map_size=feature_map_size
    )
    
    val_dataset = DF40DatasetWithMask(
        json_path=json_path,
        data_root=data_root,
        transform=val_transform,
        balance_mode='static',
        seed=seed + 10000,
        erosion_kernel_size=erosion_kernel_size,
        feature_map_size=feature_map_size
    )
    
    return train_dataset, val_dataset
