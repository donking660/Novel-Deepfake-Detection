"""
ScaleDF Dataset with Face Mask Support
ScaleDF 数据集加载器 (带面部区域掩膜支持)

在 ScaleDFDataset 基础上增加:
  - 过滤 _masks 后缀目录, 避免掩膜文件被当作训练图像
  - 加载面部区域二值掩膜 (face region mask)
  - 通过腐蚀/膨胀预计算三个区域掩膜 (face / boundary / background)
  - 下采样到 16x16 并归一化, 确保 M1 + M2 + M3 = 1
  - 掩膜与图像同步数据增强 (包括 RandomPerturbationInjection 中的空间变换)

数据集结构:
  ScaleDF/train/
    000000Xxx/          -> 真实人脸 (label=0)
    000000Xxx_masks/    -> 对应掩膜
    OtherXxx/           -> 伪造人脸 (label=1)
    OtherXxx_masks/     -> 对应掩膜

掩膜路径规则:
  图像: {data_root}/{folder}/{filename}.jpg
  掩膜: {data_root}/{folder}_masks/{filename_stem}.png
"""

import hashlib
import os
import pickle
import random
import time
from pathlib import Path
from typing import List, Tuple, Dict, Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from lib.data.transfroms import (
    CLIP_MEAN, CLIP_STD,
    _get_perturbation_pool,
)

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
TRAIN_SUBSAMPLE_RATE = 10
# 使用独立的缓存文件名, 避免与 ScaleDFDataset 冲突
CACHE_FILENAME = '_scaledf_mask_filelist_cache.pkl'


class ScaleDFDatasetWithMask(Dataset):
    """
    ScaleDF 数据集加载器 (带面部掩膜支持)

    在 ScaleDFDataset 基础上:
      1. 扫描目录时过滤 _masks 后缀文件夹
      2. 为每张图片加载对应的面部区域掩膜
      3. 数据增强时掩膜与图像同步空间变换
      4. 预计算三区域掩膜 (face / boundary / background)

    Args:
        data_root: 数据集根目录 (ScaleDF/train/)
        split: 'train' or 'val'
        train_ratio: 训练集占比
        seed: 随机种子
        resolution: 输入图像分辨率 (默认 224)
        jpeg_quality_lower: JPEG 压缩最低质量
        jpeg_quality_upper: JPEG 压缩最高质量
        erosion_kernel_size: 腐蚀/膨胀核大小 (默认 19)
        feature_map_size: 特征图空间分辨率 (默认 16, ViT-L/14: 224/14=16)
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        train_ratio: float = 0.9,
        seed: int = 42,
        resolution: int = 224,
        jpeg_quality_lower: int = 40,
        jpeg_quality_upper: int = 100,
        erosion_kernel_size: int = 19,
        feature_map_size: int = 16,
    ):
        super().__init__()

        self.data_root = str(data_root)
        self.split = split
        self.train_ratio = train_ratio
        self.seed = seed
        self.resolution = resolution
        self.erosion_kernel_size = erosion_kernel_size
        self.feature_map_size = feature_map_size

        # 形态学操作的核
        self.morph_kernel = np.ones(
            (erosion_kernel_size, erosion_kernel_size), dtype=np.uint8
        )

        # 样本列表
        self.samples: List[Tuple[str, int]] = []
        self.real_count = 0
        self.fake_count = 0
        self.num_real_folders = 0
        self.num_fake_folders = 0
        self.method_stats: Dict[str, Dict] = {}
        self._total_before_subsample = 0

        # =====================================================================
        # 构建数据增强管线 (拆分为三段, 支持掩膜同步)
        # =====================================================================
        if split == 'train':
            # 第一段: Resize + JPEG 压缩
            self.pre_transform = A.Compose([
                A.Resize(resolution, resolution),
                A.ImageCompression(
                    quality_range=(jpeg_quality_lower, jpeg_quality_upper),
                    p=1.0,
                ),
            ])
            # 第二段: 随机扰动注入 (自定义, 需要手动处理掩膜同步)
            self.perturbations = _get_perturbation_pool()
            self.use_perturbation = True
        else:
            # 验证集: 仅 Resize
            self.pre_transform = A.Compose([
                A.Resize(resolution, resolution),
            ])
            self.perturbations = None
            self.use_perturbation = False

        # 第三段: 扰动后统一 Resize (修正扰动可能导致的尺寸变化)
        self.post_resize = A.Compose([
            A.Resize(resolution, resolution),
        ])

        # 最终: 归一化 + 转 Tensor (仅作用于图像)
        self.final_transform = A.Compose([
            A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ToTensorV2(),
        ])

        # 收集样本
        self._collect_samples()
        self._print_stats()

    # ------------------------------------------------------------------
    # 确定性种子
    # ------------------------------------------------------------------
    @staticmethod
    def _folder_seed(seed: int, folder_name: str) -> int:
        """为每个文件夹生成确定性种子 (跨 Python 会话稳定)"""
        h = int(hashlib.md5(folder_name.encode()).hexdigest(), 16) % (2 ** 31)
        return seed + h

    # ------------------------------------------------------------------
    # 文件列表缓存
    # ------------------------------------------------------------------
    def _cache_path(self) -> str:
        return os.path.join(self.data_root, CACHE_FILENAME)

    def _load_cache(self) -> Optional[Dict[str, List[str]]]:
        """
        尝试从磁盘加载文件列表缓存。
        返回 {folder_name: sorted_filenames} 或 None。
        """
        cache_path = self._cache_path()
        if not os.path.isfile(cache_path):
            return None
        try:
            t0 = time.time()
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            elapsed = time.time() - t0
            if cache.get('data_root') != self.data_root:
                print(f"  缓存 data_root 不匹配, 重新扫描")
                return None
            print(f"  从缓存加载文件列表: {cache_path}  ({elapsed:.1f}s)")
            return cache['folders']
        except Exception as e:
            print(f"  缓存读取失败 ({e}), 重新扫描")
            return None

    def _save_cache(self, folders: Dict[str, List[str]]):
        cache_path = self._cache_path()
        try:
            t0 = time.time()
            cache = {
                'data_root': self.data_root,
                'folders': folders,
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            elapsed = time.time() - t0
            total_files = sum(len(v) for v in folders.values())
            print(f"  文件列表已缓存: {cache_path}  "
                  f"({len(folders)} 文件夹, {total_files} 文件, {elapsed:.1f}s)")
        except Exception as e:
            print(f"  缓存写入失败 ({e}), 不影响正常使用")

    def _scan_folders(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        扫描全部子文件夹, 返回 (sorted_subdirs, {folder_name: sorted_filenames})

        关键: 过滤掉 _masks 后缀的文件夹, 避免掩膜被当作训练数据
        """
        all_entries = sorted(os.listdir(self.data_root))
        subdirs = [
            e for e in all_entries
            if os.path.isdir(os.path.join(self.data_root, e))
            and not e.endswith('_masks')  # ← 过滤掩膜文件夹
        ]

        print(f"  扫描到 {len(subdirs)} 个数据文件夹 (已过滤 _masks 目录), "
              f"开始收集文件列表...")

        folders: Dict[str, List[str]] = {}
        for i, folder_name in enumerate(subdirs):
            folder_path = os.path.join(self.data_root, folder_name)
            filenames = sorted(
                fn for fn in os.listdir(folder_path)
                if os.path.splitext(fn)[1].lower() in VALID_EXTENSIONS
            )
            if filenames:
                folders[folder_name] = filenames

            if (i + 1) % 50 == 0 or (i + 1) == len(subdirs):
                total_files = sum(len(v) for v in folders.values())
                print(f"    [{i + 1}/{len(subdirs)}] 已扫描 {total_files} 个文件")

        return subdirs, folders

    # ------------------------------------------------------------------
    # 样本收集 (与 ScaleDFDataset 一致)
    # ------------------------------------------------------------------
    def _collect_samples(self):
        """扫描目录 (或读缓存), 按文件夹划分 train/val, 训练集做 1/10 子采样"""
        if not os.path.isdir(self.data_root):
            raise FileNotFoundError(f"数据目录不存在: {self.data_root}")

        cached = self._load_cache()
        if cached is not None:
            subdirs = sorted(cached.keys())
            folders = cached
        else:
            subdirs, folders = self._scan_folders()
            self._save_cache(folders)

        is_train = self.split == 'train'

        for i, folder_name in enumerate(subdirs):
            if folder_name not in folders:
                continue
            filenames = folders[folder_name]
            if not filenames:
                continue

            folder_path = os.path.join(self.data_root, folder_name)
            label = 0 if folder_name.startswith('000000') else 1

            rng = random.Random(self._folder_seed(self.seed, folder_name))
            shuffled = filenames.copy()
            rng.shuffle(shuffled)

            split_idx = int(len(shuffled) * self.train_ratio)
            if is_train:
                selected = shuffled[:split_idx]
                total_before = len(selected)
                selected = selected[::TRAIN_SUBSAMPLE_RATE]
                self._total_before_subsample += total_before
            else:
                selected = shuffled[split_idx:]

            for fn in selected:
                self.samples.append((os.path.join(folder_path, fn), label))

            count = len(selected)
            if label == 0:
                self.real_count += count
                self.num_real_folders += 1
            else:
                self.fake_count += count
                self.num_fake_folders += 1

            self.method_stats[folder_name] = {
                'label': 'real' if label == 0 else 'fake',
                'count': count,
                'total': len(filenames),
            }

            if (i + 1) % 50 == 0 or (i + 1) == len(subdirs):
                print(f"    [{i + 1}/{len(subdirs)}] 已收集 {len(self.samples)} 样本 "
                      f"(真实: {self.real_count}, 伪造: {self.fake_count})")

        rng_all = random.Random(self.seed)
        rng_all.shuffle(self.samples)

    def _print_stats(self):
        """打印数据集统计信息"""
        print(f"\n  ScaleDF 数据集 (带掩膜) [{self.split}] 加载完成:")
        print(f"    数据目录: {self.data_root}")
        print(f"    划分比例: {self.train_ratio:.0%} / {1 - self.train_ratio:.0%}")
        if self.split == 'train':
            print(f"    子采样: 每 {TRAIN_SUBSAMPLE_RATE} 张取 1 张 (仅训练集)")
            print(f"    子采样前: {self._total_before_subsample}")
        print(f"    真实文件夹: {self.num_real_folders}")
        print(f"    伪造文件夹: {self.num_fake_folders} (伪造方法数)")
        print(f"    腐蚀/膨胀核大小: {self.erosion_kernel_size}")
        print(f"    特征图尺寸: {self.feature_map_size}x{self.feature_map_size}")
        print(f"    样本统计:")
        print(f"      真实: {self.real_count}")
        print(f"      伪造: {self.fake_count}")
        print(f"      总计: {len(self.samples)}")
        if self.real_count > 0:
            print(f"      比例 (real:fake): 1:{self.fake_count / self.real_count:.2f}")

    # ------------------------------------------------------------------
    # 掩膜路径解析
    # ------------------------------------------------------------------
    def _get_mask_path(self, image_path: str) -> Optional[str]:
        """
        根据图片路径获取对应的面部掩膜路径

        路径规则:
          图像: {data_root}/{folder}/{filename}.jpg
          掩膜: {data_root}/{folder}_masks/{stem}.png

        扩展名处理: 无论原图是 .jpg/.jpeg/.png, 掩膜统一为 .png

        Args:
            image_path: 图片路径

        Returns:
            mask 路径, 如果找不到返回 None
        """
        path = Path(image_path)
        parent_dir = path.parent
        grandparent_dir = parent_dir.parent
        stem = path.stem  # 不含扩展名的文件名

        mask_folder_name = parent_dir.name + '_masks'
        mask_path = str(grandparent_dir / mask_folder_name / (stem + '.png'))

        if Path(mask_path).exists():
            return mask_path

        return None

    # ------------------------------------------------------------------
    # 掩膜加载
    # ------------------------------------------------------------------
    def _load_face_mask(
        self,
        mask_path: Optional[str],
        target_h: int = 224,
        target_w: int = 224,
    ) -> np.ndarray:
        """
        加载面部区域二值掩膜

        Args:
            mask_path: 掩膜路径, None 则返回全零掩膜
            target_h: 目标高度 (与图像匹配)
            target_w: 目标宽度 (与图像匹配)

        Returns:
            二值掩膜 (H, W), 值为 0 或 255, dtype=uint8
        """
        if mask_path is None or not Path(mask_path).exists():
            return np.zeros((target_h, target_w), dtype=np.uint8)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((target_h, target_w), dtype=np.uint8)

        # resize 到目标尺寸 (与图像一致, 保证 albumentations 同步)
        if mask.shape[0] != target_h or mask.shape[1] != target_w:
            mask = cv2.resize(mask, (target_w, target_h),
                              interpolation=cv2.INTER_NEAREST)

        # 确保二值化
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask

    # ------------------------------------------------------------------
    # 三区域掩膜计算
    # ------------------------------------------------------------------
    def _compute_region_masks(self, face_mask: np.ndarray) -> np.ndarray:
        """
        从面部二值掩膜计算三个区域掩膜, 并下采样到特征图尺寸

        计算步骤:
          1. 在 224x224 分辨率下腐蚀/膨胀
          2. 得到 face, boundary, background 三个区域
          3. 下采样到 16x16
          4. 归一化确保 M1 + M2 + M3 = 1

        Args:
            face_mask: 面部二值掩膜 (H, W), 值为 0 或 255, dtype=uint8

        Returns:
            region_masks: (3, feature_map_size, feature_map_size), float32
                channel 0: face (人脸内部)
                channel 1: boundary (边界)
                channel 2: background (背景)
        """
        # 1. 腐蚀和膨胀
        mask_eroded = cv2.erode(face_mask, self.morph_kernel, iterations=1)
        mask_dilated = cv2.dilate(face_mask, self.morph_kernel, iterations=1)

        # 2. 计算三个区域 (float32, 0~1)
        m_face = mask_eroded.astype(np.float32) / 255.0
        m_dilated = mask_dilated.astype(np.float32) / 255.0
        m_eroded = mask_eroded.astype(np.float32) / 255.0
        m_boundary = np.clip(m_dilated - m_eroded, 0, 1)
        m_background = np.clip(1.0 - m_dilated, 0, 1)

        # 3. 下采样到 feature_map_size x feature_map_size
        size = self.feature_map_size
        m_face_16 = cv2.resize(m_face, (size, size),
                               interpolation=cv2.INTER_LINEAR)
        m_boundary_16 = cv2.resize(m_boundary, (size, size),
                                   interpolation=cv2.INTER_LINEAR)
        m_background_16 = cv2.resize(m_background, (size, size),
                                     interpolation=cv2.INTER_LINEAR)

        # 4. 归一化确保 M1 + M2 + M3 = 1
        total = m_face_16 + m_boundary_16 + m_background_16 + 1e-8
        m_face_16 = m_face_16 / total
        m_boundary_16 = m_boundary_16 / total
        m_background_16 = m_background_16 / total

        # 5. 组合成 (3, H, W)
        region_masks = np.stack(
            [m_face_16, m_boundary_16, m_background_16], axis=0
        )
        return region_masks.astype(np.float32)

    # ------------------------------------------------------------------
    # 随机扰动注入 (掩膜同步版)
    # ------------------------------------------------------------------
    def _apply_perturbation_with_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        随机扰动注入, 保证空间变换同步作用于图像和掩膜

        概率分布:
          50% - 不施加任何扰动
          25% - 从候选池随机抽取 1 个扰动
          25% - 从候选池随机抽取 2 个扰动

        对于每个选中的扰动:
          - DualTransform (空间变换): 同时传入 image + mask, 保证同步
          - ImageOnlyTransform (颜色/噪声): 仅作用于 image, mask 不变

        Args:
            image: 输入图像 (H, W, 3), uint8
            mask: 面部掩膜 (H, W), uint8

        Returns:
            (augmented_image, augmented_mask)
        """
        r = random.random()
        if r < 0.5:
            return image, mask

        n = 1 if r < 0.75 else 2
        selected = random.sample(
            self.perturbations, min(n, len(self.perturbations))
        )

        for t in selected:
            try:
                if isinstance(t, DualTransform):
                    # 空间变换: 同时变换 image 和 mask
                    result = t(image=image, mask=mask)
                    image = result['image']
                    mask = result['mask']
                else:
                    # 颜色/噪声变换: 仅作用于 image
                    result = t(image=image)
                    image = result['image']
            except Exception:
                pass

        return image, mask

    # ------------------------------------------------------------------
    # 图像加载
    # ------------------------------------------------------------------
    def _load_image(self, img_path: str) -> np.ndarray:
        """使用 OpenCV 加载图像并转换为 RGB"""
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法加载图像: {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor]:
        """
        获取数据样本

        数据增强流程 (训练集):
          1. pre_transform: Resize(224) + JPEG压缩 (mask 同步 Resize)
          2. _apply_perturbation_with_mask: 随机扰动注入 (空间变换同步 mask)
          3. post_resize: Resize(224) 修正扰动后尺寸 (mask 同步)
          4. 掩膜二值化 + 三区域计算
          5. final_transform: Normalize + ToTensorV2 (仅图像)

        Returns:
            (image_tensor, label, region_masks) 元组
            - image_tensor: (C, H, W) 格式的 Tensor
            - label: 0 表示真实, 1 表示伪造
            - region_masks: (3, 16, 16) 的 Tensor, 三个区域掩膜
        """
        img_path, label = self.samples[idx]

        # 加载图像
        image = self._load_image(img_path)
        h, w = image.shape[:2]

        # 加载面部掩膜 (与原图同尺寸)
        mask_path = self._get_mask_path(img_path)
        face_mask = self._load_face_mask(mask_path, target_h=h, target_w=w)

        # --- 第一段: Resize + JPEG 压缩 ---
        result = self.pre_transform(image=image, mask=face_mask)
        image = result['image']
        face_mask = result['mask']

        # --- 第二段: 随机扰动注入 (仅训练集) ---
        if self.use_perturbation:
            image, face_mask = self._apply_perturbation_with_mask(
                image, face_mask
            )

        # --- 第三段: 扰动后统一 Resize ---
        result = self.post_resize(image=image, mask=face_mask)
        image = result['image']
        face_mask = result['mask']

        # --- 掩膜后处理 ---
        # 确保增强后的掩膜仍然是二值的 (空间变换可能引入插值中间值)
        if isinstance(face_mask, np.ndarray):
            _, face_mask = cv2.threshold(
                face_mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
            )
        else:
            face_mask = face_mask.numpy()
            _, face_mask = cv2.threshold(
                face_mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY
            )

        # 预计算三个区域掩膜
        region_masks = self._compute_region_masks(face_mask)
        region_masks = torch.from_numpy(region_masks)  # (3, 16, 16)

        # --- 最终: Normalize + ToTensor (仅图像) ---
        result = self.final_transform(image=image)
        image = result['image']

        return image, label, region_masks

    # ------------------------------------------------------------------
    # 统计接口 (与 ScaleDFDataset 兼容)
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        """获取样本统计"""
        return {
            'total': len(self.samples),
            'real': self.real_count,
            'fake': self.fake_count,
            'num_real_folders': self.num_real_folders,
            'num_fake_folders': self.num_fake_folders,
        }
