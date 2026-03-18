"""
ScaleDF Dataset Module
ScaleDF 数据集加载器

数据集结构:
  ScaleDF/train/
    000000Xxx/   -> 真实人脸 (label=0)
    OtherXxx/    -> 伪造人脸 (label=1), 每个文件夹代表一种伪造方法

支持:
  - 按目录结构自动识别真实/伪造样本 (000000开头为真实)
  - 按比例划分训练集/验证集 (每个子文件夹内独立划分, 保证可复现)
  - 训练集 1/10 子采样 (减轻 ~1300 万图像的存储/计算压力)
  - 文件列表磁盘缓存 (避免重复扫描海量文件)
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

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg'}
TRAIN_SUBSAMPLE_RATE = 10
CACHE_FILENAME = '_scaledf_filelist_cache.pkl'


class ScaleDFDataset(Dataset):
    """
    ScaleDF 数据集加载器

    Args:
        data_root: 数据集根目录 (ScaleDF/train/)
        split: 'train' or 'val'
        train_ratio: 训练集占比
        transform: albumentations transform (A.Compose 对象)
        seed: 随机种子
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        train_ratio: float = 0.9,
        transform: Callable = None,
        seed: int = 42
    ):
        super().__init__()

        self.data_root = str(data_root)
        self.split = split
        self.train_ratio = train_ratio
        self.transform = transform
        self.seed = seed

        self.samples: List[Tuple[str, int]] = []
        self.real_count = 0
        self.fake_count = 0
        self.num_real_folders = 0
        self.num_fake_folders = 0
        self.method_stats: Dict[str, Dict] = {}
        self._total_before_subsample = 0

        self._collect_samples()
        self._print_stats()

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
        """扫描全部子文件夹, 返回 (sorted_subdirs, {folder_name: sorted_filenames})"""
        all_entries = sorted(os.listdir(self.data_root))
        subdirs = [e for e in all_entries
                   if os.path.isdir(os.path.join(self.data_root, e))]

        print(f"  扫描到 {len(subdirs)} 个子文件夹, 开始收集文件列表...")

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
    # 样本收集
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
        print(f"\n  ScaleDF 数据集 [{self.split}] 加载完成:")
        print(f"    数据目录: {self.data_root}")
        print(f"    划分比例: {self.train_ratio:.0%} / {1 - self.train_ratio:.0%}")
        if self.split == 'train':
            print(f"    子采样: 每 {TRAIN_SUBSAMPLE_RATE} 张取 1 张 (仅训练集)")
            print(f"    子采样前: {self._total_before_subsample}")
        print(f"    真实文件夹: {self.num_real_folders}")
        print(f"    伪造文件夹: {self.num_fake_folders} (伪造方法数)")
        print(f"    样本统计:")
        print(f"      真实: {self.real_count}")
        print(f"      伪造: {self.fake_count}")
        print(f"      总计: {len(self.samples)}")
        if self.real_count > 0:
            print(f"      比例 (real:fake): 1:{self.fake_count / self.real_count:.2f}")

    def _load_image(self, img_path: str) -> np.ndarray:
        """使用 OpenCV 加载图像并转换为 RGB"""
        image = cv2.imread(img_path)
        if image is None:
            print(f"警告: 无法加载图像: {img_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = self._load_image(img_path)

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        return image, label

    def get_stats(self) -> Dict:
        """获取样本统计"""
        return {
            'total': len(self.samples),
            'real': self.real_count,
            'fake': self.fake_count,
            'num_real_folders': self.num_real_folders,
            'num_fake_folders': self.num_fake_folders,
        }
