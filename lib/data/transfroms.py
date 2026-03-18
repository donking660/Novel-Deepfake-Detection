"""
Data Transforms Module (Albumentations 版本)
数据变换模块，使用 albumentations 库实现数据增强

参考实现:
    - training/dataset/sbi_dataset.py (init_data_aug_method)
    - training/dataset/albu.py (IsotropicResize)
    - training/config/detector/sbi.yaml (data_aug 配置)
"""

import random

import cv2
import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
from albumentations.pytorch import ToTensorV2


# ============================================================================
# CLIP 归一化参数 (CLIP ViT-L/14 预训练权重使用)
# ============================================================================
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# 保留 ImageNet 参数供参考
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================================
# 默认数据增强配置 (参考 sbi.yaml)
# ============================================================================
DEFAULT_DATA_AUG_CONFIG = {
    'resolution': 224,  # CLIP ViT-L/14 需要 224x224 输入
    'resize_scale': 256,  # 先缩放到较大尺寸，用于随机裁剪
    'flip_prob': 0.5,
    'rotate_prob': 0.5,
    'rotate_limit': [-10, 10],
    'blur_prob': 0.5,
    'blur_limit': [3, 7],
    'brightness_limit': [-0.1, 0.1],
    'contrast_limit': [-0.1, 0.1],
    'quality_lower': 40,
    'quality_upper': 100,
}


# ============================================================================
# IsotropicResize (等比缩放) - 从 training/dataset/albu.py 移植
# ============================================================================
def isotropically_resize_image(
    img: np.ndarray,
    size: int,
    interpolation_down: int = cv2.INTER_AREA,
    interpolation_up: int = cv2.INTER_CUBIC
) -> np.ndarray:
    """
    等比缩放图像，使长边等于指定大小
    
    Args:
        img: 输入图像 (H, W, C)
        size: 目标长边尺寸
        interpolation_down: 缩小时的插值方式
        interpolation_up: 放大时的插值方式
    
    Returns:
        等比缩放后的图像
    """
    h, w = img.shape[:2]
    
    if max(w, h) == size:
        return img
    
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    
    interpolation = interpolation_up if scale > 1 else interpolation_down
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    """
    等比缩放变换 (Albumentations 兼容)
    
    保持图像纵横比，将长边缩放到指定大小
    
    Args:
        max_side: 目标长边尺寸
        interpolation_down: 缩小时的插值方式 (默认 cv2.INTER_AREA)
        interpolation_up: 放大时的插值方式 (默认 cv2.INTER_CUBIC)
        always_apply: 是否总是应用
        p: 应用概率
    """
    
    def __init__(
        self,
        max_side: int,
        interpolation_down: int = cv2.INTER_AREA,
        interpolation_up: int = cv2.INTER_CUBIC,
        always_apply: bool = False,
        p: float = 1.0
    ):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up
    
    def apply(
        self,
        img: np.ndarray,
        interpolation_down: int = cv2.INTER_AREA,
        interpolation_up: int = cv2.INTER_CUBIC,
        **params
    ) -> np.ndarray:
        return isotropically_resize_image(
            img,
            size=self.max_side,
            interpolation_down=interpolation_down,
            interpolation_up=interpolation_up
        )
    
    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return self.apply(
            img,
            interpolation_down=cv2.INTER_NEAREST,
            interpolation_up=cv2.INTER_NEAREST,
            **params
        )
    
    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class Resize4xAndBack(ImageOnlyTransform):
    """
    下采样再上采样变换，模拟压缩伪影
    
    随机选择 2x 或 4x 下采样，然后上采样回原尺寸
    """
    
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)
    
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        import random
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(
            img, (w, h),
            interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST])
        )
        return img


# ============================================================================
# 训练数据增强 Pipeline
# ============================================================================
def get_train_transforms(config: dict = None) -> A.Compose:
    """
    获取训练数据增强 Pipeline (Albumentations)
    
    增强流程：
    1. HorizontalFlip - 水平翻转
    2. Rotate - 随机旋转
    3. GaussianBlur - 高斯模糊
    4. IsotropicResize(256) - 等比缩放到较大尺寸
    5. RandomBrightnessContrast / FancyPCA / HueSaturationValue - 颜色增强
    6. ImageCompression - JPEG 压缩
    7. PadIfNeeded(256) - 填充到中间尺寸
    8. RandomCrop(224) - 随机裁剪到最终尺寸 (位置数据增强)
    9. Normalize + ToTensorV2 - 归一化和转 Tensor
    
    Args:
        config: 数据增强配置字典，可包含以下键:
            - resolution: 最终输出分辨率 (默认 224，CLIP ViT-L/14)
            - resize_scale: 中间缩放尺寸 (默认 256，用于随机裁剪)
            - flip_prob: 翻转概率 (默认 0.5)
            - rotate_prob: 旋转概率 (默认 0.5)
            - rotate_limit: 旋转角度范围 (默认 [-10, 10])
            - blur_prob: 模糊概率 (默认 0.5)
            - blur_limit: 模糊核大小范围 (默认 [3, 7])
            - brightness_limit: 亮度调整范围 (默认 [-0.1, 0.1])
            - contrast_limit: 对比度调整范围 (默认 [-0.1, 0.1])
            - quality_lower: JPEG 压缩最低质量 (默认 40)
            - quality_upper: JPEG 压缩最高质量 (默认 100)
    
    Returns:
        A.Compose: Albumentations 变换 Pipeline
    """
    # 合并默认配置和用户配置
    cfg = DEFAULT_DATA_AUG_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    
    resolution = cfg['resolution']  # 最终输出尺寸 (224)
    resize_scale = cfg['resize_scale']  # 中间缩放尺寸 (256)
    
    transform = A.Compose([
        # 1. 水平翻转
        A.HorizontalFlip(p=cfg['flip_prob']),
        
        # 2. 随机旋转
        A.Rotate(
            limit=cfg['rotate_limit'],
            p=cfg['rotate_prob'],
            border_mode=cv2.BORDER_CONSTANT
        ),
        
        # 3. 高斯模糊
        A.GaussianBlur(
            blur_limit=cfg['blur_limit'],
            p=cfg['blur_prob']
        ),
        
        # 4. 等比缩放到较大尺寸 (256)，为随机裁剪提供空间
        IsotropicResize(
            max_side=resize_scale,
            interpolation_down=cv2.INTER_AREA,
            interpolation_up=cv2.INTER_CUBIC,
            always_apply=True
        ),
        
        # 5. 颜色增强 (三选一)
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=cfg['brightness_limit'],
                contrast_limit=cfg['contrast_limit']
            ),
            A.FancyPCA(),
            A.HueSaturationValue()
        ], p=0.5),
        
        # 6. JPEG 压缩
        A.ImageCompression(
            quality_lower=cfg['quality_lower'],
            quality_upper=cfg['quality_upper'],
            p=0.5
        ),
        
        # 7. 填充到中间尺寸 (等比缩放后可能不是正方形)
        A.PadIfNeeded(
            min_height=resize_scale,
            min_width=resize_scale,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        
        # 8. 随机裁剪到最终尺寸 (数据增强：位置随机性)
        A.RandomCrop(height=resolution, width=resolution),
        
        # 9. 归一化 (CLIP 参数，匹配 CLIP ViT backbone)
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        
        # 10. 转换为 PyTorch Tensor
        ToTensorV2()
    ])
    
    return transform


# ============================================================================
# 验证/测试数据预处理 Pipeline
# ============================================================================
def get_val_transforms(config: dict = None) -> A.Compose:
    """
    获取验证/测试数据预处理 Pipeline (无数据增强)
    
    仅包含：
    1. IsotropicResize - 等比缩放到目标尺寸
    2. PadIfNeeded - 填充到正方形
    3. CenterCrop - 中心裁剪
    4. Normalize - 归一化
    5. ToTensorV2 - 转 Tensor
    
    Args:
        config: 配置字典，可包含:
            - resolution: 目标分辨率 (默认 224，CLIP ViT-L/14)
    
    Returns:
        A.Compose: Albumentations 变换 Pipeline
    """
    # 合并默认配置
    cfg = DEFAULT_DATA_AUG_CONFIG.copy()
    if config is not None:
        cfg.update(config)
    
    resolution = cfg['resolution']
    
    transform = A.Compose([
        # 1. 等比缩放
        IsotropicResize(
            max_side=resolution,
            interpolation_down=cv2.INTER_AREA,
            interpolation_up=cv2.INTER_CUBIC
        ),
        
        # 2. 填充到正方形
        A.PadIfNeeded(
            min_height=resolution,
            min_width=resolution,
            border_mode=cv2.BORDER_CONSTANT,
            value=0
        ),
        
        # 3. 中心裁剪确保尺寸一致
        A.CenterCrop(height=resolution, width=resolution),
        
        # 4. 归一化 (CLIP 参数，匹配 CLIP ViT backbone)
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        
        # 5. 转换为 PyTorch Tensor
        ToTensorV2()
    ])
    
    return transform


# ============================================================================
# 辅助函数
# ============================================================================
def get_default_config() -> dict:
    """
    获取默认数据增强配置
    
    Returns:
        默认配置字典
    """
    return DEFAULT_DATA_AUG_CONFIG.copy()


# ============================================================================
# ScaleDF 数据增强
# ============================================================================

def _get_perturbation_pool() -> list:
    """获取 ScaleDF 扰动候选池 (30 种常见操作)"""
    return [
        A.Blur(p=1.0),
        A.RandomScale(p=1.0),
        A.ToGray(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomResizedCrop(size=(224, 224), p=1.0),
        A.ColorJitter(p=1.0),
        A.Rotate(p=1.0),
        A.PadIfNeeded(min_height=224, min_width=224, p=1.0),
        A.Downscale(scale_range=(0.25, 0.25), p=1.0),
        A.HorizontalFlip(p=1.0),
        A.GaussNoise(p=1.0),
        A.ImageCompression(quality_range=(40, 100), p=1.0),
        A.ChannelShuffle(p=1.0),
        A.Perspective(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.Affine(shear=(-20, 20), p=1.0),
        A.Solarize(p=1.0),
        A.PixelDropout(p=1.0),
        A.InvertImg(p=1.0),
        A.OpticalDistortion(p=1.0),
        A.GlassBlur(p=1.0),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0, p=1.0),
        A.ElasticTransform(p=1.0),
        A.FancyPCA(p=1.0),
        A.GridDistortion(p=1.0),
        A.ISONoise(p=1.0),
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0),
        A.Posterize(p=1.0),
        A.RandomGamma(p=1.0),
        A.Spatter(p=1.0),
    ]


class RandomPerturbationInjection(ImageOnlyTransform):
    """
    随机扰动注入 (ScaleDF 数据增强核心)

    概率分布:
      50% - 不施加任何扰动 (0 个)
      25% - 从候选池随机抽取 1 个扰动并应用
      25% - 从候选池随机抽取 2 个扰动并依次应用

    Args:
        perturbations: 扰动候选池 (albumentations Transform 列表)
    """

    def __init__(self, perturbations=None, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.perturbations = perturbations if perturbations is not None else _get_perturbation_pool()

    def apply(self, img, **params):
        r = random.random()
        if r < 0.5:
            return img

        n = 1 if r < 0.75 else 2
        selected = random.sample(self.perturbations, min(n, len(self.perturbations)))

        for t in selected:
            try:
                result = t(image=img)
                img = result['image']
            except Exception:
                pass

        return img

    def get_transform_init_args_names(self):
        return ()


def get_scaledf_train_transforms(config: dict = None) -> A.Compose:
    """
    ScaleDF 训练数据增强 Pipeline

    流程:
    1. Resize(224, 224) - 统一输入尺寸
    2. JPEG 压缩 (quality 40~100)
    3. 随机扰动注入 (50/25/25 概率分布)
    4. Resize(224, 224) - 确保扰动后尺寸一致
    5. Normalize (CLIP) + ToTensorV2

    Args:
        config: 配置字典, 可包含:
            - resolution: 输出分辨率 (默认 224)
            - jpeg_quality_lower: JPEG 最低质量 (默认 40)
            - jpeg_quality_upper: JPEG 最高质量 (默认 100)
    """
    cfg = {
        'resolution': 224,
        'jpeg_quality_lower': 40,
        'jpeg_quality_upper': 100,
    }
    if config:
        cfg.update(config)

    resolution = cfg['resolution']

    return A.Compose([
        A.Resize(resolution, resolution),
        A.ImageCompression(
            quality_range=(cfg['jpeg_quality_lower'], cfg['jpeg_quality_upper']),
            p=1.0,
        ),
        RandomPerturbationInjection(p=1.0),
        A.Resize(resolution, resolution),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ])


def get_scaledf_val_transforms(config: dict = None) -> A.Compose:
    """
    ScaleDF 验证/测试数据预处理 Pipeline (无增强)

    流程:
    1. Resize(224, 224)
    2. Normalize (CLIP) + ToTensorV2

    Args:
        config: 配置字典, 可包含:
            - resolution: 输出分辨率 (默认 224)
    """
    cfg = {'resolution': 224}
    if config:
        cfg.update(config)

    resolution = cfg['resolution']

    return A.Compose([
        A.Resize(resolution, resolution),
        A.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
        ToTensorV2(),
    ])
