"""
Region-Decoupled MoE 模块集合

包含:
  - RegionFeatureExtractor: 区域特征提取 (掩膜池化)
  - HierarchicalDecouplingModule: 层次化解耦模块 (3→7 向量)
  - CLSGuidedMoE: CLS 引导的混合专家层
"""

from lib.models.modules.region_feature_extractor import RegionFeatureExtractor
from lib.models.modules.hierarchical_decoupling import HierarchicalDecouplingModule
from lib.models.modules.cls_guided_moe import CLSGuidedMoE
