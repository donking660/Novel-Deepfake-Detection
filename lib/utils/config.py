"""
配置文件加载工具
支持 YAML 配置文件的加载和继承

使用方法:
    from lib.utils.config import get_config
    
    cfg = get_config('configs/exp_cls_avg.yaml')
    print(cfg.model.feature_aggregation)  # 'cls_avg'
"""

import yaml
import copy
from pathlib import Path
from typing import Dict, Any, Optional


def load_yaml(path: str) -> Dict[str, Any]:
    """
    加载单个 YAML 文件
    
    Args:
        path: YAML 文件路径
    
    Returns:
        配置字典
    """
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def merge_config(base: Dict, override: Dict) -> Dict:
    """
    递归合并配置，override 覆盖 base
    
    Args:
        base: 基础配置
        override: 覆盖配置
    
    Returns:
        合并后的配置
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件，支持继承 (_base_)
    
    支持多级继承，子配置会覆盖父配置中的同名字段
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        合并后的完整配置字典
    
    Example:
        # configs/exp_cls_avg.yaml
        _base_: base.yaml
        model:
          feature_aggregation: cls_avg
        
        # 会先加载 base.yaml，然后用 exp_cls_avg.yaml 的内容覆盖
    """
    config_path = Path(config_path)
    config = load_yaml(config_path)
    
    # 处理继承
    if '_base_' in config:
        base_path = config_path.parent / config.pop('_base_')
        if not base_path.exists():
            raise FileNotFoundError(f"基础配置文件不存在: {base_path}")
        base_config = load_config(str(base_path))  # 递归加载基础配置
        config = merge_config(base_config, config)
    
    return config


class Config:
    """
    配置对象，支持属性访问和字典访问
    
    Example:
        cfg = Config({'model': {'name': 'vit'}})
        print(cfg.model.name)  # 'vit'
        print(cfg['model']['name'])  # 'vit'
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)
    
    def get(self, key, default=None):
        """安全获取属性，不存在时返回默认值"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def __repr__(self):
        return f"Config({self.to_dict()})"
    
    def __str__(self):
        return yaml.dump(self.to_dict(), default_flow_style=False, allow_unicode=True)


def get_config(config_path: str) -> Config:
    """
    加载配置并返回 Config 对象
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        Config 对象，支持属性访问
    
    Example:
        cfg = get_config('configs/exp_cls_avg.yaml')
        print(cfg.model.feature_aggregation)
        print(cfg.train.batch_size)
    """
    config_dict = load_config(config_path)
    return Config(config_dict)


def save_config(config: Config, save_path: str):
    """
    保存配置到 YAML 文件
    
    Args:
        config: Config 对象
        save_path: 保存路径
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, allow_unicode=True)


def config_to_args(config: Config) -> Dict[str, Any]:
    """
    将 Config 对象扁平化为命令行参数格式
    
    Args:
        config: Config 对象
    
    Returns:
        扁平化的参数字典
    
    Example:
        cfg = Config({'train': {'lr': 0.001, 'epochs': 100}})
        args = config_to_args(cfg)
        # {'train_lr': 0.001, 'train_epochs': 100}
    """
    def flatten(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}_{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    return flatten(config.to_dict())
