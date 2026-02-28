"""
NeuroConscious Transformer - 数据集接口
NCT Datasets: 标准化数据加载器

支持数据集:
- MNIST（手写数字）
- Fashion-MNIST（服装分类）
- CIFAR-10（物体识别）
- 自定义合成数据集（猫 vs 非猫）

功能:
1. 自动下载和缓存
2. 数据预处理（归一化、增强）
3. 转为 NCT 格式（多模态输入）
4. Few-shot 学习支持

作者：NeuroConscious 研发团队
日期：2026 年 2 月 24 日
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


# ============================================================================
# NCT 格式转换
# ============================================================================

def convert_to_nct_format(
    image: torch.Tensor,
    label: int,
    img_size: int = 28,
) -> Tuple[Dict[str, np.ndarray], int]:
    """将图像转为 NCT 多模态格式
    
    Args:
        image: PyTorch 图像张量 [C, H, W]
        label: 类别标签
        img_size: 目标大小
        
    Returns:
        sensory_data: NCT 感觉输入字典
            - 'visual': [H, W] 或 [T, H, W]
            - 'auditory': [T, F] 语谱图（可选）
            - 'interoceptive': [10] 内感受向量（可选）
        label: 标签
    """
    # 1. 视觉输入（主要模态）
    if image.dim() == 3:
        # RGB 图像 → 灰度
        if image.shape[0] == 3:
            visual = image.mean(dim=0, keepdim=True)  # [1, H, W]
        else:
            visual = image
    else:
        visual = image.unsqueeze(0)
    
    # 调整大小
    if visual.shape[1:] != (img_size, img_size):
        visual = torch.nn.functional.interpolate(
            visual.unsqueeze(0),
            size=(img_size, img_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
    
    # 归一化到 [0, 1]
    visual = (visual - visual.min()) / (visual.max() - visual.min() + 1e-8)
    
    # 转为 numpy
    visual_np = visual.squeeze().cpu().numpy()
    
    # 2. 听觉输入（可选，用噪声填充）
    # NCT 期望格式：[T, F] 语谱图
    auditory_np = np.random.randn(10, 10).astype(np.float32) * 0.1
    
    # 3. 内感受输入（可选，随机向量）
    # NCT 期望格式：[10]
    intero_np = np.random.randn(10).astype(np.float32) * 0.1
    
    sensory_data = {
        'visual': visual_np,
        # 暂时禁用听觉和内感受输入，避免维度问题
        # 'auditory': auditory_np,
        # 'interoceptive': intero_np,
    }
    
    return sensory_data, label


class SimpleAudioFix(torch.nn.Module):
    """简单的音频输入修复模块
    
    将 [T, F] 转为 [B, F, T] 以适配 AudioSpectrogramTransformer
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # [T, F] → [1, F, T]
            return x.unsqueeze(0).transpose(1, 2)
        elif x.dim() == 3:
            # [B, T, F] → [B, F, T]
            return x.transpose(1, 2)
        return x


# ============================================================================
# Few-Shot 数据集
# ============================================================================

class FewShotDataset(Dataset):
    """Few-shot 学习数据集
    
    从标准数据集中采样少量样本进行训练
    """
    
    def __init__(
        self,
        dataset: Dataset,
        n_samples_per_class: int = 10,
        n_classes: Optional[int] = None,
        seed: int = 42,
    ):
        """初始化 Few-shot 数据集
        
        Args:
            dataset: 原始数据集
            n_samples_per_class: 每类样本数
            n_classes: 使用的类别数（None=全部）
            seed: 随机种子
        """
        super().__init__()
        
        self.n_samples_per_class = n_samples_per_class
        self.n_classes = n_classes or len(dataset.classes)
        self.seed = seed
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 采样样本
        self.samples = self._sample_from_dataset(dataset)
        
        logger.info(
            f"[FewShotDataset] 创建成功："
            f"{len(self.samples)} 样本，{self.n_classes} 类，"
            f"每类{n_samples_per_class}个样本"
        )
    
    def _sample_from_dataset(self, dataset: Dataset) -> List[Tuple]:
        """从数据集采样"""
        # 按类别分组
        class_indices = {}
        
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # 限制类别数
        selected_classes = sorted(list(class_indices.keys()))[:self.n_classes]
        
        # 每类采样 n_samples_per_class 个样本
        samples = []
        for cls in selected_classes:
            indices = class_indices[cls]
            if len(indices) < self.n_samples_per_class:
                # 如果该类样本不足，重复采样
                sampled_indices = np.random.choice(
                    indices,
                    size=self.n_samples_per_class,
                    replace=True
                )
            else:
                sampled_indices = np.random.choice(
                    indices,
                    size=self.n_samples_per_class,
                    replace=False
                )
            
            for idx in sampled_indices:
                samples.append(dataset[idx])
        
        # 打乱顺序
        np.random.shuffle(samples)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, np.ndarray], int]:
        image, label = self.samples[idx]
        
        # 转为 NCT 格式
        sensory_data, label = convert_to_nct_format(image, label)
        
        return sensory_data, label


# ============================================================================
# 标准数据集加载器
# ============================================================================

def load_mnist(
    root: str = 'data',
    batch_size: int = 32,
    n_samples_per_class: Optional[int] = None,
    n_classes: Optional[int] = None,
    img_size: int = 28,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """加载 MNIST 数据集
    
    Args:
        root: 数据根目录
        batch_size: 批次大小
        n_samples_per_class: Few-shot 样本数（None=完整数据集）
        n_classes: 使用的类别数（None=全部 10 类）
        img_size: 图像大小
        seed: 随机种子
        
    Returns:
        train_loader, test_loader
    """
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # 下载/加载训练集
    train_dataset = datasets.MNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    
    # 测试集
    test_dataset = datasets.MNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    
    # Few-shot 模式
    if n_samples_per_class is not None:
        train_dataset = FewShotDataset(
            train_dataset,
            n_samples_per_class=n_samples_per_class,
            n_classes=n_classes,
            seed=seed,
        )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    logger.info(
        f"[MNIST] 加载完成："
        f"训练集={len(train_dataset)}, 测试集={len(test_dataset)}"
    )
    
    return train_loader, test_loader


def load_fashion_mnist(
    root: str = 'data',
    batch_size: int = 32,
    n_samples_per_class: Optional[int] = None,
    n_classes: Optional[int] = None,
    img_size: int = 28,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """加载 Fashion-MNIST 数据集
    
    参数和返回值同 load_mnist
    """
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    # 训练集
    train_dataset = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    
    # 测试集
    test_dataset = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    
    # Few-shot 模式
    if n_samples_per_class is not None:
        train_dataset = FewShotDataset(
            train_dataset,
            n_samples_per_class=n_samples_per_class,
            n_classes=n_classes,
            seed=seed,
        )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    logger.info(
        f"[FashionMNIST] 加载完成："
        f"训练集={len(train_dataset)}, 测试集={len(test_dataset)}"
    )
    
    return train_loader, test_loader


def load_cifar10(
    root: str = 'data',
    batch_size: int = 32,
    n_samples_per_class: Optional[int] = None,
    n_classes: Optional[int] = None,
    img_size: int = 32,
    grayscale: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """加载 CIFAR-10 数据集
    
    Args:
        grayscale: 是否转灰度（True 适合 NCT）
    """
    # 数据变换
    transform_list = [
        transforms.Resize((img_size, img_size)),
    ]
    
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    
    transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    
    # 训练集
    train_dataset = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    
    # 测试集
    test_dataset = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )
    
    # Few-shot 模式
    if n_samples_per_class is not None:
        train_dataset = FewShotDataset(
            train_dataset,
            n_samples_per_class=n_samples_per_class,
            n_classes=n_classes,
            seed=seed,
        )
    
    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    logger.info(
        f"[CIFAR10] 加载完成："
        f"训练集={len(train_dataset)}, 测试集={len(test_dataset)}"
    )
    
    return train_loader, test_loader


# ============================================================================
# 通用加载接口
# ============================================================================

def load_dataset(
    name: str,
    root: str = 'data',
    batch_size: int = 32,
    n_samples_per_class: Optional[int] = None,
    n_classes: Optional[int] = None,
    img_size: int = 28,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """通用数据集加载接口
    
    Args:
        name: 数据集名称 ('mnist', 'fashion_mnist', 'cifar10')
        其他参数同各具体函数
        
    Returns:
        train_loader, test_loader
    """
    name = name.lower()
    
    if name == 'mnist':
        return load_mnist(
            root=root,
            batch_size=batch_size,
            n_samples_per_class=n_samples_per_class,
            n_classes=n_classes,
            img_size=img_size,
            seed=seed,
        )
    elif name == 'fashion_mnist':
        return load_fashion_mnist(
            root=root,
            batch_size=batch_size,
            n_samples_per_class=n_samples_per_class,
            n_classes=n_classes,
            img_size=img_size,
            seed=seed,
        )
    elif name == 'cifar10':
        return load_cifar10(
            root=root,
            batch_size=batch_size,
            n_samples_per_class=n_samples_per_class,
            n_classes=n_classes,
            img_size=img_size if img_size != 28 else 32,
            seed=seed,
        )
    else:
        raise ValueError(f"不支持的数据集：{name}")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'convert_to_nct_format',
    'FewShotDataset',
    'load_mnist',
    'load_fashion_mnist',
    'load_cifar10',
    'load_dataset',
]
