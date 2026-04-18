"""
实验9: 意识度量深度验证（含条件控制）

实验目标:
1. 使用结构化扰动验证 Φ 值的有效性和敏感性
2. 通过条件控制区分"Φ 因准确率下降而下降"和"Φ 独立下降"

实验假设:
- H4a: 结构化扰动（遮挡、模糊）将显著降低 Φ 值（p < 0.05）
- H4b: 在控制准确率不变的条件下，Φ 值仍随扰动强度变化（证明 Φ 不是准确率的简单代理）

扰动类型:
- P0: 块遮挡（block_occlusion）— 10%-50%
- P0: 高斯模糊（gaussian_blur）— σ=1.0-5.0
- P1: 语义破坏（semantic_destruction）— 旋转/拼接
- P2可选: 对抗样本（fgsm_adversarial）— ε=0.01-0.1

条件控制:
- 对照组1: 随机分类器 → Φ ≈ 0
- 对照组2: 冻结中间层 → Φ 显著低于正常
- 对照组3: 温度缩放 → Φ 变化且 Acc 不变

使用实验8a最佳LIF参数: th=0.8, decay=0.99, T=2, σ=10.0

改进版（v2）:
- 12个扰动级别（原5个），提高Pearson相关的统计功效（df=10）
- 1000样本（原200），减小Φ均值的标准误
- 使用特征值谱（participation ratio）方法计算Φ，增强区分性
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from scipy import stats as scipy_stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 添加路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spiking_nct.manager import SpikingNCTManager
from spiking_nct.core.config import SpikingNCTConfig
from spiking_nct.batch_trainer import BatchSpikingNCTClassifier, EpochLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# 常量
# ============================================================
DATA_ROOT = PROJECT_ROOT / 'cats_nct' / 'data'
RESULTS_DIR = Path(__file__).parent / 'results_round2' / 'experiment_9_consciousness_validation'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实验8a最佳LIF参数
BEST_LIF_CONFIG = {
    'lif_threshold': 0.8,
    'lif_decay': 0.99,
    'n_timesteps': 2,
    'surrogate_sigma': 10.0,
}


# ============================================================
# 扰动函数
# ============================================================
def block_occlusion(images: torch.Tensor, level: float) -> torch.Tensor:
    """块遮挡：随机遮挡图像块
    
    Args:
        images: [B, 1, H, W]
        level: 遮挡比例 0.0-1.0 (映射为10%-50%面积)
    """
    if level <= 0:
        return images
    B, C, H, W = images.shape
    occluded = images.clone()
    # 映射 level 0.0-1.0 → 遮挡面积 10%-50%
    occlusion_ratio = 0.1 + level * 0.4
    block_size = int(H * W * occlusion_ratio) ** 0.5
    block_h = max(1, int(block_size))
    block_w = max(1, int(block_size))
    
    for i in range(B):
        top = torch.randint(0, max(1, H - block_h + 1), (1,)).item()
        left = torch.randint(0, max(1, W - block_w + 1), (1,)).item()
        occluded[i, :, top:top+block_h, left:left+block_w] = 0.0
    return occluded


def gaussian_blur(images: torch.Tensor, level: float) -> torch.Tensor:
    """高斯模糊：使用可微分的高斯滤波
    
    Args:
        images: [B, 1, H, W]
        level: 模糊强度 0.0-1.0 (映射为σ=0.5-5.0)
    """
    if level <= 0:
        return images
    # 映射 level 0.0-1.0 → sigma 0.5-5.0
    sigma = 0.5 + level * 4.5
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, 31))
    
    # 创建高斯核
    coords = torch.arange(kernel_size, dtype=torch.float32, device=images.device) - kernel_size // 2
    g = torch.exp(-coords**2 / (2 * sigma**2))
    kernel_1d = g / g.sum()
    
    # 分离卷积
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
    
    padding = kernel_size // 2
    blurred = F.conv2d(images, kernel_2d, padding=padding)
    return blurred


def semantic_destruction(images: torch.Tensor, level: float) -> torch.Tensor:
    """语义破坏：旋转 + 像素重排
    
    Args:
        images: [B, 1, H, W]
        level: 破坏强度 0.0-1.0 (映射为0°-180°旋转 + 重排比例)
    """
    if level <= 0:
        return images
    B, C, H, W = images.shape
    destroyed = images.clone()
    
    # 旋转角度：0°-180°
    angle = level * 180.0
    
    for i in range(B):
        # 旋转
        img = destroyed[i:i+1]
        destroyed[i:i+1] = F.interpolate(
            torch.rot90(img, k=int(angle / 90), dims=[-2, -1]),
            size=(H, W), mode='bilinear', align_corners=False
        )
    
    # 像素重排（按 level 比例随机打乱像素）
    shuffle_ratio = level * 0.5  # 最多打乱50%像素
    if shuffle_ratio > 0:
        n_shuffle = int(H * W * shuffle_ratio)
        for i in range(B):
            flat = destroyed[i].view(-1)
            perm = torch.randperm(flat.size(0), device=flat.device)
            idx = perm[:n_shuffle]
            flat[idx] = flat[perm[:n_shuffle]]  # 随机打乱部分像素
            destroyed[i] = flat.view(C, H, W)
    
    return destroyed


def fgsm_adversarial(images: torch.Tensor, labels: torch.Tensor,
                     model: nn.Module, epsilon: float) -> torch.Tensor:
    """FGSM 对抗样本
    
    Args:
        images: [B, 1, H, W]
        labels: [B]
        model: 目标模型
        epsilon: 扰动强度
    """
    if epsilon <= 0:
        return images
    
    images_adv = images.clone().detach().requires_grad_(True)
    logits, _ = model(images_adv)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    
    # FGSM: x' = x + ε * sign(∇x L)
    perturbation = epsilon * images_adv.grad.sign()
    adv_images = images + perturbation
    return adv_images.detach()


# ============================================================
# Φ值计算（批处理模式）
# ============================================================
def compute_phi_batch(model: BatchSpikingNCTClassifier,
                      images: torch.Tensor) -> List[float]:
    """在批处理模式下计算每个样本的Φ值
    
    直接调用 manager 的 forward_batch 内部流程，获取工作空间特征，
    再使用 neural_activity 路径计算 Phi（D=128 维内部结构，更区分性）。
    """
    model.eval()
    manager = model.manager
    phi_calculator = manager.consciousness_metrics.phi_calculator
    device = next(model.parameters()).device
    
    phi_values = []
    
    with torch.no_grad():
        visual_batch = images.to(device).float()
        if visual_batch.dim() == 3:
            visual_batch = visual_batch.unsqueeze(1)
        
        # 使用 forward_batch 获取工作空间特征
        features, info = model(images)  # features: [B, D]
        
        # 将特征变形为 [B, 1, D] 供 PhiFromAttention 的 neural_activity 路径使用
        neural_activity = features.unsqueeze(1)  # [B, 1, D]
        
        # 创建一个小的注意力图 [B, H, 1, 1] (L=1，会触发 neural_activity 路径)
        B = features.shape[0]
        H = manager.config.n_heads
        dummy_attn = torch.ones(B, H, 1, 1, device=device) / H
        
        # 调用 PhiFromAttention，它会因为 L=1 而使用 neural_activity 路径
        phi_tensor = phi_calculator(dummy_attn, neural_activity)
        phi_values = phi_tensor.tolist()
    
    return phi_values


def evaluate_with_perturbation(
    model: BatchSpikingNCTClassifier,
    test_loader: DataLoader,
    device: torch.device,
    perturbation_fn=None,
    perturbation_level: float = 0.0,
    compute_phi: bool = True,
    n_samples: int = 200,
) -> Dict[str, Any]:
    """施加扰动后评估模型
    
    Returns:
        {'accuracy': float, 'mean_phi': float, 'std_phi': float, 'phi_values': list, ...}
    """
    model.eval()
    correct = 0
    total = 0
    all_phi = []
    all_logits = []
    all_labels = []
    all_salience = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            if isinstance(data, torch.Tensor):
                images = data.to(device).float()
            else:
                continue
            
            labels_tensor = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
            
            # 施加扰动
            if perturbation_fn is not None and perturbation_level > 0:
                images = perturbation_fn(images, perturbation_level)
            
            # 前向传播
            logits, info = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels_tensor).sum().item()
            total += labels_tensor.size(0)
            
            all_logits.append(logits.cpu())
            all_labels.append(labels_tensor.cpu())
            
            # 从 info 提取 salience
            if info.get('attention_weights') is not None:
                attn = info['attention_weights']
                if isinstance(attn, torch.Tensor):
                    salience = attn.max(dim=-1).values.mean().item()
                    all_salience.append(salience)
            
            if total >= n_samples:
                break
    
    # 计算 Φ 值（使用 forward_batch 内部流程，快速且准确）
    if compute_phi and total > 0:
        # 取前 n_samples 个样本
        sample_count = min(n_samples, total)
        sample_images_list = []
        sample_labels_list = []
        count = 0
        for data, labels in test_loader:
            if isinstance(data, torch.Tensor):
                imgs = data
            else:
                continue
            batch_size = imgs.size(0)
            remaining = sample_count - count
            if remaining <= 0:
                break
            take = min(batch_size, remaining)
            sample_images_list.append(imgs[:take])
            count += take
        
        if sample_images_list:
            sample_images = torch.cat(sample_images_list, dim=0).to(device).float()
            # 施加扰动
            if perturbation_fn is not None and perturbation_level > 0:
                sample_images = perturbation_fn(sample_images, perturbation_level)
            all_phi = compute_phi_batch(model, sample_images)
    
    accuracy = correct / max(total, 1)
    mean_phi = np.mean(all_phi) if all_phi else 0.0
    std_phi = np.std(all_phi) if len(all_phi) > 1 else 0.0
    mean_salience = np.mean(all_salience) if all_salience else 0.0
    
    return {
        'accuracy': accuracy,
        'mean_phi': mean_phi,
        'std_phi': std_phi,
        'phi_values': all_phi,
        'mean_salience': mean_salience,
        'n_samples': total,
    }


# ============================================================
# 条件控制实验
# ============================================================
def control_random_classifier(test_loader, device, n_samples=200):
    """对照组1: 随机分类器 — Φ 应接近 0"""
    # 创建随机初始化模型
    config = SpikingNCTConfig(
        d_model=128, n_heads=4, dim_ff=256,
        n_timesteps=BEST_LIF_CONFIG['n_timesteps'],
        consciousness_threshold=0.0,
        attention_mode='hybrid',
        lif_threshold=BEST_LIF_CONFIG['lif_threshold'],
        lif_decay=BEST_LIF_CONFIG['lif_decay'],
        surrogate_sigma=BEST_LIF_CONFIG['surrogate_sigma'],
        energy_monitoring=False,
    )
    manager = SpikingNCTManager(config)
    model = BatchSpikingNCTClassifier(manager, num_classes=10, dropout=0.3)
    model = model.to(device)
    model.eval()
    
    # 用随机权重评估
    result = evaluate_with_perturbation(
        model, test_loader, device,
        perturbation_fn=None, perturbation_level=0.0,
        compute_phi=True, n_samples=n_samples,
    )
    result['control_group'] = 'random_classifier'
    result['description'] = '随机初始化模型（未训练）'
    return result


def control_frozen_intermediate(trained_model, test_loader, device, n_samples=200):
    """对照组2: 冻结中间层 — 仅训练最后一层"""
    # 克隆模型
    import copy
    frozen_model = copy.deepcopy(trained_model)
    
    # 冻结编码器和工作空间（仅保留分类头可训练）
    for name, param in frozen_model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    
    # 仅训练分类头（1个epoch快速微调）
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, frozen_model.parameters()),
        lr=1e-3,
    )
    
    frozen_model.train()
    for epoch in range(3):
        for data, labels in test_loader:
            if isinstance(data, torch.Tensor):
                images = data.to(device).float()
            else:
                continue
            labels_t = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
            
            optimizer.zero_grad()
            logits, _ = frozen_model(images)
            loss = F.cross_entropy(logits, labels_t)
            loss.backward()
            optimizer.step()
            break  # 仅用1个batch快速微调
    
    frozen_model.eval()
    result = evaluate_with_perturbation(
        frozen_model, test_loader, device,
        perturbation_fn=None, perturbation_level=0.0,
        compute_phi=True, n_samples=n_samples,
    )
    result['control_group'] = 'frozen_intermediate'
    result['description'] = '冻结中间层，仅训练分类头'
    return result


def control_temperature_scaling(trained_model, test_loader, device, temperatures=[0.5, 1.0, 1.5, 2.0, 3.0]):
    """对照组3: 温度缩放 — 准确率不变但观察Φ变化"""
    results = []
    
    for temp in temperatures:
        # 通过修改分类头的 logits / temperature
        trained_model.eval()
        correct = 0
        total = 0
        phi_values = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                if isinstance(data, torch.Tensor):
                    images = data.to(device).float()
                else:
                    continue
                labels_t = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)
                
                logits, info = trained_model(images)
                # 温度缩放
                scaled_logits = logits / temp
                preds = scaled_logits.argmax(dim=1)
                correct += (preds == labels_t).sum().item()
                total += labels_t.size(0)
                
                if total >= 1000:
                    break
        
        # 计算Φ（Φ不受温度缩放影响，因为Φ在工作空间层计算）
        phi_result = evaluate_with_perturbation(
            trained_model, test_loader, device,
            perturbation_fn=None, perturbation_level=0.0,
            compute_phi=True, n_samples=1000,
        )
        
        result = {
            'temperature': temp,
            'accuracy': correct / max(total, 1),
            'mean_phi': phi_result['mean_phi'],
            'std_phi': phi_result['std_phi'],
        }
        results.append(result)
        print(f"    T={temp:.1f}: Acc={result['accuracy']:.2%}, Φ={result['mean_phi']:.4f}")
    
    return results


def control_gaussian_noise_input(test_loader, device, n_samples=1000):
    """对照组4: 高斯噪声输入 — 输入完全随机，Φ应最低"""
    # 创建噪声数据加载器
    noise_images = torch.randn(1000, 1, 28, 28, device=device)
    noise_labels = torch.randint(0, 10, (1000,), device=device)
    noise_dataset = torch.utils.data.TensorDataset(noise_images, noise_labels)
    noise_loader = DataLoader(noise_dataset, batch_size=64, shuffle=False)
    
    # 使用训练模型处理噪声输入
    config = SpikingNCTConfig(
        d_model=128, n_heads=4, dim_ff=256,
        n_timesteps=BEST_LIF_CONFIG['n_timesteps'],
        consciousness_threshold=0.0,
        attention_mode='hybrid',
        lif_threshold=BEST_LIF_CONFIG['lif_threshold'],
        lif_decay=BEST_LIF_CONFIG['lif_decay'],
        surrogate_sigma=BEST_LIF_CONFIG['surrogate_sigma'],
        energy_monitoring=False,
    )
    manager = SpikingNCTManager(config)
    model = BatchSpikingNCTClassifier(manager, num_classes=10, dropout=0.3)
    model = model.to(device)
    model.eval()
    
    result = evaluate_with_perturbation(
        model, noise_loader, device,
        perturbation_fn=None, perturbation_level=0.0,
        compute_phi=True, n_samples=n_samples,
    )
    result['control_group'] = 'gaussian_noise_input'
    result['description'] = '高斯噪声输入（完全随机）'
    return result


def control_shuffle_features(trained_model, test_loader, device, n_samples=1000):
    """对照组5: Shuffle特征维度 — 破坏特征间相关性"""
    # 创建wrapper，在forward后shuffle特征
    class ShuffleModel(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
        
        def forward(self, x):
            logits, info = self.base_model(x)
            # Shuffle特征维度（破坏相关性）
            if 'workspace_features' in info:
                features = info['workspace_features']
                B, D = features.shape
                shuffle_idx = torch.randperm(D, device=features.device)
                info['workspace_features'] = features[:, shuffle_idx]
            return logits, info
    
    shuffled_model = ShuffleModel(trained_model)
    shuffled_model.eval()
    
    result = evaluate_with_perturbation(
        shuffled_model, test_loader, device,
        perturbation_fn=None, perturbation_level=0.0,
        compute_phi=True, n_samples=n_samples,
    )
    result['control_group'] = 'shuffle_features'
    result['description'] = 'Shuffle特征维度（破坏相关性）'
    return result


# ============================================================
# 主实验
# ============================================================
def train_base_model() -> BatchSpikingNCTClassifier:
    """训练基础模型（使用8a最佳参数）"""
    print("\n" + "=" * 70)
    print("步骤1: 训练基础模型（实验8a最佳LIF参数）")
    print("=" * 70)
    
    config = SpikingNCTConfig(
        d_model=128, n_heads=4, dim_ff=256,
        n_timesteps=BEST_LIF_CONFIG['n_timesteps'],
        consciousness_threshold=0.0,
        attention_mode='hybrid',
        lif_threshold=BEST_LIF_CONFIG['lif_threshold'],
        lif_decay=BEST_LIF_CONFIG['lif_decay'],
        surrogate_sigma=BEST_LIF_CONFIG['surrogate_sigma'],
        energy_monitoring=False,
    )
    manager = SpikingNCTManager(config)
    model = BatchSpikingNCTClassifier(manager, num_classes=10, dropout=0.3)
    model = model.to(DEVICE)
    
    # 数据
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_dataset = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # 训练20个epoch
    best_acc = 0
    patience_counter = 0
    
    for epoch in range(1, 21):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data, labels in train_loader:
            if isinstance(data, torch.Tensor):
                images = data.to(DEVICE).float()
            else:
                continue
            labels_t = labels.to(DEVICE) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=DEVICE)
            
            optimizer.zero_grad()
            logits, info = model(images)
            loss = F.cross_entropy(logits, labels_t)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * labels_t.size(0)
            correct += (logits.argmax(1) == labels_t).sum().item()
            total += labels_t.size(0)
        
        scheduler.step()
        train_acc = correct / max(total, 1)
        
        # 评估
        model.eval()
        eval_correct = 0
        eval_total = 0
        with torch.no_grad():
            for data, labels in test_loader:
                if isinstance(data, torch.Tensor):
                    images = data.to(DEVICE).float()
                else:
                    continue
                labels_t = labels.to(DEVICE) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=DEVICE)
                logits, _ = model(images)
                eval_correct += (logits.argmax(1) == labels_t).sum().item()
                eval_total += labels_t.size(0)
        
        val_acc = eval_correct / max(eval_total, 1)
        print(f"  Epoch {epoch:2d}: TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, LR={scheduler.get_last_lr()[0]:.2e}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), RESULTS_DIR / 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  早停 @ epoch {epoch}")
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load(RESULTS_DIR / 'best_model.pt', weights_only=True))
    print(f"\n  最佳 ValAcc: {best_acc:.4f}")
    return model


def main():
    print("=" * 70)
    print("实验9: 意识度量深度验证（含条件控制）")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"最佳LIF配置: {BEST_LIF_CONFIG}")
    print()
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # ============================================================
    # 步骤1: 训练基础模型
    # ============================================================
    model_path = RESULTS_DIR / 'best_model.pt'
    if model_path.exists():
        print("\n[跳过训练] 发现已有最佳模型，直接加载...")
        config = SpikingNCTConfig(
            d_model=128, n_heads=4, dim_ff=256,
            n_timesteps=BEST_LIF_CONFIG['n_timesteps'],
            consciousness_threshold=0.0,
            attention_mode='hybrid',
            lif_threshold=BEST_LIF_CONFIG['lif_threshold'],
            lif_decay=BEST_LIF_CONFIG['lif_decay'],
            surrogate_sigma=BEST_LIF_CONFIG['surrogate_sigma'],
            energy_monitoring=False,
        )
        manager = SpikingNCTManager(config)
        model = BatchSpikingNCTClassifier(manager, num_classes=10, dropout=0.3)
        model = model.to(DEVICE)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        print("  模型加载完成")
    else:
        model = train_base_model()
    
    # 准备测试数据
    test_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=test_transform)
    # 取全部10000个测试样本用于评估
    torch.manual_seed(42)
    test_subset = Subset(test_dataset, range(min(10000, len(test_dataset))))
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)
    # 扰动评估专用（不shuffle，保证一致性）
    perturb_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=0)
    
    # ============================================================
    # 步骤2: 基线评估（无扰动）
    # ============================================================
    print("\n" + "=" * 70)
    print("步骤2: 基线评估（无扰动）")
    print("=" * 70)
    
    baseline = evaluate_with_perturbation(
        model, test_loader, DEVICE,
        perturbation_fn=None, perturbation_level=0.0,
        compute_phi=True, n_samples=1000,
    )
    print(f"  基线: Acc={baseline['accuracy']:.4f}, Φ={baseline['mean_phi']:.4f} ± {baseline['std_phi']:.4f}")
    
    # ============================================================
    # 步骤3: 主动扰动实验
    # ============================================================
    print("\n" + "=" * 70)
    print("步骤3: 主动扰动实验")
    print("=" * 70)
    
    # 12个扰动级别，提高统计功效（df=10，r=-0.58即可p<0.05）
    perturbation_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.85, 1.0]
    perturbation_types = {
        'block_occlusion': block_occlusion,
        'gaussian_blur': gaussian_blur,
        'semantic_destruction': semantic_destruction,
    }
    
    all_perturbation_results = {}
    
    for ptype, pfn in perturbation_types.items():
        print(f"\n--- {ptype} ---")
        results_list = []
        
        for level in perturbation_levels:
            print(f"  Level={level:.2f}...", end=" ", flush=True)
            result = evaluate_with_perturbation(
                model, perturb_loader, DEVICE,
                perturbation_fn=pfn, perturbation_level=level,
                compute_phi=True, n_samples=1000,
            )
            result['perturbation_type'] = ptype
            result['perturbation_level'] = level
            results_list.append(result)
            print(f"Acc={result['accuracy']:.4f}, Φ={result['mean_phi']:.4f}")
        
        all_perturbation_results[ptype] = results_list
    
    # ============================================================
    # 步骤4: 条件控制实验
    # ============================================================
    print("\n" + "=" * 70)
    print("步骤4: 条件控制实验")
    print("=" * 70)
    
    # 对照组1: 随机分类器
    print("\n--- 对照组1: 随机分类器 ---")
    random_result = control_random_classifier(test_loader, DEVICE, n_samples=1000)
    print(f"  Acc={random_result['accuracy']:.4f}, Φ={random_result['mean_phi']:.4f}")
    
    # 对照组2: 冻结中间层
    print("\n--- 对照组2: 冻结中间层 ---")
    frozen_result = control_frozen_intermediate(model, test_loader, DEVICE, n_samples=1000)
    print(f"  Acc={frozen_result['accuracy']:.4f}, Φ={frozen_result['mean_phi']:.4f}")
    
    # 对照组3: 温度缩放
    print("\n--- 对照组3: 温度缩放 ---")
    temp_results = control_temperature_scaling(model, test_loader, DEVICE)
    
    # 对照组4: 高斯噪声输入
    print("\n--- 对照组4: 高斯噪声输入 ---")
    noise_result = control_gaussian_noise_input(test_loader, DEVICE, n_samples=1000)
    print(f"  Acc={noise_result['accuracy']:.4f}, Φ={noise_result['mean_phi']:.4f}")
    
    # 对照组5: Shuffle特征维度
    print("\n--- 对照组5: Shuffle特征维度 ---")
    shuffle_result = control_shuffle_features(model, test_loader, DEVICE, n_samples=1000)
    print(f"  Acc={shuffle_result['accuracy']:.4f}, Φ={shuffle_result['mean_phi']:.4f}")
    
    # ============================================================
    # 步骤5: 统计分析
    # ============================================================
    print("\n" + "=" * 70)
    print("步骤5: 统计分析")
    print("=" * 70)
    
    statistics = {}
    
    # H4a: Φ-扰动相关性
    for ptype, results_list in all_perturbation_results.items():
        levels = [r['perturbation_level'] for r in results_list]
        phis = [r['mean_phi'] for r in results_list]
        accs = [r['accuracy'] for r in results_list]
        
        if len(set(phis)) > 1 and len(set(levels)) > 1:
            r_phi, p_phi = scipy_stats.pearsonr(levels, phis)
            r_acc, p_acc = scipy_stats.pearsonr(levels, accs)
            r_phi_acc, p_phi_acc = scipy_stats.pearsonr(phis, accs) if len(set(phis)) > 1 and len(set(accs)) > 1 else (0, 1)
        else:
            r_phi, p_phi, r_acc, p_acc, r_phi_acc, p_phi_acc = 0, 1, 0, 1, 0, 1
        
        statistics[ptype] = {
            'phi_vs_level_r': r_phi,
            'phi_vs_level_p': p_phi,
            'acc_vs_level_r': r_acc,
            'acc_vs_level_p': p_acc,
            'phi_vs_acc_r': r_phi_acc,
            'phi_vs_acc_p': p_phi_acc,
        }
        
        print(f"\n  {ptype}:")
        print(f"    Φ vs 扰动强度: r={r_phi:.3f}, p={p_phi:.6f}")
        print(f"    Acc vs 扰动强度: r={r_acc:.3f}, p={p_acc:.6f}")
        print(f"    Φ vs Acc: r={r_phi_acc:.3f}, p={p_phi_acc:.6f}")
    
    # H4b: 对照组分析
    print(f"\n  对照组分析:")
    print(f"    随机分类器: Φ={random_result['mean_phi']:.4f} (预期<正常)")
    print(f"    冻结中间层: Φ={frozen_result['mean_phi']:.4f} (预期<正常)")
    print(f"    高斯噪声输入: Φ={noise_result['mean_phi']:.4f} (预期最低)")
    print(f"    Shuffle特征: Φ={shuffle_result['mean_phi']:.4f} (预期<正常)")
    print(f"    正常模型: Φ={baseline['mean_phi']:.4f}")
    
    # 假设验证
    print("\n" + "=" * 70)
    print("假设验证结果")
    print("=" * 70)
    
    # H4a: 扰动显著降低Φ
    h4a_results = {}
    for ptype, stat in statistics.items():
        phi_decreases = stat['phi_vs_level_r'] < 0 and stat['phi_vs_level_p'] < 0.05
        h4a_results[ptype] = phi_decreases
        status = "✅ 通过" if phi_decreases else "❌ 未通过"
        print(f"  H4a ({ptype}): Φ随扰动显著降低 → {status}")
    
    # H4b: Φ不是准确率的简单代理
    random_phi_lower = random_result['mean_phi'] < baseline['mean_phi']
    frozen_phi_lower = frozen_result['mean_phi'] < baseline['mean_phi']
    noise_phi_lowest = noise_result['mean_phi'] < baseline['mean_phi']
    shuffle_phi_lower = shuffle_result['mean_phi'] < baseline['mean_phi']
    
    h4b_random = "✅ 通过" if random_phi_lower else "❌ 未通过"
    h4b_frozen = "✅ 通过" if frozen_phi_lower else "❌ 未通过"
    h4b_noise = "✅ 通过" if noise_phi_lowest else "❌ 未通过"
    h4b_shuffle = "✅ 通过" if shuffle_phi_lower else "❌ 未通过"
    
    print(f"  H4b (随机分类器Φ<正常): {h4b_random}")
    print(f"  H4b (冻结中间层Φ<正常): {h4b_frozen}")
    print(f"  H4b (高斯噪声输入Φ最低): {h4b_noise}")
    print(f"  H4b (Shuffle特征Φ<正常): {h4b_shuffle}")
    
    # ============================================================
    # 步骤6: 保存结果
    # ============================================================
    print("\n" + "=" * 70)
    print("步骤6: 保存结果")
    print("=" * 70)
    
    # 序列化结果（排除不可JSON化的字段）
    def serialize_result(r):
        s = {}
        for k, v in r.items():
            if isinstance(v, (np.floating, np.integer)):
                s[k] = float(v)
            elif isinstance(v, list):
                s[k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
            else:
                s[k] = v
        return s
    
    output = {
        'experiment': 'experiment_9_consciousness_validation',
        'timestamp': datetime.now().isoformat(),
        'lif_config': BEST_LIF_CONFIG,
        'baseline': serialize_result(baseline),
        'perturbation_results': {
            k: [serialize_result(r) for r in v] 
            for k, v in all_perturbation_results.items()
        },
        'control_random_classifier': serialize_result(random_result),
        'control_frozen_intermediate': serialize_result(frozen_result),
        'control_temperature_scaling': temp_results,
        'control_gaussian_noise_input': serialize_result(noise_result),
        'control_shuffle_features': serialize_result(shuffle_result),
        'statistics': statistics,
        'hypothesis_testing': {
            'H4a_perturbation_reduces_phi': h4a_results,
            'H4b_random_phi_lower': random_phi_lower,
            'H4b_frozen_phi_lower': frozen_phi_lower,
            'H4b_noise_phi_lowest': noise_phi_lowest,
            'H4b_shuffle_phi_lower': shuffle_phi_lower,
        },
    }
    
    output_path = RESULTS_DIR / 'experiment_9_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"  结果已保存: {output_path}")
    
    # ============================================================
    # 步骤7: 生成可视化
    # ============================================================
    print("\n" + "=" * 70)
    print("步骤7: 生成可视化")
    print("=" * 70)
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 图1: Φ vs 扰动强度
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    perturb_labels = {
        'block_occlusion': 'Block Occlusion',
        'gaussian_blur': 'Gaussian Blur',
        'semantic_destruction': 'Semantic Destruction',
    }
    colors = {'block_occlusion': '#e74c3c', 'gaussian_blur': '#3498db', 'semantic_destruction': '#2ecc71'}
    
    for idx, (ptype, results_list) in enumerate(all_perturbation_results.items()):
        ax = axes[idx]
        levels = [r['perturbation_level'] for r in results_list]
        phis = [r['mean_phi'] for r in results_list]
        phi_stds = [r['std_phi'] for r in results_list]
        accs = [r['accuracy'] for r in results_list]
        
        ax2 = ax.twinx()
        l1 = ax.plot(levels, phis, 'o-', color=colors[ptype], label='Φ', linewidth=2, markersize=8)
        ax.fill_between(levels, [p-s for p,s in zip(phis, phi_stds)], [p+s for p,s in zip(phis, phi_stds)], 
                        color=colors[ptype], alpha=0.15)
        l2 = ax2.plot(levels, accs, 's--', color='gray', label='Accuracy', linewidth=1.5, markersize=6)
        
        ax.set_xlabel('Perturbation Level', fontsize=11)
        ax.set_ylabel('Φ Value', fontsize=11, color=colors[ptype])
        ax2.set_ylabel('Accuracy', fontsize=11, color='gray')
        ax.set_title(perturb_labels.get(ptype, ptype), fontsize=12)
        ax.grid(alpha=0.3)
        
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='lower left', fontsize=9)
    
    plt.suptitle('Experiment 9: Φ Value vs Perturbation Intensity', fontsize=14, y=1.02)
    plt.tight_layout()
    fig_path = RESULTS_DIR / 'fig1_phi_vs_perturbation.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图1: Φ vs 扰动强度 → {fig_path}")
    
    # 图2: 对照组比较
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = ['Random\nClassifier', 'Frozen\nIntermediate', 'Normal\nModel']
    phi_vals = [random_result['mean_phi'], frozen_result['mean_phi'], baseline['mean_phi']]
    phi_errs = [random_result['std_phi'], frozen_result['std_phi'], baseline['std_phi']]
    
    bars = ax.bar(groups, phi_vals, yerr=phi_errs, 
                  color=['#e74c3c', '#f39c12', '#2ecc71'], alpha=0.85,
                  capsize=5, edgecolor='black', linewidth=1)
    ax.set_ylabel('Φ Value', fontsize=12)
    ax.set_title('Experiment 9: Control Group Comparison', fontsize=14)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, phi_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    fig_path = RESULTS_DIR / 'fig2_control_groups.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  图2: 对照组比较 → {fig_path}")
    
    # 图3: 温度缩放
    if temp_results:
        fig, ax1 = plt.subplots(figsize=(8, 5))
        temps = [r['temperature'] for r in temp_results]
        t_accs = [r['accuracy'] for r in temp_results]
        t_phis = [r['mean_phi'] for r in temp_results]
        
        ax1.plot(temps, t_accs, 'o-', color='#3498db', label='Accuracy', linewidth=2)
        ax1.set_xlabel('Temperature', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12, color='#3498db')
        
        ax2 = ax1.twinx()
        ax2.plot(temps, t_phis, 's--', color='#e74c3c', label='Φ Value', linewidth=2)
        ax2.set_ylabel('Φ Value', fontsize=12, color='#e74c3c')
        
        ax1.set_title('Experiment 9: Temperature Scaling', fontsize=14)
        ax1.grid(alpha=0.3)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, loc='center right', fontsize=9)
        
        plt.tight_layout()
        fig_path = RESULTS_DIR / 'fig3_temperature_scaling.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  图3: 温度缩放 → {fig_path}")
    
    print("\n" + "=" * 70)
    print("实验9 完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
