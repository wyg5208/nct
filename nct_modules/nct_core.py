"""
NeuroConscious Transformer (NCT) - 下一代神经形态意识架构
NeuroConscious Transformer: Next-Generation Neuromorphic Consciousness Architecture

核心理论创新：
1. Attention-Based Global Workspace - 用多头注意力替代简单竞争
2. Transformer-STDP Hybrid Learning - 全局调制的突触可塑性
3. Predictive Coding as Decoder-Only Transformer - Friston 自由能原理 = Transformer 训练目标
4. Multi-Modal Cross-Attention Fusion - 真正的语义级多模态整合
5. γ-Synchronization as Update Cycle - γ同步作为 Transformer 更新周期

架构总览：
┌─────────────────────────────────────────────┐
│   Multi-Modal Encoders (ViT/Audio/Intero)   │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│   Cross-Modal Integration Layer             │
│   (8-head Cross-Attention)                  │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│   Global Workspace Transformer (4 layers)   │
│   - Causal Self-Attention (预测编码)         │
│   - Transformer-STDP Hybrid Learning        │
│   - γ-Synchronization as Update Cycle       │
└────────────────┬────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────┐
│   Consciousness Metrics Head                │
│   - Φ Calculator (from attention flow)      │
│   - Awareness Level (from loss landscape)   │
└─────────────────────────────────────────────┘

作者：WinClaw Research Team
创建：2026 年 2 月 21 日
版本：v3.0.0-alpha (NeuroConscious Transformer)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class NCTConfig:
    """NeuroConscious Transformer 配置
    
    生物合理性参数：
    - n_heads=8: 对应 Miller's Law 7±2，工作空间容量
    - n_layers=4: 对应皮层层次结构（L1-L4）
    - d_model=768: 类似 BERT-base，足够的表征能力
    - gamma_freq=40Hz: γ波频率，意识同步标志
    """
    # 架构参数
    n_heads: int = 8                    # 注意力头数（≈工作空间容量）
    n_layers: int = 4                   # Transformer 层数（≈皮层层次）
    d_model: int = 768                  # 模型维度
    dim_ff: int = 3072                  # 前馈网络维度（4×d_model）
    
    # 多模态参数
    visual_patch_size: int = 4          # 视觉 patch 大小
    visual_embed_dim: int = 256         # 视觉 embedding 维度
    audio_embed_dim: int = 256          # 音频 embedding 维度
    intero_embed_dim: int = 256         # 内感受 embedding 维度
    
    # 神经生物学参数
    gamma_freq: float = 40.0            # γ波频率（Hz）
    consciousness_threshold: float = 0.7  # 意识阈值
    stdp_learning_rate: float = 0.01    # STDP 学习率
    attention_modulation_lambda: float = 0.1  # 注意力调制强度λ
    
    # 训练参数
    dropout: float = 0.1                # Dropout 率
    max_seq_len: int = 512              # 最大序列长度
    batch_first: bool = True            # batch 优先格式
    
    def __post_init__(self):
        """验证配置合理性"""
        assert self.n_heads > 0 and self.n_layers > 0
        assert self.d_model % self.n_heads == 0, "d_model 必须能被 n_heads 整除"
        logger.info(f"[NCTConfig] 初始化配置：{self.n_heads} heads, {self.n_layers} layers, d_model={self.d_model}")


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class NCTConsciousContent:
    """NCT 意识内容（增强版 ConsciousContent）
    
    包含完整的 Transformer 状态信息，支持：
    - 注意力图谱可视化
    - Φ值追踪
    - 预测误差监控
    """
    content_id: str
    representation: np.ndarray          # 意识表征向量 [B, D]
    salience: float = 0.0               # 显著性（来自 attention weights）
    gamma_phase: float = 0.0            # γ波相位
    timestamp: float = field(default_factory=time.time)
    
    # Transformer 特有字段
    attention_maps: Optional[np.ndarray] = None  # [B, H, L, L] 注意力图谱
    phi_value: float = 0.0              # 整合信息量Φ
    prediction_error: float = 0.0       # 预测误差（自由能）
    awareness_level: str = "minimal"    # 意识水平
    
    # 多模态来源追踪
    modality_weights: Optional[Dict[str, float]] = None  # 各模态贡献度
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典格式（兼容旧系统）"""
        return {
            'content_id': self.content_id,
            'representation': self.representation,
            'salience': self.salience,
            'gamma_phase': self.gamma_phase,
            'timestamp': self.timestamp,
            'phi_value': self.phi_value,
            'awareness_level': self.awareness_level,
            'modality_weights': self.modality_weights,
        }


# ============================================================================
# 模块 1: 多模态编码器
# ============================================================================

class MultiModalEncoder(nn.Module):
    """多模态编码器
    
    将不同感觉模态映射到统一的 embedding 空间 [B, N, D]
    
    生物合理性：
    - Visual Encoder: 模拟 V1→V2→V4→IT 腹侧通路（ViT 架构）
    - Audio Encoder: 模拟 A1→A2→Belt 听觉通路（频谱 Transformer）
    - Interoceptive Encoder: 模拟岛叶内感受处理（MLP）
    """
    
    def __init__(self, config: NCTConfig):
        super().__init__()
        self.config = config
        
        # 1. 视觉编码器（Vision Transformer 简化版）
        self.visual_encoder = VisionTransformer(
            patch_size=config.visual_patch_size,
            embed_dim=config.visual_embed_dim,
            n_heads=8,
            n_layers=4,
        )
        
        # 2. 听觉编码器（频谱 Transformer）
        self.audio_encoder = AudioSpectrogramTransformer(
            embed_dim=config.audio_embed_dim,
            n_heads=8,
        )
        
        # 3. 内感受编码器（简单 MLP）
        self.intero_encoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.GELU(),
            nn.Linear(64, config.intero_embed_dim),
            nn.LayerNorm(config.intero_embed_dim),
        )
        
        # 4. 模态融合投影（统一到 d_model）
        self.modal_projection = nn.ModuleDict({
            'visual': nn.Linear(config.visual_embed_dim, config.d_model),
            'audio': nn.Linear(config.audio_embed_dim, config.d_model),
            'intero': nn.Linear(config.intero_embed_dim, config.d_model),
        })
        
    def forward(self, sensory_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """多模态编码
        
        Args:
            sensory_data: 字典格式的感觉输入
                - 'visual': [B, H, W] 或 [B, T, H, W]
                - 'auditory': [B, T, F] 语谱图
                - 'interoceptive': [B, 10] 内感受向量
        
        Returns:
            字典格式的 embeddings:
                - 'visual_emb': [B, N_v, D]
                - 'audio_emb': [B, N_a, D]
                - 'intero_emb': [B, 1, D]
        """
        embeddings = {}
        
        # 视觉编码
        if 'visual' in sensory_data:
            visual_input = sensory_data['visual']
            visual_emb = self.visual_encoder(visual_input)
            # 投影到统一维度
            visual_emb = self.modal_projection['visual'](visual_emb)
            embeddings['visual_emb'] = visual_emb
            logger.debug(f"[MultiModalEncoder] 视觉编码：{visual_input.shape} → {visual_emb.shape}")
        
        # 听觉编码
        if 'auditory' in sensory_data:
            audio_input = sensory_data['auditory']
            audio_emb = self.audio_encoder(audio_input)
            audio_emb = self.modal_projection['audio'](audio_emb)
            embeddings['audio_emb'] = audio_emb
            logger.debug(f"[MultiModalEncoder] 听觉编码：{audio_input.shape} → {audio_emb.shape}")
        
        # 内感受编码
        if 'interoceptive' in sensory_data:
            intero_input = sensory_data['interoceptive']
            intero_emb = self.intero_encoder(intero_input)
            intero_emb = intero_emb.unsqueeze(1)  # [B, 1, D]
            intero_emb = self.modal_projection['intero'](intero_emb)
            embeddings['intero_emb'] = intero_emb
            logger.debug(f"[MultiModalEncoder] 内感受编码：{intero_input.shape} → {intero_emb.shape}")
        
        return embeddings


class VisionTransformer(nn.Module):
    """简化版 Vision Transformer（用于视觉编码）
    
    架构：
    Patch Embedding → Position Embedding → Transformer Encoder → Output
    
    生物映射：
    - Patch Embedding: V1 简单细胞的感受野
    - Self-Attention: V4 复杂细胞的全局整合
    - Output: IT 皮层的物体表征
    """
    
    def __init__(self, patch_size: int = 4, embed_dim: int = 256, 
                 n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # 1. Patch Embedding（将图像分块并投影）
        self.patch_embed = nn.Conv2d(
            in_channels=1,  # 假设单通道输入
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # 2. 位置编码（可学习）
        self.pos_embed = nn.Parameter(torch.randn(1, 64, embed_dim))  # 最多 64 个 patches
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """视觉编码
        
        Args:
            x: 输入张量 [B, H, W] 或 [B, T, H, W]
        
        Returns:
            视觉 embedding [B, N_patches, embed_dim]
        """
        # 确保是 4D 输入 [B, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, 1, H, W]
        elif x.dim() == 4 and x.shape[1] != 1:
            # 如果是 [B, T, H, W]，取平均或第一帧
            x = x.mean(dim=1, keepdim=True)
        
        B = x.shape[0]
        
        # Patch Embedding
        x = self.patch_embed(x)  # [B, D, H', W']
        x = x.flatten(2)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        
        # 添加位置编码
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer 编码
        x = self.transformer(x)  # [B, N, D]
        
        return x


class AudioSpectrogramTransformer(nn.Module):
    """音频语谱图 Transformer
    
    架构：
    Spectrogram → Patch Embedding → Positional Encoding → Transformer
    
    生物映射：
    - Patch Embedding: 耳蜗基底膜的频率分析
    - Self-Attention: 听觉皮层的时间模式整合
    """
    
    def __init__(self, embed_dim: int = 256, n_heads: int = 8, n_layers: int = 4):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 1. 频谱 Patch Embedding（沿时间轴分块）
        self.spectrogram_embed = nn.Conv1d(
            in_channels=1,  # 单通道语谱图
            out_channels=embed_dim,
            kernel_size=10,  # 10 帧为一个 patch
            stride=10
        )
        
        # 2. 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 50, embed_dim))  # 最多 50 个时间 patches
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """音频编码
        
        Args:
            x: 语谱图 [B, T, F] 或 [B, F, T]
        
        Returns:
            音频 embedding [B, N_patches, embed_dim]
        """
        # 确保是 [B, F, T] 格式
        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)  # [B, T, F] → [B, F, T]
        
        B, F, T = x.shape
        
        # 展平为 [B*F, 1, T] 以便 Conv1d 处理
        x = x.transpose(1, 2).reshape(B * F, 1, T)
        
        # Patch Embedding
        x = self.spectrogram_embed(x)  # [B*F, D, T']
        x = x.flatten(2).transpose(1, 2)  # [B*F, N, D]
        
        # 位置编码
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Transformer
        x = self.transformer(x)  # [B*F, N, D]
        
        # 重塑回 [B, F*N, D]
        _, N, D = x.shape
        x = x.reshape(B, F * N, D)
        
        return x


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'NCTConfig',
    'NCTConsciousContent',
    'MultiModalEncoder',
    'VisionTransformer',
    'AudioSpectrogramTransformer',
]
