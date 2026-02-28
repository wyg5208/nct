"""
NeuroConscious Transformer - Gamma Synchronization Mechanism
γ同步机制：作为 Transformer 更新周期的相位锁定

核心理论创新：
1. γ振荡（30-80Hz）控制 Transformer 的更新节奏
2. Phase locking = 注意力同步
3. 40Hz 周期 = Transformer forward pass 的时间窗口

生物合理性：
- PING 模型（Pyramidal-Interneuron Network Gamma）
- PV+ 中间神经元的快速发放产生γ振荡
- γ同步解决 binding problem（特征整合问题）

数学形式：
phase(t) = 2π · (t mod T) / T, where T = 1000/40 ms
synchronized_update = update · cos(phase)

作者：WinClaw Research Team
创建：2026 年 2 月 21 日
版本：v3.1.0
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# γ振荡器
# ============================================================================

class GammaSynchronizer(nn.Module):
    """γ同步器
    
    功能：
    1. 生成 40Hz 正弦振荡
    2. 提供相位参考给所有模块
    3. 调制 Transformer 更新节奏
    
    生物物理实现：
    - PV+ 中间神经元网络
    - GABA_A 受体介导的快速抑制
    - PING 机制（Pyramidal-Interneuron Network Gamma）
    """
    
    def __init__(
        self,
        frequency: float = 40.0,
        phase_offset: float = 0.0,
    ):
        super().__init__()
        
        self.frequency = frequency  # Hz
        self.period_ms = 1000.0 / frequency  # γ周期（毫秒）
        self.phase_offset = phase_offset  # 初始相位偏移
        
        self.start_time = time.time()
        
        logger.info(
            f"[GammaSynchronizer] 初始化："
            f"f={frequency}Hz, period={self.period_ms:.2f}ms"
        )
    
    def get_current_phase(self, current_time: Optional[float] = None) -> float:
        """获取当前相位（弧度）
        
        Args:
            current_time: 当前时间戳，None 使用系统时间
        
        Returns:
            phase: [0, 2π]
        """
        if current_time is None:
            current_time = time.time()
        
        elapsed = current_time - self.start_time
        phase = 2 * np.pi * (elapsed % self.period_ms) / self.period_ms
        phase += self.phase_offset
        phase = phase % (2 * np.pi)
        
        return float(phase)
    
    def get_gamma_cycle(self) -> float:
        """获取当前γ周期数"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        cycle = int(elapsed / self.period_ms)
        return cycle
    
    def modulate_by_phase(
        self,
        signal: torch.Tensor,
        current_time: Optional[float] = None,
    ) -> torch.Tensor:
        """用γ相位调制信号
        
        Args:
            signal: 输入信号
            current_time: 当前时间
        
        Returns:
            modulated_signal: 调制后的信号
        """
        phase = self.get_current_phase(current_time)
        
        # 余弦调制（峰期增强，谷期抑制）
        modulation = torch.cos(torch.tensor(phase))
        
        # 归一化到 [0, 1] 范围
        modulation = (modulation + 1.0) / 2.0
        
        modulated_signal = signal * modulation
        
        logger.debug(
            f"[GammaSynchronizer] 相位调制："
            f"phase={phase:.2f} rad, modulation={modulation:.3f}"
        )
        
        return modulated_signal
    
    def synchronize_modules(
        self,
        modules: List[nn.Module],
        current_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """同步所有模块的更新
        
        Args:
            modules: 需要同步的模块列表
            current_time: 当前时间
        
        Returns:
            sync_info: 同步信息
        """
        phase = self.get_current_phase(current_time)
        cycle = self.get_gamma_cycle()
        
        # 在γ峰期（phase ≈ 0）执行更新
        is_peak_phase = abs(phase) < (np.pi / 4) or abs(phase - 2*np.pi) < (np.pi / 4)
        
        sync_info = {
            'phase': phase,
            'cycle': cycle,
            'is_peak_phase': is_peak_phase,
            'modules_synchronized': len(modules),
            'timestamp': current_time or time.time(),
        }
        
        logger.debug(
            f"[GammaSynchronizer] 同步 {len(modules)} 个模块："
            f"cycle={cycle}, phase={phase:.2f} rad, "
            f"is_peak={is_peak_phase}"
        )
        
        return sync_info


# ============================================================================
# 相位编码层
# ============================================================================

class PhaseEncodingLayer(nn.Module):
    """相位编码层
    
    将γ相位信息编码到神经表征中
    
    理论依据：
    - γ相位编码携带语义信息（Fries, 2005）
    - 相位进动（phase precession）支持时间序列学习
    """
    
    def __init__(self, d_model: int = 768, n_freqs: int = 8):
        super().__init__()
        
        self.d_model = d_model
        self.n_freqs = n_freqs
        
        # 可学习的相位 embedding
        self.phase_embedding = nn.Parameter(torch.randn(n_freqs, d_model))
        
        # 投影层
        self.projection = nn.Linear(d_model + n_freqs, d_model)
        
        logger.info("[PhaseEncodingLayer] 初始化")
    
    def forward(
        self,
        x: torch.Tensor,
        gamma_phase: float,
    ) -> torch.Tensor:
        """添加相位编码
        
        Args:
            x: 输入表征 [B, D]
            gamma_phase: γ相位（弧度）
        
        Returns:
            encoded: 相位编码后的表征 [B, D]
        """
        B, D = x.shape
        
        # 将相位分解为多个频率分量
        phase_features = []
        for i in range(self.n_freqs):
            freq = 2 * np.pi * (i + 1)
            sin_component = torch.sin(torch.tensor([freq * gamma_phase]) * torch.ones(B))
            cos_component = torch.cos(torch.tensor([freq * gamma_phase]) * torch.ones(B))
            phase_features.append(sin_component)
            phase_features.append(cos_component)
        
        # 拼接相位特征
        phase_features = torch.stack(phase_features, dim=1)  # [B, n_freqs*2]
        
        # 与原始表征结合
        combined = torch.cat([x, phase_features[:, :self.n_freqs]], dim=1)
        
        # 投影回原维度
        encoded = self.projection(combined)
        
        return encoded


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'GammaSynchronizer',
    'PhaseEncodingLayer',
]
