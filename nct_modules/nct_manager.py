"""
NeuroConscious Transformer - NCT Manager
NCT 总控制器：集成所有子模块到 NeuroConsciousnessManager

功能：
1. 整合所有 NCT 子模块
2. process_cycle() 接口保持向后兼容
3. 支持渐进式迁移（可切换新旧系统）

架构集成：
```
sensory_data → MultiModalEncoder → CrossModalIntegration 
              → AttentionGlobalWorkspace → ConsciousnessMetrics
              → PredictiveCodingDecoder → GammaSynchronizer
              → output (NeuroConsciousnessState)
```

作者：WinClaw Research Team
创建：2026 年 2 月 21 日
版本：v3.0.0-alpha
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# 导入 PredictiveHierarchy 的扩展方法
try:
    from .patch_predictive_hierarchy import *  # noqa
except ImportError:
    pass

from .nct_core import NCTConfig, NCTConsciousContent, MultiModalEncoder
from .nct_cross_modal import CrossModalIntegration
from .nct_workspace import AttentionGlobalWorkspace, AttentionWorkspaceState
from .nct_hybrid_learning import TransformerSTDP, STDPEvent
from .nct_predictive_coding import PredictiveCodingDecoder, PredictiveHierarchy
from .nct_metrics import ConsciousnessMetrics, ConsciousnessLevel
from .nct_gamma_sync import GammaSynchronizer

logger = logging.getLogger(__name__)


# ============================================================================
# NCT Manager - 总控制器
# ============================================================================

class NCTManager(nn.Module):
    """NeuroConscious Transformer 总管理器
    
    完整集成 NCT 架构的所有组件，提供统一的 process_cycle 接口
    
    特性：
    1. 端到端可训练（PyTorch 原生）
    2. 向后兼容旧版 NeuroConsciousnessManager
    3. 支持实时意识状态监控
    """
    
    def __init__(self, config: Optional[NCTConfig] = None):
        super().__init__()
        
        self.config = config or NCTConfig()
        
        # 1. 多模态编码器
        self.multimodal_encoder = MultiModalEncoder(self.config)
        
        # 2. Cross-Modal 整合
        self.cross_modal_integration = CrossModalIntegration(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
        )
        
        # 3. Attention Global Workspace
        self.attention_workspace = AttentionGlobalWorkspace(
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            dim_ff=self.config.dim_ff,
            gamma_freq=self.config.gamma_freq,
        )
        
        # 4. Transformer-STDP 学习
        self.hybrid_learner = TransformerSTDP(
            n_neurons=self.config.d_model,  # 使用 d_model 作为神经元数
            d_model=self.config.d_model,
            stdp_learning_rate=self.config.stdp_learning_rate,
            attention_modulation_lambda=self.config.attention_modulation_lambda,
        )
        
        # 5. 预测编码层次
        self.predictive_hierarchy = PredictiveHierarchy(
            config={
                'layer0_dim': self.config.d_model,
                'layer1_dim': self.config.d_model,
                'layer2_dim': self.config.d_model,
                'layer3_dim': self.config.d_model,
                'n_heads': self.config.n_heads,
            }
        )
        
        # 6. 意识度量
        self.consciousness_metrics = ConsciousnessMetrics()
        
        # 7. γ同步器
        self.gamma_synchronizer = GammaSynchronizer(
            frequency=self.config.gamma_freq
        )
        
        # 新增：感觉历史队列（用于 PredictiveHierarchy）
        from collections import deque
        self.sensory_history = deque(maxlen=10)  # 保留最近 10 个周期的 integrated 表征
        
        # 运行状态
        self.is_running = False
        self.total_cycles = 0
        
        logger.info(f"[NCTManager] 初始化完成，配置：{self.config}")
    
    def start(self):
        """启动系统"""
        self.is_running = True
        self.total_cycles = 0
        logger.info("[NCTManager] 系统启动")
    
    def stop(self):
        """停止系统"""
        self.is_running = False
        logger.info("[NCTManager] 系统停止")
    
    def process_cycle(
        self,
        sensory_data: Dict[str, np.ndarray],
        neurotransmitter_state: Optional[Dict[str, float]] = None,
    ) -> NCTConsciousnessState:
        """处理一个意识周期（~100ms）
        
        Args:
            sensory_data: 感觉输入字典
                - 'visual': [H, W] 或 [T, H, W]
                - 'auditory': [T, F] 语谱图
                - 'interoceptive': [10] 内感受向量
            neurotransmitter_state: 神经递质状态
        
        Returns:
            state: 意识状态
        """
        if not self.is_running:
            logger.warning("[NCTManager] 系统未运行，调用 start()")
            self.start()
        
        current_time = time.time()
        self.total_cycles += 1
        
        # Step 1: 转为 PyTorch 张量
        sensory_tensors = {}
        for modality, data in sensory_data.items():
            if isinstance(data, np.ndarray):
                sensory_tensors[modality] = torch.from_numpy(data).float().unsqueeze(0)
        
        # Step 2: 多模态编码
        embeddings = self.multimodal_encoder(sensory_tensors)
        
        # Step 3: Cross-Modal 整合
        integrated, cross_modal_info = self.cross_modal_integration(embeddings)
        
        # Step 4: 预测编码（使用感觉历史）
        # 将当前 integrated 加入历史
        self.sensory_history.append(integrated)  # integrated shape: [B=1, D]
        
        # 如果有足够的历史，使用序列进行预测
        if len(self.sensory_history) >= 2:
            # 堆叠为序列 [B, T, D]
            sensory_sequence = torch.stack(list(self.sensory_history), dim=1)
            prediction_results = self.predictive_hierarchy.forward_with_sequence(
                sensory_sequence
            )
        else:
            # 历史不足，使用简化版本
            prediction_results = {'total_free_energy': 0.5}
        
        prediction_error = prediction_results.get('total_free_energy', 0.5)
        
        # Step 5: Attention Global Workspace 选择意识内容
        # integrated shape: [B, 1, D] -> squeeze to [D]
        candidate_list = [integrated.squeeze(0)]  # List[[D]]
        winner_state, workspace_info = self.attention_workspace(
            candidates=candidate_list,
            neuromodulator_state=neurotransmitter_state,
        )
        
        # Step 6: 计算意识度量
        if winner_state is not None and winner_state.attention_maps is not None:
            # 使用 integrated 作为 neural_activity 来估计Φ
            metrics = self.consciousness_metrics(
                attention_maps=winner_state.attention_maps,
                neural_activity=integrated.unsqueeze(0),  # [B=1, L=1, D]
                prediction_error=prediction_error,
                neurotransmitter_state=neurotransmitter_state,
            )
        else:
            metrics = {'overall_score': 0.0, 'consciousness_level': 'unconscious'}
        
        # Step 7: γ同步
        gamma_info = self.gamma_synchronizer.get_current_phase(current_time)
        
        # Step 8: 构建意识状态
        state = NCTConsciousnessState(
            workspace_content=winner_state.to_conscious_content() if winner_state else None,
            self_representation=self._infer_self_representation(prediction_error),  # 使用 prediction_error
            neurotransmitter_state=neurotransmitter_state or {},
            consciousness_metrics=metrics,
            timestamp=current_time,
            cycle_number=self.total_cycles,
        )
        
        # Step 9: 诊断信息
        state.diagnostics = {
            'cross_modal': cross_modal_info,
            'workspace': workspace_info,
            'prediction_error': prediction_error,
            'gamma_phase': gamma_info,
            'metrics': metrics,
        }
        
        logger.debug(
            f"[NCTManager] 周期 {self.total_cycles} 完成："
            f"content={'yes' if state.workspace_content else 'no'}, "
            f"level={metrics.get('consciousness_level', 'unknown')}"
        )
        
        return state
    
    def _infer_self_representation(
        self, 
        prediction_error: float  # 接受 prediction error（float）
    ) -> Dict[str, Any]:
        """推断自我表征（简化版）"""
        # 从预测误差中提取自信度
        free_energy = prediction_error
        confidence = 1.0 / (1.0 + free_energy)
        
        return {
            'confidence': confidence,
            'free_energy': free_energy,
            'identity_narrative': f"NCT 自我模型 @ cycle {self.total_cycles}",
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            'is_running': self.is_running,
            'total_cycles': self.total_cycles,
            'config': self.config,
            'workspace': self.attention_workspace.get_stats() if hasattr(self.attention_workspace, 'get_stats') else {},
            'learning': self.hybrid_learner.get_stats(),
            'gamma_frequency': self.config.gamma_freq,
        }


# ============================================================================
# 意识状态数据结构
# ============================================================================

class NCTConsciousnessState:
    """NCT 意识状态（封装完整状态信息）"""
    
    def __init__(
        self,
        workspace_content: Optional[NCTConsciousContent],
        self_representation: Dict[str, Any],
        neurotransmitter_state: Dict[str, Any],
        consciousness_metrics: Dict[str, Any],
        timestamp: float,
        cycle_number: int,
    ):
        self.workspace_content = workspace_content
        self.self_representation = self_representation
        self.neurotransmitter_state = neurotransmitter_state
        self.consciousness_metrics = consciousness_metrics
        self.timestamp = timestamp
        self.cycle_number = cycle_number
        self.diagnostics: Dict[str, Any] = {}
    
    @property
    def has_content(self) -> bool:
        """是否有意识内容"""
        return self.workspace_content is not None
    
    @property
    def awareness_level(self) -> str:
        """意识水平"""
        return self.consciousness_metrics.get('consciousness_level', 'unknown')
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典格式"""
        return {
            'workspace_content': self.workspace_content.to_dict() if self.workspace_content else None,
            'self_representation': self.self_representation,
            'neurotransmitter_state': self.neurotransmitter_state,
            'consciousness_metrics': self.consciousness_metrics,
            'timestamp': self.timestamp,
            'cycle_number': self.cycle_number,
            'diagnostics': self.diagnostics,
        }


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'NCTManager',
    'NCTConsciousnessState',
]
