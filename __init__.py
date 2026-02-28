"""
NeuroConscious Transformer (NCT) - 神经形态意识 Transformer 架构

版本：v3.1.0
作者：WinClaw Research Team
创建：2026 年 2 月 21 日

核心理论创新：
1. Attention-Based Global Workspace - 用多头注意力替代简单竞争
2. Transformer-STDP Hybrid Learning - 全局调制的突触可塑性
3. Predictive Coding as Decoder-Only Transformer - Friston 自由能 = Transformer 训练目标
4. Multi-Modal Cross-Attention Fusion - 真正的语义级多模态整合
5. γ-Synchronization as Update Cycle - γ同步作为 Transformer 更新周期
6. Φ Calculator from Attention Flow - 避免 IIT 的 NP-hard 问题

完整文档：docs/NCT 完整实施方案.md
快速开始：examples/quickstart.py
"""

__version__ = "3.1.0"
__author__ = "WinClaw Research Team"

# 核心模块导出
from nct_modules import (
    # 核心配置
    NCTConfig,
    NCTConsciousContent,
    
    # 编码与整合
    MultiModalEncoder,
    VisionTransformer,
    AudioSpectrogramTransformer,
    CrossModalIntegration,
    ModalityGating,
    
    # 工作空间与注意
    AttentionGlobalWorkspace,
    AttentionWorkspaceState,
    GammaOscillator,
    NeuralModule,
    
    # 学习与可塑性
    TransformerSTDP,
    ClassicSTDP,
    STDPEvent,
    SynapticUpdate,
    AttentionGradientLearner,
    NeuromodulatorGate,
    
    # 预测编码
    PredictiveCodingDecoder,
    PredictiveHierarchy,
    SelfModelInference,
    
    # 意识度量
    ConsciousnessLevel,
    PhiFromAttention,
    ConsciousnessMetrics,
    
    # γ同步
    GammaSynchronizer,
    PhaseEncodingLayer,
    
    # 总控制器
    NCTManager,
    NCTConsciousnessState,
)

__all__ = [
    # 包信息
    '__version__',
    '__author__',
    
    # 核心配置
    'NCTConfig',
    'NCTConsciousContent',
    
    # 编码与整合
    'MultiModalEncoder',
    'VisionTransformer',
    'AudioSpectrogramTransformer',
    'CrossModalIntegration',
    'ModalityGating',
    
    # 工作空间与注意
    'AttentionGlobalWorkspace',
    'AttentionWorkspaceState',
    'GammaOscillator',
    'NeuralModule',
    
    # 学习与可塑性
    'TransformerSTDP',
    'ClassicSTDP',
    'STDPEvent',
    'SynapticUpdate',
    'AttentionGradientLearner',
    'NeuromodulatorGate',
    
    # 预测编码
    'PredictiveCodingDecoder',
    'PredictiveHierarchy',
    'SelfModelInference',
    
    # 意识度量
    'ConsciousnessLevel',
    'PhiFromAttention',
    'ConsciousnessMetrics',
    
    # γ同步
    'GammaSynchronizer',
    'PhaseEncodingLayer',
    
    # 总控制器
    'NCTManager',
    'NCTConsciousnessState',
]
