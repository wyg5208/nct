"""
NCT 核心模块包
NeuroConscious Transformer Core Modules

导出所有 NCT v3.1 的核心组件
"""

from .nct_modules import (
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

__version__ = "3.1.0"
__all__ = [
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
