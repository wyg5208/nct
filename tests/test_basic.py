"""
NCT v3.0 基础功能测试套件

运行方式：
    python test_basic.py

测试内容：
1. NCTManager 初始化和生命周期
2. 单周期处理流程
3. 多模态编码功能
4. Attention Global Workspace 选择
5. Transformer-STDP 学习
6. Φ值计算
7. 意识度量评估

预期结果：所有测试通过 ✅
"""

import sys
import os
import unittest
import numpy as np
import torch

# 添加路径（向上一级）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nct_modules import (
    NCTManager, 
    NCTConfig,
    MultiModalEncoder,
    CrossModalIntegration,
    AttentionGlobalWorkspace,
    TransformerSTDP,
    PhiFromAttention,
    ConsciousnessMetrics,
)


class TestNCTBasic(unittest.TestCase):
    """NCT 基础功能测试"""
    
    def setUp(self):
        """测试前准备"""
        self.config = NCTConfig()
        self.manager = NCTManager(self.config)
        self.sensory_data = {
            'visual': np.random.randn(20).astype(np.float32),
            'auditory': np.random.randn(10).astype(np.float32),
            'interoceptive': np.random.randn(10).astype(np.float32),
        }
    
    def test_manager_initialization(self):
        """测试 1: 管理器初始化"""
        print("\n✓ 测试 1: NCTManager 初始化")
        
        self.assertIsNotNone(self.manager)
        self.assertTrue(hasattr(self.manager, 'multimodal_encoder'))
        self.assertTrue(hasattr(self.manager, 'attention_workspace'))
        self.assertTrue(hasattr(self.manager, 'hybrid_learner'))
        self.assertTrue(hasattr(self.manager, 'consciousness_metrics'))
        
        print(f"  - 配置：{self.config}")
        print(f"  - 状态：就绪")
    
    def test_start_stop_lifecycle(self):
        """测试 2: 启动/停止生命周期"""
        print("\n✓ 测试 2: 启动/停止生命周期")
        
        # 启动
        self.manager.start()
        self.assertTrue(self.manager.is_running)
        print(f"  - ✓ 启动成功")
        
        # 停止
        self.manager.stop()
        self.assertFalse(self.manager.is_running)
        print(f"  - ✓ 停止成功")
    
    def test_single_cycle_processing(self):
        """测试 3: 单周期处理"""
        print("\n✓ 测试 3: 单周期处理流程")
        
        self.manager.start()
        
        # 处理一个周期
        state = self.manager.process_cycle(self.sensory_data)
        
        # 验证返回状态
        self.assertIsNotNone(state)
        self.assertTrue(hasattr(state, 'workspace_content'))
        self.assertTrue(hasattr(state, 'self_representation'))
        self.assertTrue(hasattr(state, 'consciousness_metrics'))
        
        # 验证指标
        metrics = state.consciousness_metrics
        self.assertIn('overall_score', metrics)
        self.assertIn('consciousness_level', metrics)
        self.assertIn('phi_value', metrics)
        
        print(f"  - 意识水平：{state.awareness_level}")
        print(f"  - Φ值：{metrics.get('phi_value', 0):.3f}")
        print(f"  - 自信度：{state.self_representation['confidence']:.3f}")
        
        self.manager.stop()
    
    def test_multimodal_encoding(self):
        """测试 4: 多模态编码功能"""
        print("\n✓ 测试 4: 多模态编码功能")
        
        encoder = MultiModalEncoder(self.config)
        
        # 准备输入
        sensory_tensors = {
            'visual': torch.randn(1, 20),
            'auditory': torch.randn(1, 10),
            'interoceptive': torch.randn(1, 10),
        }
        
        # 编码
        embeddings = encoder(sensory_tensors)
        
        # 验证输出
        self.assertIn('visual_emb', embeddings)
        self.assertIn('audio_emb', embeddings)
        self.assertIn('intero_emb', embeddings)
        
        print(f"  - Visual embedding: {embeddings['visual_emb'].shape}")
        print(f"  - Audio embedding: {embeddings['audio_emb'].shape}")
        print(f"  - Intero embedding: {embeddings['intero_emb'].shape}")
    
    def test_cross_modal_integration(self):
        """测试 5: Cross-Modal 整合"""
        print("\n✓ 测试 5: Cross-Modal 整合功能")
        
        integration = CrossModalIntegration()
        
        # 模拟 embeddings
        embeddings = {
            'visual_emb': torch.randn(1, 5, 768),
            'audio_emb': torch.randn(1, 3, 768),
            'intero_emb': torch.randn(1, 1, 768),
        }
        
        # 整合
        integrated, info = integration(embeddings)
        
        # 验证
        self.assertIsNotNone(integrated)
        self.assertEqual(integrated.shape, (1, 768))
        self.assertIn('modality_contributions', info)
        
        print(f"  - 整合后维度：{integrated.shape}")
        print(f"  - 模态贡献：{info['modality_contributions']}")
    
    def test_attention_workspace_selection(self):
        """测试 6: Attention 工作空间选择"""
        print("\n✓ 测试 6: Attention Global Workspace 选择")
        
        workspace = AttentionGlobalWorkspace()
        
        # 创建候选
        candidates = [torch.randn(768) for _ in range(5)]
        
        # 选择获胜者
        winner_state, info = workspace(candidates)
        
        # 验证
        if winner_state is not None:
            print(f"  - ✓ 选出获胜者：idx={info['winner_idx']}, salience={winner_state.salience:.3f}")
            self.assertTrue(winner_state.salience > 0)
        else:
            print(f"  - ✗ 无获胜者（低于阈值）")
    
    def test_stdp_learning(self):
        """测试 7: Transformer-STDP 学习"""
        print("\n✓ 测试 7: Transformer-STDP 混合学习")
        
        stdp = TransformerSTDP(n_neurons=100, d_model=768)
        
        # 创建 STDP 事件
        from nct_hybrid_learning import STDPEvent
        events = [
            STDPEvent(pre_neuron_id=0, post_neuron_id=1, 
                     pre_spike_time=10.0, post_spike_time=15.0),
            STDPEvent(pre_neuron_id=1, post_neuron_id=2, 
                     pre_spike_time=20.0, post_spike_time=18.0),
        ]
        
        # 更新突触
        updates = stdp(events)
        
        # 验证
        self.assertEqual(len(updates), 2)
        print(f"  - 更新了 {len(updates)} 个突触")
        for i, update in enumerate(updates):
            print(f"    突触 {i+1}: Δw={update.total_delta_w:.4f}")
    
    def test_phi_calculation(self):
        """测试 8: Φ值计算"""
        print("\n✓ 测试 8: Φ值计算器（基于 Attention Flow）")
        
        phi_calc = PhiFromAttention()
        
        # 模拟 attention maps
        attn_maps = torch.randn(1, 8, 10, 10).abs()  # [B, H, L, L]
        attn_maps = attn_maps / attn_maps.sum(dim=-1, keepdim=True)  # 归一化
        
        # 计算Φ
        phi_values = phi_calc(attn_maps)
        
        # 验证
        self.assertEqual(len(phi_values), 1)
        phi = phi_values[0].item()
        self.assertGreaterEqual(phi, 0.0)
        self.assertLessEqual(phi, 1.0)
        
        print(f"  - Φ值：{phi:.3f}")
    
    def test_consciousness_metrics(self):
        """测试 9: 意识度量综合评估"""
        print("\n✓ 测试 9: 意识度量综合评估")
        
        metrics = ConsciousnessMetrics()
        
        # 模拟输入
        attn_maps = torch.randn(1, 8, 10, 10).abs()
        attn_maps = attn_maps / attn_maps.sum(dim=-1, keepdim=True)
        
        nt_state = {
            'dopamine': 0.7,
            'serotonin': 0.6,
            'norepinephrine': 0.5,
            'acetylcholine': 0.6,
        }
        
        # 评估
        result = metrics(
            attention_maps=attn_maps,
            prediction_error=0.5,
            neurotransmitter_state=nt_state,
        )
        
        # 验证
        self.assertIn('overall_score', result)
        self.assertIn('consciousness_level', result)
        
        print(f"  - 综合评分：{result['overall_score']:.3f}")
        print(f"  - 意识水平：{result['consciousness_level']}")
        print(f"  - Φ值：{result['phi_value']:.3f}")


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("NCT v3.0 基础功能测试套件")
    print("=" * 70)
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestNCTBasic)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 总结
    print("\n" + "=" * 70)
    print(f"测试结果：{result.testsRun} 个测试")
    print(f"成功：{result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败：{len(result.failures)}")
    print(f"错误：{len(result.errors)}")
    print("=" * 70)
    
    return len(result.failures) == 0 and len(result.errors) == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
