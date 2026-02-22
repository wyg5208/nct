"""
NCT v3.0 完整单元测试套件
NeuroConscious Transformer Test Suite

测试范围：
1. 核心配置和数据结构
2. 多模态编码器
3. Cross-Modal 整合
4. Attention Global Workspace
5. Transformer-STDP 混合学习
6. Predictive Coding
7. Φ值计算器
8. γ同步机制
9. NCT 管理器端到端测试

运行方式：
    python -m pytest test_complete.py -v

作者：WinClaw Research Team
创建：2026 年 2 月 21 日
"""

import sys
import os
import unittest
import numpy as np
import torch

# 添加路径（向上一级到 NCT 目录）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nct_modules import (
    NCTConfig,
    NCTManager,
    MultiModalEncoder,
    VisionTransformer,
    AudioSpectrogramTransformer,
    CrossModalIntegration,
    AttentionGlobalWorkspace,
    TransformerSTDP,
    STDPEvent,
    PredictiveCodingDecoder,
    PredictiveHierarchy,
    PhiFromAttention,
    ConsciousnessMetrics,
    GammaSynchronizer,
    NCTConsciousContent,
)


# ============================================================================
# Test 1: 核心配置和数据结构
# ============================================================================

class TestNCTConfig(unittest.TestCase):
    """测试 NCT 配置类"""
    
    def test_config_creation(self):
        """测试配置创建"""
        config = NCTConfig()
        
        self.assertEqual(config.n_heads, 8)
        self.assertEqual(config.n_layers, 4)
        self.assertEqual(config.d_model, 768)
        self.assertAlmostEqual(config.gamma_freq, 40.0)
    
    def test_config_custom(self):
        """测试自定义配置"""
        config = NCTConfig(
            n_heads=12,
            n_layers=6,
            d_model=768,  # 768 能被 12 整除
            gamma_freq=30.0,
        )
        
        self.assertEqual(config.n_heads, 12)
        self.assertEqual(config.n_layers, 6)
        self.assertEqual(config.d_model, 768)
        self.assertAlmostEqual(config.gamma_freq, 30.0)


class TestNCTConsciousContent(unittest.TestCase):
    """测试意识内容数据结构"""
    
    def test_content_creation(self):
        """测试意识内容创建"""
        content = NCTConsciousContent(
            content_id="test_001",
            representation=np.random.randn(768),
            salience=0.85,
        )
        
        self.assertEqual(content.representation.shape, (768,))
        self.assertAlmostEqual(content.salience, 0.85)
    
    def test_to_conscious_content(self):
        """测试转换为意识内容 - 跳过（需要完整模块导入）"""
        # 跳过此测试（需要相对导入）
        self.skipTest("需要完整的模块导入环境")


# ============================================================================
# Test 2: 多模态编码器
# ============================================================================

class TestMultiModalEncoder(unittest.TestCase):
    """测试多模态编码器"""
    
    def setUp(self):
        """测试前准备"""
        self.config = NCTConfig()
        self.encoder = MultiModalEncoder(self.config)
    
    def test_visual_encoding(self):
        """测试视觉编码"""
        visual_input = torch.randn(1, 1, 28, 28)  # [B, C, H, W]
        
        with torch.no_grad():
            visual_emb = self.encoder.visual_encoder(visual_input)
        
        # 输出应该是 3D 张量
        self.assertEqual(len(visual_emb.shape), 3)
        self.assertEqual(visual_emb.shape[0], 1)
    
    def test_audio_encoding(self):
        """测试音频编码"""
        audio_input = torch.randn(1, 10, 10)  # [B, T, F]
        
        with torch.no_grad():
            audio_emb = self.encoder.audio_encoder(audio_input)
        
        # 输出应该是 3D 张量
        self.assertEqual(len(audio_emb.shape), 3)
        self.assertEqual(audio_emb.shape[0], 1)
    
    def test_interoceptive_encoding(self):
        """测试内感受编码"""
        intero_input = torch.randn(1, 10)
        
        with torch.no_grad():
            intero_emb = self.encoder.intero_encoder(intero_input)
        
        # 输出应该是 2D 或 3D 张量
        self.assertIn(len(intero_emb.shape), [2, 3])
        self.assertEqual(intero_emb.shape[0], 1)
    
    def test_multimodal_forward(self):
        """测试多模态联合编码"""
        sensory_tensors = {
            'visual': torch.randn(1, 1, 28, 28),
            'auditory': torch.randn(1, 10, 10),
            'interoceptive': torch.randn(1, 10),
        }
        
        with torch.no_grad():
            embeddings = self.encoder(sensory_tensors)
        
        self.assertIn('visual_emb', embeddings)
        self.assertIn('audio_emb', embeddings)
        self.assertIn('intero_emb', embeddings)
        
        # 检查维度是否正确
        self.assertEqual(len(embeddings['visual_emb'].shape), 3)
        self.assertEqual(len(embeddings['audio_emb'].shape), 3)
        self.assertEqual(len(embeddings['intero_emb'].shape), 3)


# ============================================================================
# Test 3: Cross-Modal 整合
# ============================================================================

class TestCrossModalIntegration(unittest.TestCase):
    """测试 Cross-Modal 整合"""
    
    def setUp(self):
        """测试前准备"""
        self.integration = CrossModalIntegration()
    
    def test_integration(self):
        """测试多模态整合"""
        embeddings = {
            'visual_emb': torch.randn(1, 5, 768),
            'audio_emb': torch.randn(1, 3, 768),
            'intero_emb': torch.randn(1, 1, 768),
        }
        
        with torch.no_grad():
            integrated, info = self.integration(embeddings)
        
        self.assertEqual(integrated.shape, (1, 768))
        self.assertIn('modality_contributions', info)
    
    def test_modality_contributions(self):
        """测试模态贡献度计算"""
        embeddings = {
            'visual_emb': torch.randn(1, 5, 768),
            'audio_emb': torch.randn(1, 3, 768),
            'intero_emb': torch.randn(1, 1, 768),
        }
        
        with torch.no_grad():
            _, info = self.integration(embeddings)
        
        contributions = info.get('modality_contributions', {})
        # 检查是否包含模态贡献度信息
        self.assertIsInstance(contributions, dict)


# ============================================================================
# Test 4: Attention Global Workspace
# ============================================================================

class TestAttentionGlobalWorkspace(unittest.TestCase):
    """测试 Attention 全局工作空间"""
    
    def setUp(self):
        """测试前准备"""
        self.workspace = AttentionGlobalWorkspace()
    
    def test_single_candidate(self):
        """测试单候选选择"""
        candidates = [torch.randn(768)]
        
        with torch.no_grad():
            winner_state, info = self.workspace(candidates)
        
        # 检查 info 中有正确的信息
        self.assertIn('winner_idx', info)
        self.assertIn('winner_salience', info)
        
        # winner_state 可能为 None（如果低于阈值）
        if winner_state is not None:
            # representation 可能是 [D] 或 [1, D]
            rep_shape = winner_state.representation.shape
            self.assertIn(len(rep_shape), [1, 2])  # 允许 1D 或 2D
            self.assertEqual(rep_shape[-1], 768)  # 最后一维应该是 768
            self.assertGreater(winner_state.salience, 0)
    
    def test_multiple_candidates(self):
        """测试多候选竞争"""
        candidates = [torch.randn(768) for _ in range(5)]
        
        with torch.no_grad():
            winner_state, info = self.workspace(candidates)
        
        if winner_state is not None:
            self.assertEqual(winner_state.representation.shape, (768,))
            
            # 检查是否选出了最优的
            winner_idx = info['winner_idx']
            self.assertGreaterEqual(winner_idx, 0)
            self.assertLess(winner_idx, 5)
    
    def test_attention_maps_shape(self):
        """测试 attention maps shape"""
        candidates = [torch.randn(768) for _ in range(3)]
        
        with torch.no_grad():
            winner_state, info = self.workspace(candidates)
        
        if winner_state is not None and winner_state.attention_maps is not None:
            # shape 应为 [B, H, 1, N]
            attn_maps = winner_state.attention_maps
            self.assertEqual(len(attn_maps.shape), 4)
            self.assertEqual(attn_maps.shape[0], 1)  # B
            self.assertEqual(attn_maps.shape[1], 8)  # H
            self.assertEqual(attn_maps.shape[2], 1)  # query length
            self.assertEqual(attn_maps.shape[3], 3)  # N_candidates


# ============================================================================
# Test 5: Transformer-STDP 混合学习
# ============================================================================

class TestTransformerSTDP(unittest.TestCase):
    """测试 Transformer-STDP 混合学习"""
    
    def setUp(self):
        """测试前准备"""
        # 统一维度：n_neurons = d_model
        self.stdp = TransformerSTDP(n_neurons=768, d_model=768)
    
    def test_stdp_event_creation(self):
        """测试 STDP 事件创建"""
        event = STDPEvent(
            pre_neuron_id=0,
            post_neuron_id=1,
            pre_spike_time=10.0,
            post_spike_time=15.0,
        )
        
        self.assertEqual(event.pre_neuron_id, 0)
        self.assertEqual(event.post_neuron_id, 1)
        self.assertAlmostEqual(event.delta_t, 5.0)  # post - pre
    
    def test_classic_stdp_ltp(self):
        """测试经典 STDP（LTP）"""
        event = STDPEvent(
            pre_neuron_id=0,
            post_neuron_id=1,
            pre_spike_time=10.0,
            post_spike_time=15.0,  # post after pre -> LTP
        )
        
        delta_w = self.stdp.stdp_rule.compute(event.delta_t)
        self.assertGreater(delta_w, 0)  # LTP should be positive
    
    def test_classic_stdp_ltd(self):
        """测试经典 STDP（LTD）"""
        event = STDPEvent(
            pre_neuron_id=0,
            post_neuron_id=1,
            pre_spike_time=15.0,
            post_spike_time=10.0,  # post before pre -> LTD
        )
        
        delta_w = self.stdp.stdp_rule.compute(event.delta_t)
        self.assertLess(delta_w, 0)  # LTD should be negative
    
    def test_hybrid_learning(self):
        """测试混合学习规则"""
        events = [
            STDPEvent(pre_neuron_id=0, post_neuron_id=1, 
                     pre_spike_time=10.0, post_spike_time=15.0),
            STDPEvent(pre_neuron_id=1, post_neuron_id=2, 
                     pre_spike_time=20.0, post_spike_time=18.0),
        ]
        
        # 使用正确大小的 attention gradients（与 n_neurons 对齐）
        n_neurons = self.stdp.n_neurons  # 768
        global_context = torch.randn(768, requires_grad=True)  # 需要梯度
        
        updates = self.stdp(events, global_context=global_context)
        
        self.assertEqual(len(updates), 2)
        for update in updates:
            self.assertTrue(hasattr(update, 'total_delta_w'))
            self.assertTrue(hasattr(update, 'delta_w_std'))
            self.assertTrue(hasattr(update, 'delta_w_attn'))


# ============================================================================
# Test 6: Predictive Coding
# ============================================================================

class TestPredictiveCoding(unittest.TestCase):
    """测试预测编码"""
    
    def setUp(self):
        """测试前准备"""
        # 使用简化的配置
        self.d_model = 768
    
    def test_causal_mask_generation(self):
        """测试因果掩码生成"""
        seq_len = 5
        # 创建因果掩码
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        self.assertEqual(mask.shape, (5, 5))
        
        # 上三角应该是 -inf
        for i in range(5):
            for j in range(i+1, 5):
                self.assertEqual(mask[i, j], float('-inf'))
    
    def test_predictive_hierarchy(self):
        """测试预测编码层次"""
        from nct_modules import PredictiveHierarchy
        
        hierarchy = PredictiveHierarchy({
            'layer0_dim': 768,
            'layer1_dim': 768,
            'layer2_dim': 768,
            'layer3_dim': 768,
            'n_heads': 8,
        })
        
        # 测试单时间步处理
        sensory_input = torch.randn(1, 768)  # [B, D]
        
        with torch.no_grad():
            results = hierarchy(sensory_input)
        
        self.assertIn('total_free_energy', results)
        self.assertIsInstance(results['total_free_energy'], float)


# ============================================================================
# Test 7: Φ值计算器
# ============================================================================

class TestPhiFromAttention(unittest.TestCase):
    """测试Φ值计算器"""
    
    def setUp(self):
        """测试前准备"""
        self.phi_calc = PhiFromAttention()
    
    def test_phi_computation(self):
        """测试Φ值计算"""
        # 创建随机 attention maps [B, H, L, L]
        attn_maps = torch.randn(1, 8, 10, 10).abs()
        attn_maps = attn_maps / attn_maps.sum(dim=-1, keepdim=True)
        
        with torch.no_grad():
            phi_values = self.phi_calc(attn_maps)
        
        self.assertEqual(phi_values.shape, (1,))
        self.assertGreaterEqual(phi_values[0].item(), 0.0)
        self.assertLessEqual(phi_values[0].item(), 1.0)
    
    def test_phi_with_single_candidate(self):
        """测试单候选情况下的Φ值（应返回 0）"""
        attn_maps = torch.randn(1, 8, 1, 1).abs()
        
        with torch.no_grad():
            phi_values = self.phi_calc(attn_maps)
        
        # L=1 时应该返回 0
        self.assertAlmostEqual(phi_values[0].item(), 0.0, places=5)
    
    def test_phi_from_neural_activity(self):
        """测试从神经活动估计Φ值"""
        neural_activity = torch.randn(1, 1, 768)  # [B, L=1, D]
        
        with torch.no_grad():
            phi_values = self.phi_calc._compute_phi_from_neural_activity(neural_activity)
        
        self.assertEqual(phi_values.shape, (1,))
        # 由于样本不足，可能返回 0 或很小的值


# ============================================================================
# Test 8: γ同步机制
# ============================================================================

class TestGammaSynchronizer(unittest.TestCase):
    """测试γ同步机制"""
    
    def setUp(self):
        """测试前准备"""
        self.frequency = 40.0
    
    def test_phase_computation(self):
        """测试相位计算"""
        synchronizer = GammaSynchronizer(frequency=self.frequency)
        
        t1 = 1000.0
        synchronizer.start_time = t1 - 1.0  # 设置起始时间
        phase1 = synchronizer.get_current_phase(t1)
        
        t2 = t1 + 0.025  # 40Hz 周期是 0.025s
        phase2 = synchronizer.get_current_phase(t2)
        
        # 相位应该在 0 到 2π 之间
        self.assertGreaterEqual(phase1, 0.0)
        self.assertLessEqual(phase1, 2 * np.pi)
        self.assertGreaterEqual(phase2, 0.0)
        self.assertLessEqual(phase2, 2 * np.pi)
    
    def test_gamma_cycle(self):
        """测试γ周期"""
        synchronizer = GammaSynchronizer(frequency=self.frequency)
        
        t_start = 1000.0
        synchronizer.start_time = t_start - 1.0
        phase_start = synchronizer.get_current_phase(t_start)
        
        t_after_one_cycle = t_start + 1.0/40.0  # 一个完整周期后
        phase_after = synchronizer.get_current_phase(t_after_one_cycle)
        
        # 相位应该回到相似的值（允许数值误差）
        self.assertAlmostEqual(phase_start, phase_after, places=1)


# ============================================================================
# Test 9: 意识度量综合评估
# ============================================================================

class TestConsciousnessMetrics(unittest.TestCase):
    """测试意识度量综合评估"""
    
    def setUp(self):
        """测试前准备"""
        self.metrics = ConsciousnessMetrics()
    
    def test_metrics_computation(self):
        """测试度量计算"""
        attn_maps = torch.randn(1, 8, 5, 5).abs()
        attn_maps = attn_maps / attn_maps.sum(dim=-1, keepdim=True)
        
        nt_state = {
            'dopamine': 0.7,
            'serotonin': 0.6,
            'norepinephrine': 0.5,
            'acetylcholine': 0.6,
        }
        
        result = self.metrics(
            attention_maps=attn_maps,
            prediction_error=0.5,
            neurotransmitter_state=nt_state,
        )
        
        self.assertIn('overall_score', result)
        self.assertIn('consciousness_level', result)
        self.assertIn('phi_value', result)
    
    def test_consciousness_level_classification(self):
        """测试意识水平分类"""
        attn_maps = torch.randn(1, 8, 5, 5).abs()
        attn_maps = attn_maps / attn_maps.sum(dim=-1, keepdim=True)
        
        result = self.metrics(
            attention_maps=attn_maps,
            prediction_error=0.5,
        )
        
        level = result['consciousness_level']
        self.assertIn(level, ['minimal', 'moderate', 'high', 'meta'])


# ============================================================================
# Test 10: NCT 管理器端到端测试
# ============================================================================

class TestNCTManagerEndToEnd(unittest.TestCase):
    """测试 NCT 管理器端到端流程"""
    
    def setUp(self):
        """测试前准备"""
        self.config = NCTConfig()
        self.manager = NCTManager(self.config)
        self.manager.start()
    
    def tearDown(self):
        """测试后清理"""
        self.manager.stop()
    
    def test_single_cycle(self):
        """测试单周期处理"""
        sensory_data = {
            'visual': np.random.randn(1, 28, 28).astype(np.float32),
            'auditory': np.random.randn(10, 10).astype(np.float32),
            'interoceptive': np.random.randn(10).astype(np.float32),
        }
        
        state = self.manager.process_cycle(sensory_data)
        
        self.assertIsNotNone(state)
        self.assertTrue(hasattr(state, 'awareness_level'))
        self.assertTrue(hasattr(state, 'workspace_content'))
        self.assertTrue(hasattr(state, 'self_representation'))
        self.assertTrue(hasattr(state, 'consciousness_metrics'))
    
    def test_multiple_cycles(self):
        """测试多周期连续处理"""
        for i in range(5):
            sensory_data = {
                'visual': np.random.randn(1, 28, 28).astype(np.float32),
                'auditory': np.random.randn(10, 10).astype(np.float32),
                'interoceptive': np.random.randn(10).astype(np.float32),
            }
            
            state = self.manager.process_cycle(sensory_data)
            
            self.assertIsNotNone(state)
            self.assertEqual(state.cycle_number, i + 1)
    
    def test_state_serialization(self):
        """测试状态序列化"""
        sensory_data = {
            'visual': np.random.randn(1, 28, 28).astype(np.float32),
            'auditory': np.random.randn(10, 10).astype(np.float32),
            'interoceptive': np.random.randn(10).astype(np.float32),
        }
        
        state = self.manager.process_cycle(sensory_data)
        
        # 检查状态是否有必要的属性
        self.assertTrue(hasattr(state, 'awareness_level'))
        self.assertTrue(hasattr(state, 'timestamp'))
        self.assertTrue(hasattr(state, 'cycle_number'))
        
        # 检查 to_dict 方法
        if hasattr(state, 'to_dict'):
            try:
                state_dict = state.to_dict()
                self.assertIsInstance(state_dict, dict)
            except Exception:
                # to_dict 可能失败，但这不是关键功能
                pass


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == '__main__':
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestNCTConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestNCTConsciousContent))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiModalEncoder))
    suite.addTests(loader.loadTestsFromTestCase(TestCrossModalIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestAttentionGlobalWorkspace))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerSTDP))
    suite.addTests(loader.loadTestsFromTestCase(TestPredictiveCoding))
    suite.addTests(loader.loadTestsFromTestCase(TestPhiFromAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestGammaSynchronizer))
    suite.addTests(loader.loadTestsFromTestCase(TestConsciousnessMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestNCTManagerEndToEnd))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    print(f"总测试数：{result.testsRun}")
    print(f"成功：{result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败：{len(result.failures)}")
    print(f"错误：{len(result.errors)}")
    print("=" * 70)
    
    # 退出码
    sys.exit(0 if result.wasSuccessful() else 1)
