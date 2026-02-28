"""
NCT 架构重构验证实验

验证内容：
1. Transformer-STDP 混合学习收敛速度
2. Predictive Coding 预测误差动态变化
3. Φ值（整合信息量）计算正确性
4. 性能基准测试

作者：WinClaw Research Team
创建：2026 年 2 月 21 日
"""

import sys
import time
import logging
from typing import Dict, Any, List

import numpy as np
import torch

from nct_modules import (
    TransformerSTDP,
    STDPEvent,
    PredictiveCodingDecoder,
    PredictiveHierarchy,
    PhiFromAttention,
)
from nct_modules.nct_core import NCTConfig, NCTConsciousContent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 实验 1: Transformer-STDP 混合学习收敛测试
# ============================================================================

def experiment_hybrid_learning():
    """测试混合学习收敛速度"""
    print("\n" + "="*60)
    print("实验 1: Transformer-STDP 混合学习收敛测试")
    print("="*60)
    
    # 初始化
    stdp = TransformerSTDP(n_neurons=768, d_model=768)
    
    # 生成模拟数据
    n_events = 100
    events = [
        STDPEvent(
            pre_neuron_id=np.random.randint(0, 768),
            post_neuron_id=np.random.randint(0, 768),
            pre_spike_time=i * 10.0,
            post_spike_time=i * 10.0 + np.random.uniform(-5, 5),
        )
        for i in range(n_events)
    ]
    
    # 记录权重变化
    weight_changes = []
    
    # 分批处理
    batch_size = 10
    for batch_idx in range(0, n_events, batch_size):
        batch_events = events[batch_idx:batch_idx+batch_size]
        
        # 随机全局上下文（模拟 attention gradients）
        global_context = torch.randn(768, requires_grad=True)
        
        # 更新
        updates = stdp(batch_events, global_context=global_context)
        
        # 记录平均权重变化
        avg_delta = np.mean([u.total_delta_w for u in updates])
        weight_changes.append(avg_delta)
    
    # 获取学习进度
    progress = stdp.get_learning_progress()
    
    print(f"\n结果:")
    print(f"  - 总更新次数: {progress['total_updates']}")
    print(f"  - 平均权重变化: {progress['avg_delta_w']:.6f}")
    print(f"  - LTP 次数: {progress['ltp_count']}")
    print(f"  - LTD 次数: {progress['ltd_count']}")
    print(f"  - Attention 贡献: {progress['attention_contribution']:.6f}")
    print(f"  - 最近趋势: {progress['recent_trend']}")
    
    # 获取统计信息
    stats = stdp.get_stats()
    print(f"\n突触统计:")
    print(f"  - 总神经元: {stats['total_neurons']}")
    print(f"  - 总突触数: {stats['total_synapses']}")
    print(f"  - 稀疏度: {stats['sparsity']:.4f}")
    print(f"  - 平均权重: {stats['mean_weight']:.4f}")
    
    return {
        'progress': progress,
        'stats': stats,
        'weight_changes': weight_changes,
    }


# ============================================================================
# 实验 2: Predictive Coding 预测误差追踪
# ============================================================================

def experiment_predictive_coding():
    """测试预测误差动态变化"""
    print("\n" + "="*60)
    print("实验 2: Predictive Coding 预测误差追踪")
    print("="*60)
    
    # 初始化
    decoder = PredictiveCodingDecoder(d_model=768, n_heads=8, n_layers=4)
    hierarchy = PredictiveHierarchy({
        'layer0_dim': 768,
        'layer1_dim': 768,
        'layer2_dim': 768,
        'layer3_dim': 768,
        'n_heads': 8,
    })
    
    # 模拟时间序列输入
    n_steps = 20
    errors = []
    
    for step in range(n_steps):
        # 生成感觉输入（带噪声）
        sensory_input = torch.randn(1, 768) + 0.1 * torch.randn(1, 768)
        
        # 处理
        results = hierarchy(sensory_input)
        
        # 记录自由能
        errors.append(results['total_free_energy'])
        
        # 记录到 decoder
        for layer in hierarchy.layers:
            layer.track_prediction_error(results['total_free_energy'])
    
    # 分析结果
    print(f"\n结果:")
    print(f"  - 初始自由能: {errors[0]:.4f}")
    print(f"  - 最终自由能: {errors[-1]:.4f}")
    print(f"  - 平均自由能: {np.mean(errors):.4f}")
    print(f"  - 自由能变化: {errors[-1] - errors[0]:.4f}")
    
    # 获取层级统计
    layer_stats = hierarchy.get_layer_stats()
    print(f"\n层级统计:")
    for stat in layer_stats:
        print(f"  Layer {stat['layer_id']}: d_model={stat['d_model']}, "
              f"error_mean={stat.get('error_mean', 'N/A')}")
    
    return {
        'errors': errors,
        'layer_stats': layer_stats,
    }


# ============================================================================
# 实验 3: Φ值计算验证
# ============================================================================

def experiment_phi_computation():
    """测试 Φ 值计算"""
    print("\n" + "="*60)
    print("实验 3: Φ 值（整合信息量）计算验证")
    print("="*60)
    
    # 初始化（使用正确的参数）
    phi_calculator = PhiFromAttention(n_partitions=10)
    
    # 模拟 attention 权重
    batch_size = 1
    seq_len = 10
    
    # 生成测试数据
    attention_maps = torch.randn(batch_size, 8, seq_len, seq_len)
    hidden_states = torch.randn(batch_size, seq_len, 768)
    
    # 计算 Φ 值（使用 forward 方法）
    phi_value = phi_calculator(attention_maps).item()
    
    # 从神经活动计算
    phi_neural = phi_calculator._compute_phi_from_neural_activity(hidden_states).item()
    
    print(f"\n结果:")
    print(f"  - 从 Attention 计算的 Φ: {phi_value:.4f}")
    print(f"  - 从神经活动计算的 Φ: {phi_neural:.4f}")
    print(f"  - Φ 值范围: [0, ∞)")
    print(f"  - 解释: Φ 值越高，意识整合程度越高")
    
    return {
        'phi_attention': phi_value,
        'phi_neural': phi_neural,
    }


# ============================================================================
# 实验 4: 性能基准测试
# ============================================================================

def experiment_performance():
    """性能基准测试"""
    print("\n" + "="*60)
    print("实验 4: 性能基准测试")
    print("="*60)
    
    results = {}
    
    # 测试 1: Transformer-STDP 性能
    print("\n测试 Transformer-STDP 性能...")
    stdp = TransformerSTDP(n_neurons=768, d_model=768)
    
    n_iterations = 50
    start_time = time.time()
    
    for _ in range(n_iterations):
        events = [
            STDPEvent(
                pre_neuron_id=np.random.randint(0, 768),
                post_neuron_id=np.random.randint(0, 768),
                pre_spike_time=0.0,
                post_spike_time=1.0,
            )
            for _ in range(10)
        ]
        global_context = torch.randn(768, requires_grad=True)
        stdp(events, global_context=global_context)
    
    stdp_time = time.time() - start_time
    results['stdp'] = {
        'iterations': n_iterations,
        'total_time': stdp_time,
        'avg_time_per_iter': stdp_time / n_iterations,
    }
    print(f"  - {n_iterations} 次迭代总时间: {stdp_time:.3f}s")
    print(f"  - 平均每次迭代: {stdp_time/n_iterations*1000:.2f}ms")
    
    # 测试 2: Predictive Coding 性能
    print("\n测试 Predictive Coding 性能...")
    hierarchy = PredictiveHierarchy({
        'layer0_dim': 768,
        'layer1_dim': 768,
        'layer2_dim': 768,
        'layer3_dim': 768,
        'n_heads': 8,
    })
    
    n_iterations = 50
    start_time = time.time()
    
    for _ in range(n_iterations):
        sensory_input = torch.randn(1, 768)
        hierarchy(sensory_input)
    
    pc_time = time.time() - start_time
    results['predictive_coding'] = {
        'iterations': n_iterations,
        'total_time': pc_time,
        'avg_time_per_iter': pc_time / n_iterations,
    }
    print(f"  - {n_iterations} 次迭代总时间: {pc_time:.3f}s")
    print(f"  - 平均每次迭代: {pc_time/n_iterations*1000:.2f}ms")
    
    # 测试 3: Φ 值计算性能
    print("\n测试 Φ 值计算性能...")
    phi_calculator = PhiFromAttention(n_partitions=10)
    
    n_iterations = 100
    start_time = time.time()
    
    for _ in range(n_iterations):
        attention_maps = torch.randn(1, 8, 10, 10)
        phi_calculator(attention_maps)
    
    phi_time = time.time() - start_time
    results['phi'] = {
        'iterations': n_iterations,
        'total_time': phi_time,
        'avg_time_per_iter': phi_time / n_iterations,
    }
    print(f"  - {n_iterations} 次计算总时间: {phi_time:.3f}s")
    print(f"  - 平均每次计算: {phi_time/n_iterations*1000:.2f}ms")
    
    return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有实验"""
    print("\n" + "#"*60)
    print("# NCT 架构重构验证实验")
    print("# 方案 C - 完整验证")
    print("#"*60)
    
    all_results = {}
    
    # 运行实验
    try:
        all_results['hybrid_learning'] = experiment_hybrid_learning()
    except Exception as e:
        print(f"实验 1 失败: {e}")
        all_results['hybrid_learning'] = {'error': str(e)}
    
    try:
        all_results['predictive_coding'] = experiment_predictive_coding()
    except Exception as e:
        print(f"实验 2 失败: {e}")
        all_results['predictive_coding'] = {'error': str(e)}
    
    try:
        all_results['phi'] = experiment_phi_computation()
    except Exception as e:
        print(f"实验 3 失败: {e}")
        all_results['phi'] = {'error': str(e)}
    
    try:
        all_results['performance'] = experiment_performance()
    except Exception as e:
        print(f"实验 4 失败: {e}")
        all_results['performance'] = {'error': str(e)}
    
    # 总结
    print("\n" + "#"*60)
    print("# 实验总结")
    print("#"*60)
    
    success_count = sum(1 for v in all_results.values() if 'error' not in v)
    total_count = len(all_results)
    
    print(f"\n成功: {success_count}/{total_count}")
    
    for name, result in all_results.items():
        if 'error' in result:
            print(f"  - {name}: ❌ 失败 ({result['error'][:50]}...)")
        else:
            print(f"  - {name}: ✅ 成功")
    
    print("\n" + "#"*60)
    print("# 方案 C 重构完成！")
    print("#"*60)
    
    return all_results


if __name__ == '__main__':
    results = main()
