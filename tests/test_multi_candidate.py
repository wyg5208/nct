"""
测试多候选竞争方案

验证：
1. 4 个候选（整合、视觉、听觉、内感受）正确生成
2. 注意力机制正常选择获胜者
3. 可视化展示竞争结果
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from nct_modules import NCTManager, NCTConfig


def test_multi_candidate_competition():
    """测试多候选竞争"""
    print("=" * 60)
    print("🧪 测试多候选竞争方案")
    print("=" * 60)
    
    # 创建配置和管理器
    config = NCTConfig(
        d_model=256,  # 使用较小的维度加快测试
        n_heads=8,
        n_layers=4,
        gamma_freq=40.0,
    )
    
    manager = NCTManager(config)
    manager.start()
    
    # 模拟感觉输入
    sensory_data = {
        'visual': np.random.randn(1, 28, 28).astype(np.float32),
        'auditory': np.random.randn(10, 10).astype(np.float32),
        'interoceptive': np.random.randn(10).astype(np.float32),
    }
    
    print("\n📊 运行第一个周期...")
    state1 = manager.process_cycle(sensory_data)
    
    # 检查诊断信息
    if hasattr(state1, 'diagnostics') and 'workspace' in state1.diagnostics:
        workspace_info = state1.diagnostics['workspace']
        
        print("\n✅ 工作空间信息:")
        print(f"   获胜者索引：{workspace_info.get('winner_idx', 'N/A')}")
        print(f"   获胜者显著性：{workspace_info.get('winner_salience', 0):.4f}")
        print(f"   所有候选显著性：{workspace_info.get('all_candidates_salience', [])}")
        
        # 验证候选数量
        all_salience = workspace_info.get('all_candidates_salience', [])
        if len(all_salience) == 4:
            print("\n✅ 候选数量正确：4 个候选")
            candidate_names = ['整合表征', '视觉特征', '听觉特征', '内感受特征']
            for i, (name, salience) in enumerate(zip(candidate_names, all_salience)):
                print(f"   - {name}: {salience:.4f}")
        else:
            print(f"\n⚠️ 候选数量异常：期望 4 个，实际 {len(all_salience)} 个")
    
    # 检查意识状态
    print("\n📈 意识状态:")
    print(f"   意识水平：{state1.awareness_level}")
    print(f"   Φ值：{state1.consciousness_metrics.get('phi_value', 0):.4f}")
    print(f"   自由能：{state1.self_representation['free_energy']:.4f}")
    print(f"   自信度：{state1.self_representation['confidence']:.4f}")
    
    # 运行多个周期观察稳定性
    print("\n🔄 运行连续 5 个周期...")
    results = []
    for cycle in range(5):
        # 添加一些变化到输入
        sensory_data['visual'] = np.random.randn(1, 28, 28).astype(np.float32) * 0.5 + 0.5
        sensory_data['auditory'] = np.random.randn(10, 10).astype(np.float32) * 0.3 + 0.5
        sensory_data['interoceptive'] = np.random.randn(10).astype(np.float32) * 0.2
        
        state = manager.process_cycle(sensory_data)
        
        if hasattr(state, 'diagnostics') and 'workspace' in state.diagnostics:
            workspace_info = state.diagnostics['workspace']
            winner_idx = workspace_info.get('winner_idx', -1)
            winner_salience = workspace_info.get('winner_salience', 0)
            all_salience = workspace_info.get('all_candidates_salience', [])
            
            results.append({
                'cycle': cycle + 1,
                'winner_idx': winner_idx,
                'winner_salience': winner_salience,
                'all_salience': all_salience,
                'phi': state.consciousness_metrics.get('phi_value', 0),
                'free_energy': state.self_representation['free_energy'],
            })
            
            candidate_names = ['整合', '视觉', '听觉', '内感受']
            winner_name = candidate_names[winner_idx] if 0 <= winner_idx < 4 else '未知'
            print(f"   周期 {cycle+1}: 获胜者={winner_name}, 显著性={winner_salience:.3f}, Φ={results[-1]['phi']:.3f}")
    
    manager.stop()
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    
    # 统计获胜分布
    if results:
        winner_counts = {}
        for r in results:
            idx = r['winner_idx']
            winner_counts[idx] = winner_counts.get(idx, 0) + 1
        
        print("\n📊 获胜分布统计:")
        candidate_names = ['整合表征', '视觉特征', '听觉特征', '内感受特征']
        for idx, count in sorted(winner_counts.items()):
            name = candidate_names[idx] if 0 <= idx < 4 else '未知'
            percentage = count / len(results) * 100
            print(f"   {name}: {count}次 ({percentage:.1f}%)")
    
    return results


if __name__ == "__main__":
    test_multi_candidate_competition()
