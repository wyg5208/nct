"""
多候选竞争方案 - 稳定性验证实验

运行 20 个意识周期，保存所有实验数据用于论文结果对比
"""

import numpy as np
import torch
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nct_modules import NCTManager, NCTConfig


def run_multi_candidate_experiment(n_cycles=20, save_dir='experiments/results'):
    """运行多候选竞争实验
    
    Args:
        n_cycles: 周期数
        save_dir: 结果保存目录
    """
    print("=" * 80)
    print("🧪 多候选竞争方案 - 稳定性验证实验")
    print("=" * 80)
    
    # 创建配置和管理器
    config = NCTConfig(
        d_model=256,
        n_heads=8,
        n_layers=4,
        gamma_freq=40.0,
    )
    
    manager = NCTManager(config)
    manager.start()
    
    results = []
    candidate_names = ['整合表征', '视觉特征', '听觉特征', '内感受特征']
    
    print(f"\n📊 开始运行 {n_cycles} 个意识周期...\n")
    
    for cycle in range(n_cycles):
        # 生成连续性感觉输入（使用正弦波模拟自然刺激）
        t = cycle * 0.2
        sensory_data = {
            'visual': (np.sin(t) * 0.5 + 0.5 + np.random.randn(1, 28, 28) * 0.1).astype(np.float32),
            'auditory': (np.sin(t * 1.5) * 0.4 + 0.5 + np.random.randn(10, 10) * 0.1).astype(np.float32),
            'interoceptive': (np.sin(t * 0.5) * 0.3 + np.random.randn(10) * 0.05).astype(np.float32),
        }
        
        # 处理周期
        state = manager.process_cycle(sensory_data)
        
        # 提取诊断信息
        workspace_info = state.diagnostics.get('workspace', {})
        winner_idx = workspace_info.get('winner_idx', -1)
        winner_salience = workspace_info.get('winner_salience', 0)
        all_salience = workspace_info.get('all_candidates_salience', [])
        
        # 记录结果
        result = {
            'cycle': cycle + 1,
            'timestamp': datetime.now().isoformat(),
            'winner_idx': winner_idx,
            'winner_name': candidate_names[winner_idx] if 0 <= winner_idx < 4 else '未知',
            'winner_salience': float(winner_salience),
            'all_candidates_salience': [float(s) for s in all_salience],
            'phi_value': float(state.consciousness_metrics.get('phi_value', 0)),
            'free_energy': float(state.self_representation['free_energy']),
            'confidence': float(state.self_representation['confidence']),
            'awareness_level': state.awareness_level,
        }
        results.append(result)
        
        # 实时输出
        salience_str = ', '.join([f"{s:.3f}" for s in all_salience])
        print(f"周期 {cycle+1:2d}: 获胜者={result['winner_name']:6s}, "
              f"显著性={winner_salience:.3f}, Φ={result['phi_value']:.3f}, "
              f"自由能={result['free_energy']:.3f}")
        print(f"         候选分布：[{salience_str}]")
    
    manager.stop()
    
    # 统计分析
    print("\n" + "=" * 80)
    print("📊 实验结果统计")
    print("=" * 80)
    
    # 获胜分布
    winner_counts = {}
    for r in results:
        name = r['winner_name']
        winner_counts[name] = winner_counts.get(name, 0) + 1
    
    print("\n🏆 获胜者分布:")
    for name, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
        percentage = count / n_cycles * 100
        print(f"   {name:8s}: {count:2d}次 ({percentage:5.1f}%)")
    
    # 平均指标
    avg_phi = np.mean([r['phi_value'] for r in results])
    avg_free_energy = np.mean([r['free_energy'] for r in results])
    avg_confidence = np.mean([r['confidence'] for r in results])
    avg_winner_salience = np.mean([r['winner_salience'] for r in results])
    
    print(f"\n📈 平均指标:")
    print(f"   Φ值：{avg_phi:.4f}")
    print(f"   自由能：{avg_free_energy:.4f}")
    print(f"   自信度：{avg_confidence:.4f}")
    print(f"   获胜者显著性：{avg_winner_salience:.4f}")
    
    # 保存结果
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存为 JSON
    json_file = save_path / f'multi_candidate_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump({
            'config': {
                'd_model': config.d_model,
                'n_heads': config.n_heads,
                'n_layers': config.n_layers,
                'gamma_freq': config.gamma_freq,
            },
            'n_cycles': n_cycles,
            'results': results,
            'statistics': {
                'winner_distribution': winner_counts,
                'avg_phi': avg_phi,
                'avg_free_energy': avg_free_energy,
                'avg_confidence': avg_confidence,
                'avg_winner_salience': avg_winner_salience,
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到：{json_file}")
    
    # 保存为 CSV
    df = pd.DataFrame(results)
    csv_file = save_path / f'multi_candidate_exp_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"📊 CSV 数据已保存到：{csv_file}")
    
    print("\n" + "=" * 80)
    print("✅ 实验完成！")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_multi_candidate_experiment(n_cycles=20)
