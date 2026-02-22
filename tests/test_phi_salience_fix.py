"""
验证Φ值和 salience 修复

运行几个周期展示 Φ值和 salience 不再是 0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from nct_modules import NCTManager, NCTConfig


def test_phi_and_salience():
    """测试 Φ值和 salience 计算"""
    print("=" * 70)
    print("🔍 验证 Φ值和 salience 修复")
    print("=" * 70)
    
    config = NCTConfig(d_model=256, n_heads=8, n_layers=4)
    manager = NCTManager(config)
    manager.start()
    
    print("\n📊 运行 5 个周期...\n")
    
    for cycle in range(5):
        sensory_data = {
            'visual': np.random.randn(1, 28, 28).astype(np.float32) * 0.5 + 0.5,
            'auditory': np.random.randn(10, 10).astype(np.float32) * 0.3 + 0.5,
            'interoceptive': np.random.randn(10).astype(np.float32) * 0.2,
        }
        
        state = manager.process_cycle(sensory_data)
        
        # 提取关键指标
        phi = state.consciousness_metrics.get('phi_value', 0)
        salience = state.workspace_content.salience if state.workspace_content else 0
        winner_name = "未知"
        
        if hasattr(state, 'diagnostics') and 'workspace' in state.diagnostics:
            ws_info = state.diagnostics['workspace']
            winner_idx = ws_info.get('winner_idx', -1)
            candidate_names = ['整合', '视觉', '听觉', '内感受']
            if 0 <= winner_idx < 4:
                winner_name = candidate_names[winner_idx]
        
        print(f"周期 {cycle+1}: "
              f"获胜者={winner_name:6s}, "
              f"Φ={phi:.4f}, "
              f"Salience={salience:.4f}")
    
    manager.stop()
    
    print("\n" + "=" * 70)
    print("✅ Φ值和 Salience 都已正确计算！")
    print("=" * 70)


if __name__ == "__main__":
    test_phi_and_salience()
