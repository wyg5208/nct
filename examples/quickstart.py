"""
NCT v3.0 快速开始示例
NeuroConscious Transformer Quickstart

运行方式：
    python quickstart.py

功能：
1. 初始化 NCT 管理器
2. 处理 5 个意识周期
3. 显示意识状态指标
"""

import sys
import os

# 添加 NCT 模块到路径（向上一级）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nct_modules import NCTManager, NCTConfig
import numpy as np


def main():
    print("=" * 70)
    print("NeuroConscious Transformer v3.0 - 快速开始")
    print("=" * 70)
    
    # 1. 创建配置
    config = NCTConfig(
        n_heads=8,           # 工作空间容量（Miller's Law 7±2）
        n_layers=4,          # 皮层层次结构
        d_model=768,         # 表征维度
        gamma_freq=40.0,     # γ波频率（Hz）
    )
    
    print(f"\n[OK] 配置已创建:")
    print(f"  - 注意力头数：{config.n_heads}")
    print(f"  - Transformer 层数：{config.n_layers}")
    print(f"  - 模型维度：{config.d_model}")
    print(f"  - γ频率：{config.gamma_freq} Hz")
    
    # 2. 创建管理器
    manager = NCTManager(config)
    manager.start()
    
    print(f"\n[OK] NCT 管理器已启动")
    
    # 3. 处理意识周期
    print(f"\n开始处理意识周期...")
    print("-" * 70)
    
    for cycle in range(5):
        # 准备感觉输入（模拟）
        # Visual: 28x28 灰度图像
        visual_input = np.random.randn(1, 28, 28).astype(np.float32)
        # Audio: 10 个时间步 x 10 个特征 (T, F)
        audio_input = np.random.randn(10, 10).astype(np.float32)
        # Interoceptive: 10 个内感受信号
        intero_input = np.random.randn(10).astype(np.float32)
        
        sensory_data = {
            'visual': visual_input,
            'auditory': audio_input,
            'interoceptive': intero_input,
        }
        
        # 处理周期
        state = manager.process_cycle(sensory_data)
        
        # 显示结果
        print(f"\n【周期 {cycle + 1}/5】")
        print(f"  意识水平：{state.awareness_level}")
        print(f"  Φ值（整合信息）: {state.consciousness_metrics.get('phi_value', 0):.3f}")
        print(f"  自信度：{state.self_representation['confidence']:.3f}")
        print(f"  自由能（预测误差）: {state.self_representation['free_energy']:.4f}")
        
        if state.workspace_content:
            print(f"  [OK] 有意识内容 (显著性={state.workspace_content.salience:.3f})")
            if hasattr(state.workspace_content, 'modality_weights') and state.workspace_content.modality_weights:
                print(f"  模态贡献：{state.workspace_content.modality_weights}")
        else:
            print(f"  [XX] 无意识内容")
    
    # 4. 停止系统
    manager.stop()
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
    
    # 5. 显示统计信息
    stats = manager.get_stats()
    print(f"\n系统统计:")
    print(f"  - 总周期数：{stats['total_cycles']}")
    print(f"  - 运行状态：{stats['is_running']}")
    if 'workspace' in stats and stats['workspace']:
        print(f"  - 工作空间状态：{stats['workspace']}")


if __name__ == "__main__":
    main()
