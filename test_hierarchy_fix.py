"""测试 PredictiveHierarchy 修复"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from nct_modules.nct_predictive_coding import PredictiveHierarchy

# 创建配置
config = {
    'layer0_dim': 256,
    'layer1_dim': 256,
    'layer2_dim': 256,
    'layer3_dim': 256,
    'n_heads': 8,
}

# 创建层次结构
hierarchy = PredictiveHierarchy(config)

# 测试 forward 方法
print("Testing forward()...")
sensory_input = torch.randn(1, 256)  # [B, D]
try:
    results = hierarchy.forward(sensory_input)
    print(f"[OK] forward() success!")
    print(f"  - Total free energy: {results['total_free_energy']:.4f}")
    print(f"  - Layers: {len(results['predictions'])}")
except Exception as e:
    print(f"[FAIL] forward() failed: {e}")

# 测试 forward_with_sequence 方法
print("\nTesting forward_with_sequence()...")
sensory_sequence = torch.randn(1, 5, 256)  # [B, T, D]
try:
    results = hierarchy.forward_with_sequence(sensory_sequence)
    print(f"[OK] forward_with_sequence() success!")
    print(f"  - Total free energy: {results['total_free_energy']:.4f}")
    print(f"  - Layers: {len(results['predictions'])}")
except Exception as e:
    print(f"[FAIL] forward_with_sequence() failed: {e}")

print("\n所有测试完成！")
