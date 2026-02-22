"""
为 PredictiveHierarchy 添加 forward_with_sequence 方法
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nct_modules.nct_predictive_coding import PredictiveHierarchy

# 动态添加新方法
def forward_with_sequence(self, sensory_sequence):
    """处理序列感觉输入（支持时间维度）
    
    Args:
        sensory_sequence: [B, T, D] 感觉输入序列
    
    Returns:
        包含各层预测和误差的字典
    """
    import torch
    import torch.nn.functional as F
    
    results = {
        'predictions': [],
        'errors': [],
        'hidden_states': [],
    }
    
    B, T, D = sensory_sequence.shape
    x = sensory_sequence  # [B, T, D]
    
    for i, layer in enumerate(self.layers):
        # 该层的预测（使用完整的 Transformer Decoder）
        try:
            prediction, hidden = layer.forward(x)
            
            # 计算预测误差
            if i == 0:
                # L1 对比最后一个时间步的实际输入
                actual = sensory_sequence[:, -1, :]  # [B, D]
                error = F.mse_loss(prediction, actual, reduction='mean')
            else:
                # 高层对比来自低层的输入
                error = torch.abs(prediction - x[:, -1, :])
            
            results['predictions'].append(prediction)
            results['errors'].append(error)
            results['hidden_states'].append(hidden)
            
            # 传递到下一层（自下而上）
            if i < len(self.layers) - 1:
                # hidden shape: [B, T, D_next]
                # 确保是 3D 张量
                if hidden.dim() == 2:
                    hidden = hidden.unsqueeze(1)  # [B, 1, D_next]
                x = self.bottom_up_projections[i](hidden)  # [B, T, D_next]
        except Exception as e:
            from logging import getLogger
            import traceback
            logger = getLogger(__name__)
            logger.warning(f"[PredictiveHierarchy] Layer {i} 处理失败：{e}")
            logger.warning(f"Traceback:\n{traceback.format_exc()}")
            results['predictions'].append(torch.zeros(B, D))
            results['errors'].append(torch.zeros(B, D))  # 修复：返回与 prediction 相同形状的 tensor
            results['hidden_states'].append(torch.zeros(B, T, D))
            
            # 继续传递到下一层（使用零向量）
            if i < len(self.layers) - 1:
                x = torch.zeros(B, 1, self.bottom_up_projections[i].out_features)
    
    # 总自由能
    total_free_energy = sum(
        e.mean().item() if isinstance(e, torch.Tensor) else e 
        for e in results['errors']
    )
    results['total_free_energy'] = total_free_energy
    
    return results

# 绑定到类
PredictiveHierarchy.forward_with_sequence = forward_with_sequence

print("[OK] 成功添加 forward_with_sequence 方法到 PredictiveHierarchy")

if __name__ == '__main__':
    # 测试
    from nct_modules import NCTConfig, NCTManager
    import numpy as np
    
    config = NCTConfig()
    manager = NCTManager(config)
    
    # 运行几个周期来填充历史
    for i in range(3):
        sensory = {
            'visual': np.random.randn(1, 28, 28).astype(np.float32),
            'auditory': np.random.randn(10, 10).astype(np.float32),
            'interoceptive': np.random.randn(10).astype(np.float32),
        }
        state = manager.process_cycle(sensory)
        print(f"周期 {i+1}: Φ={state.consciousness_metrics.get('phi_value', 0):.3f}, "
              f"自信度={state.self_representation['confidence']:.3f}")
