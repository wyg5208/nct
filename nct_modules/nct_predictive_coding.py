"""
NeuroConscious Transformer - Predictive Coding as Decoder-Only Transformer
预测编码 = GPT 风格的因果 Transformer

核心理论统一：
Friston 自由能原理 ↔ Transformer 训练目标

数学等价性证明：
F = E_q(z)[ln q(z) - ln p(s,z)]           # 变分自由能
  = CrossEntropy(predictions, actual)      # Transformer loss
    + KL(q||p)                             # 正则化项

架构设计：
- Causal self-attention（GPT 风格）
- Next token prediction = 预测下一时刻感觉输入
- Loss = Free Energy（预测误差）

生物合理性：
- 皮层层次结构（L1→L2→L3→L4）↔ Transformer Decoder 层
- 自下而上预测误差 ↔ Residual connections
- 自上而下预测 ↔ Causal attention

作者：WinClaw Research Team
创建：2026 年 2 月 21 日
版本：v3.1.0
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# 核心模块：Predictive Coding Decoder
# ============================================================================

class PredictiveCodingDecoder(nn.TransformerDecoder):
    """预测编码解码器
    
    将 Friston 的预测编码理论实现为 GPT 风格的 causal transformer
    
    关键特性：
    1. Causal masking（只能看到过去的信息）
    2. Next token prediction = 预测下一时刻感觉输入
    3. Loss = Free Energy（预测误差）
    
    层级对应：
    - L1: 初级感觉皮层（V1/A1）
    - L2-L3: 联合皮层
    - L4: 前额叶（最高级预测）
    """
    
    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        # 构建 decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        super().__init__(decoder_layer, num_layers=n_layers)
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # 位置编码（可学习）
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # 输出投影层（预测下一时刻）
        self.output_projection = nn.Linear(d_model, d_model)
        
        logger.info(
            f"[PredictiveCodingDecoder] 初始化："
            f"{n_layers} layers, d_model={d_model}, {n_heads} heads"
        )
    
    def forward(
        self,
        sensory_sequence: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播：预测下一时刻感觉输入
        
        Args:
            sensory_sequence: [B, T, D] 感觉输入序列
            memory: 可选的长期记忆（来自海马）
        
        Returns:
            prediction: 预测的下一时刻 [B, D]
            hidden_states: 中间隐藏状态 [B, T, D]
        """
        B, T, D = sensory_sequence.shape
        
        # Step 1: 添加位置编码
        x = sensory_sequence + self.pos_encoding[:, :T, :]
        
        # Step 2: 生成 causal mask（只能看到过去）
        causal_mask = self._generate_causal_mask(T)
        
        # Step 3: 处理 memory（decoder-only 模式）
        # 如果 memory=None，使用 x 作为 memory（自注意力）
        # 这是 GPT 风格的 decoder-only 实现
        if memory is None:
            memory = x  # decoder-only: 使用自身作为 memory
        
        # Step 4: Transformer 前向传播（逐层传递）
        hidden_states = x
        for module in self.layers:
            hidden_states = module(hidden_states, memory=memory, tgt_mask=causal_mask)
        
        # Step 5: 提取最后一层最后一个 token 作为预测基础
        last_hidden = hidden_states[:, -1, :]  # [B, D]
        
        # Step 6: 投影到预测空间
        prediction = self.output_projection(last_hidden)
        
        return prediction, hidden_states
    
    @staticmethod
    def _generate_causal_mask(seq_len: int) -> torch.Tensor:
        """生成因果掩码（上三角为 -inf）"""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def predict_next_sensory(
        self,
        sensory_history: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, float]:
        """预测下一个感觉输入
        
        Args:
            sensory_history: 历史感觉输入列表
        
        Returns:
            prediction: 预测值
            prediction_error: 预测误差（自由能）
        """
        if not sensory_history:
            return torch.zeros(1, self.d_model), float('inf')
        
        # 堆叠为序列
        sequence = torch.stack(sensory_history, dim=0).unsqueeze(0)  # [1, T, D]
        
        # 预测
        with torch.no_grad():
            prediction, _ = self.forward(sequence)
        
        # 如果有实际值，计算预测误差
        if len(sensory_history) > 1:
            actual = sensory_history[-1]
            prediction_error = F.mse_loss(prediction.squeeze(0), actual).item()
        else:
            prediction_error = 0.0
        
        return prediction, prediction_error
    
    def minimize_free_energy(
        self,
        predictions: torch.Tensor,
        actual_values: torch.Tensor,
    ) -> torch.Tensor:
        """最小化自由能（反向传播）
        
        自由能 = 预测误差 + KL 散度
        
        Args:
            predictions: 预测值 [B, D]
            actual_values: 实际值 [B, D]
        
        Returns:
            free_energy: 标量
        """
        # 预测误差（主要项）
        prediction_error = F.mse_loss(predictions, actual_values, reduction='mean')
        
        # KL 散度（正则化项，防止过拟合）
        # 这里简化处理，使用 L2 正则化近似
        kl_divergence = sum(p.pow(2).sum() for p in self.parameters()) * 1e-4
        
        # 总自由能
        free_energy = prediction_error + kl_divergence
        
        # 反向传播
        free_energy.backward()
        
        logger.debug(
            f"[PredictiveCodingDecoder] 自由能："
            f"F={free_energy.item():.4f} (error={prediction_error.item():.4f}, "
            f"KL={kl_divergence.item():.4f})"
        )
        
        return free_energy
    
    def get_prediction_error_history(self) -> List[float]:
        """获取预测误差历史"""
        return self._error_history if hasattr(self, '_error_history') else []
    
    def track_prediction_error(self, error: float):
        """记录预测误差"""
        if not hasattr(self, '_error_history'):
            self._error_history = []
        self._error_history.append(error)
        # 保持最近 1000 条记录
        if len(self._error_history) > 1000:
            self._error_history.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
        }
        
        error_history = self.get_prediction_error_history()
        if error_history:
            stats['error_history'] = {
                'length': len(error_history),
                'mean': float(np.mean(error_history)),
                'std': float(np.std(error_history)),
                'min': float(np.min(error_history)),
                'max': float(np.max(error_history)),
                'recent': error_history[-10:],
            }
        
        return stats
    
    def visualize_error_history(self, save_path: str = None) -> Dict[str, Any]:
        """可视化预测误差历史"""
        error_history = self.get_prediction_error_history()
        
        if not error_history:
            return {'error': 'No error history available'}
        
        result = {
            'length': len(error_history),
            'mean': float(np.mean(error_history)),
            'trend': 'decreasing' if error_history[-1] < error_history[0] else 'increasing',
        }
        
        if save_path:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(error_history, color='steelblue', linewidth=1)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Prediction Error (Free Energy)')
            ax.set_title('Prediction Error Over Time')
            ax.grid(True, alpha=0.3)
            
            # 添加趋势线
            if len(error_history) > 10:
                z = np.polyfit(range(len(error_history)), error_history, 1)
                p = np.poly1d(z)
                ax.plot(range(len(error_history)), p(range(len(error_history))), 
                       'r--', linewidth=2, label=f'Trend: {z[0]:.6f}/step')
                ax.legend()
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            result['saved_path'] = save_path
            logger.info(f"[PredictiveCodingDecoder] 误差历史图已保存到 {save_path}")
        
        return result


# ============================================================================
# 预测编码层次结构
# ============================================================================

class PredictiveHierarchy(nn.Module):
    """预测编码层次结构
    
    模拟皮层的 4 层层次结构：
    L1 → L2 → L3 → L4
    
    每层都进行预测和误差计算：
    - 低层传递预测误差到高层
    - 高层传递预测到低层
    """
    
    def __init__(self, config: Dict[str, int]):
        super().__init__()
        
        # 4 层预测编码器
        self.layers = nn.ModuleList([
            PredictiveCodingDecoder(
                d_model=config.get(f'layer{i}_dim', 768),
                n_heads=config.get('n_heads', 8),
                n_layers=config.get(f'layer{i}_layers', 2),
            )
            for i in range(4)
        ])
        
        # 层间连接（自下而上）
        self.bottom_up_projections = nn.ModuleList([
            nn.Linear(config.get(f'layer{i}_dim', 768), 
                     config.get(f'layer{i+1}_dim', 768))
            for i in range(3)
        ])
        
        # 层间连接（自上而下）
        self.top_down_projections = nn.ModuleList([
            nn.Linear(config.get(f'layer{i+1}_dim', 768), 
                     config.get(f'layer{i}_dim', 768))
            for i in range(3)
        ])
        
        # 历史缓存（用于序列处理）
        self.history_buffer: List[torch.Tensor] = []
        self.max_history = 10
        
        logger.info("[PredictiveHierarchy] 初始化 4 层预测编码结构")
    
    def forward(
        self,
        sensory_input: torch.Tensor,
    ) -> Dict[str, Any]:
        """处理感觉输入，计算各层预测误差
        
        Args:
            sensory_input: [B, D] 当前感觉输入
        
        Returns:
            包含各层预测和误差的字典
        """
        results = {
            'predictions': [],
            'errors': [],
            'hidden_states': [],
        }
        
        # 添加到历史缓存
        self.history_buffer.append(sensory_input.detach())
        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)
        
        # 构造序列：使用历史缓存
        if len(self.history_buffer) > 1:
            sequence = torch.stack(self.history_buffer, dim=1)  # [B, T, D]
        else:
            sequence = sensory_input.unsqueeze(1)  # [B, 1, D]
        
        # L1: 初级感觉皮层
        x = sequence
        
        for i, layer in enumerate(self.layers):
            try:
                # 该层的预测
                prediction, hidden = layer(x)
                
                # 计算预测误差
                if i == 0:
                    # L1 直接对比感觉输入
                    error = F.mse_loss(prediction.squeeze(1), sensory_input, reduction='none')
                else:
                    # 高层对比来自低层的输入
                    error = torch.abs(prediction.squeeze(1) - x.mean(dim=1))
                
                results['predictions'].append(prediction)
                results['errors'].append(error)
                results['hidden_states'].append(hidden)
                
                # 传递到下一层（自下而上）
                if i < len(self.layers) - 1:
                    x = self.bottom_up_projections[i](hidden)
            except Exception as e:
                logger.warning(f"[PredictiveHierarchy] Layer {i} 处理失败：{e}，跳过")
                results['predictions'].append(None)
                results['errors'].append(torch.zeros_like(sensory_input))
                results['hidden_states'].append(None)
        
        # 总自由能（所有层的误差之和）
        valid_errors = [e for e in results['errors'] if e is not None]
        if valid_errors:
            total_free_energy = sum(e.mean().item() for e in valid_errors)
        else:
            total_free_energy = 0.0
        results['total_free_energy'] = total_free_energy
        
        logger.debug(
            f"[PredictiveHierarchy] 自由能：F={total_free_energy:.4f}"
        )
        
        return results
    
    def forward_with_sequence(
        self,
        sensory_sequence: torch.Tensor,
    ) -> Dict[str, Any]:
        """处理完整序列（用于多时间步预测）
        
        Args:
            sensory_sequence: [B, T, D] 感觉输入序列
        
        Returns:
            包含各层预测和误差的字典
        """
        results = {
            'predictions': [],
            'errors': [],
            'hidden_states': [],
        }
        
        x = sensory_sequence
        
        for i, layer in enumerate(self.layers):
            try:
                # 该层的预测
                prediction, hidden = layer(x)
                
                # 计算预测误差
                error = F.mse_loss(prediction, x[:, -1, :], reduction='none')
                
                results['predictions'].append(prediction)
                results['errors'].append(error)
                results['hidden_states'].append(hidden)
                
                # 传递到下一层（自下而上）
                if i < len(self.layers) - 1:
                    x = self.bottom_up_projections[i](hidden)
            except Exception as e:
                logger.warning(f"[PredictiveHierarchy] Layer {i} 处理失败：{e}，跳过")
                results['predictions'].append(None)
                results['errors'].append(torch.zeros(sensory_sequence.shape[0], sensory_sequence.shape[2]))
                results['hidden_states'].append(None)
        
        # 总自由能
        valid_errors = [e for e in results['errors'] if e is not None]
        if valid_errors:
            total_free_energy = sum(e.mean().item() for e in valid_errors)
        else:
            total_free_energy = 0.0
        results['total_free_energy'] = total_free_energy
        
        return results
    
    def get_layer_stats(self) -> List[Dict[str, Any]]:
        """获取各层统计信息"""
        stats = []
        for i, layer in enumerate(self.layers):
            layer_stat = {
                'layer_id': i,
                'd_model': layer.d_model,
                'n_heads': layer.n_heads,
                'n_layers': layer.n_layers,
            }
            
            # 获取误差历史
            error_history = layer.get_prediction_error_history()
            if error_history:
                layer_stat['error_mean'] = float(np.mean(error_history))
                layer_stat['error_std'] = float(np.std(error_history))
            
            stats.append(layer_stat)
        
        return stats
    
    def get_total_free_energy_history(self) -> List[float]:
        """获取总自由能历史"""
        return self._free_energy_history if hasattr(self, '_free_energy_history') else []
    
    def reset_history(self):
        """重置历史缓存"""
        self.history_buffer = []
        self._free_energy_history = []
        for layer in self.layers:
            if hasattr(layer, '_error_history'):
                layer._error_history = []
        logger.info("[PredictiveHierarchy] 历史已重置")
    
    def visualize_hierarchy_errors(self, save_path: str = None) -> Dict[str, Any]:
        """可视化各层预测误差"""
        layer_stats = self.get_layer_stats()
        
        result = {
            'layer_count': len(layer_stats),
            'layers': layer_stats,
        }
        
        if save_path:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
            
            for i, layer in enumerate(self.layers):
                if i >= 4:
                    break
                
                error_history = layer.get_prediction_error_history()
                if error_history:
                    axes[i].plot(error_history, color='steelblue', linewidth=1)
                    axes[i].set_xlabel('Time Step')
                    axes[i].set_ylabel('Prediction Error')
                    axes[i].set_title(f'Layer {i} (L{i+1} 皮层)')
                    axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            result['saved_path'] = save_path
            logger.info(f"[PredictiveHierarchy] 层级误差图已保存到 {save_path}")
        
        return result
    
    def export_state(self) -> Dict[str, Any]:
        """导出状态（用于保存）"""
        return {
            'history_buffer': [t.cpu().numpy().tolist() for t in self.history_buffer],
            'max_history': self.max_history,
        }
    
    def import_state(self, state: Dict[str, Any]):
        """导入状态（用于恢复）"""
        self.history_buffer = [torch.tensor(t) for t in state.get('history_buffer', [])]
        self.max_history = state.get('max_history', 10)
        logger.info(f"[PredictiveHierarchy] 已恢复状态，历史长度={len(self.history_buffer)}")


# ============================================================================
# 自我模型推断
# ============================================================================

class SelfModelInference(nn.Module):
    """自我模型推断
    
    基于预测编码的自我表征：
    "自我"是一个不断更新的生成模型
    
    数学形式：
    self_model = argmin_z F(sensory_data, z)
    
    其中 z 是自我表征，F 是自由能
    """
    
    def __init__(self, d_model: int = 768):
        super().__init__()
        
        # 自我表征（可学习的先验）
        self.self_prior = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 预测编码器
        self.predictive_coding = PredictiveCodingDecoder(d_model=d_model)
        
        # 自我表征投影
        self.self_projection = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, 64),  # 压缩到 64 维自我表征
        )
        
        logger.info("[SelfModelInference] 初始化自我模型")
    
    def infer_self(
        self,
        sensory_data: torch.Tensor,
        n_steps: int = 10,
    ) -> Dict[str, Any]:
        """推断自我模型
        
        Args:
            sensory_data: [B, D] 感觉输入
            n_steps: 优化步数
        
        Returns:
            self_representation: 自我表征
            confidence: 自信度
            free_energy: 自由能
        """
        B, D = sensory_data.shape
        
        # 初始自我表征（从先验开始）
        self_repr = self.self_prior.expand(B, -1, -1).clone()
        self_repr.requires_grad_(True)
        
        optimizer = torch.optim.Adam([self_repr], lr=0.01)
        
        # 迭代优化自由能
        for step in range(n_steps):
            optimizer.zero_grad()
            
            # 用自我表征预测感觉输入
            prediction, _ = self.predictive_coding(self_repr)
            
            # 计算自由能
            free_energy = F.mse_loss(prediction.squeeze(1), sensory_data)
            
            # 反向传播
            free_energy.backward()
            optimizer.step()
        
        # 投影到自我表征空间
        final_repr = self.self_projection(self_repr.squeeze(1))
        
        # 自信度（逆自由能）
        confidence = 1.0 / (1.0 + free_energy.detach().item())
        
        result = {
            'self_representation': final_repr.detach(),
            'confidence': confidence,
            'free_energy': free_energy.detach().item(),
            'identity_narrative': self._generate_identity_narrative(final_repr),
        }
        
        logger.info(
            f"[SelfModelInference] 自我推断完成："
            f"confidence={confidence:.3f}, F={free_energy.item():.4f}"
        )
        
        return result
    
    @staticmethod
    def _generate_identity_narrative(self_repr: torch.Tensor) -> str:
        """生成身份叙事（简化版）"""
        # 实际应用中可以用 LLM 生成
        return f"自我表征向量：{self_repr.shape}"


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    'PredictiveCodingDecoder',
    'PredictiveHierarchy',
    'SelfModelInference',
]
