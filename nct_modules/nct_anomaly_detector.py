"""
NCT Batch-Compatible Simplified Model for Anomaly Detection
============================================================
为异常检测实验创建的简化版 NCT，支持批量训练和推理

核心特性：
1. 保留 NCT 核心理论（GWT、Φ计算、预测编码）
2. 使用标准 PyTorch 接口（支持批量处理）
3. 优化用于异常检测任务

Author: WENG YONGGANG
Affiliation: NeuroConscious Lab, Universiti Teknologi Malaysia
Date: February 24, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class SimplifiedNCT(nn.Module):
    """
    简化版 NCT 分类器
    
    架构设计原则：
    1. 保留注意力全局工作空间机制
    2. 集成预测编码层次
    3. 实时计算 Φ 值
    4. 支持批量处理
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        d_model: int = 384,
        n_heads: int = 6,
        n_layers: int = 3,
        dim_ff: int = 768,
        num_classes: int = 10,
        dropout_rate: float = 0.4,
        n_candidates: int = 15
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_candidates = n_candidates
        
        # 1. 图像编码器（CNN backbone）
        self.encoder = self._create_encoder(input_shape[0])
        
        # 2. 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # 3. Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. 多候选生成器（Global Workspace）
        self.candidate_generator = nn.Sequential(
            nn.Linear(d_model, d_model * n_candidates),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # 5. 注意力竞争机制
        self.competition_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
        # 6. 预测编码层次（简化版）
        self.predictive_hierarchy = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(n_layers)
        ])
        
        # 7. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _create_encoder(self, input_channels: int):
        """创建 CNN 编码器"""
        return nn.Sequential(
            # Block 1: [B, C, H, W] → [B, 64, H/2, W/2]
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 2: [B, 64, H/2, W/2] → [B, 128, H/4, W/4]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 3: [B, 128, H/4, W/4] → [B, 256, H/8, W/8]
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            nn.Flatten(1),
            
            # Projection to d_model
            nn.Linear(256, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.4)
        )
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入图像"""
        features = self.encoder(x)  # [B, d_model]
        return features
    
    def generate_candidates(self, features: torch.Tensor) -> torch.Tensor:
        """生成多个候选表征"""
        batch_size = features.size(0)
        
        # 生成 n_candidates 个假设
        candidates_flat = self.candidate_generator(features)  # [B, d_model * n_cand]
        candidates = candidates_flat.view(batch_size, self.n_candidates, self.d_model)
        
        return candidates  # [B, n_cand, d_model]
    
    def compete(self, query: torch.Tensor, candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意力竞争机制
        
        Returns:
            winner: 获胜的候选 [B, d_model]
            attention_weights: 注意力权重 [B, n_cand]
        """
        # query shape: [B, 1, d_model]
        # candidates shape: [B, n_cand, d_model]
        
        attended, attention_weights = self.competition_attention(
            query=query,
            key=candidates,
            value=candidates
        )
        
        # attention_weights shape: [B, 1, n_cand]
        attention_weights = attention_weights.squeeze(1)  # [B, n_cand]
        
        # 选择获胜者
        winner_idx = attention_weights.argmax(dim=1, keepdim=True)  # [B, 1]
        winner = torch.gather(candidates, 1, winner_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        winner = winner.squeeze(1)  # [B, d_model]
        
        return winner, attention_weights
    
    def compute_phi(self, candidates: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        计算整合信息量 Φ
        
        基于注意力流的简化实现：
        Φ = 有效信息 (Effective Information)
        """
        batch_size, n_candidates, _ = candidates.shape
        
        # 计算注意力流矩阵（归一化）
        # T_ij = 从候选 j 到整体的信息流
        attention_probs = F.softmax(attention_weights, dim=-1)  # [B, n_cand]
        
        # 计算熵（不确定性）
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-9), dim=-1)  # [B]
        
        # 最大熵（均匀分布）
        max_entropy = np.log(n_candidates)
        
        # 归一化整合度
        integration = 1.0 - (entropy / max_entropy)  # [B]
        
        # Φ = integration * attention strength
        phi = integration * attention_probs.max(dim=-1)[0]  # [B]
        
        return phi
    
    def compute_prediction_error(self, features: torch.Tensor) -> torch.Tensor:
        """计算预测误差（Free Energy 代理）"""
        prediction_errors = []
        
        x = features
        for layer in self.predictive_hierarchy:
            # 前向预测
            predicted = layer(x)
            
            # 计算误差
            error = (predicted - x).pow(2).mean(dim=-1)  # [B]
            prediction_errors.append(error)
            
            x = predicted
        
        # 总预测误差
        total_pe = torch.stack(prediction_errors).mean(dim=0)  # [B]
        
        return total_pe
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Returns:
            dict: {
                'output': 分类 logits [B, num_classes],
                'phi': Φ值 [B],
                'prediction_error': 预测误差 [B],
                'attention_weights': 注意力权重 [B, n_cand],
                'features': 特征表示 [B, d_model]
            }
        """
        batch_size = x.size(0)
        
        # Step 1: 编码
        features = self.encode(x)  # [B, d_model]
        
        # Step 2: 添加位置维度
        features_with_pos = features.unsqueeze(1) + self.pos_embed  # [B, 1, d_model]
        
        # Step 3: Transformer 处理
        transformed = self.transformer(features_with_pos)  # [B, 1, d_model]
        transformed = transformed.squeeze(1)  # [B, d_model]
        
        # Step 4: 生成候选
        candidates = self.generate_candidates(transformed)  # [B, n_cand, d_model]
        
        # Step 5: 竞争选择
        query = transformed.unsqueeze(1)  # [B, 1, d_model]
        winner, attention_weights = self.compete(query, candidates)
        
        # Step 6: 计算 Φ
        phi = self.compute_phi(candidates, attention_weights)
        
        # Step 7: 计算预测误差
        prediction_error = self.compute_prediction_error(transformed)
        
        # Step 8: 分类
        output = self.classifier(winner)  # [B, num_classes]
        
        return {
            'output': output,
            'phi': phi,
            'prediction_error': prediction_error,
            'attention_weights': attention_weights,
            'features': winner,
            'all_candidates': candidates
        }


class NCTAnomalyDetectorV2:
    """
    异常检测器 V2 - 基于 SimplifiedNCT
    
    三种检测方法：
    1. 预测误差（Prediction Error）
    2. Φ值下降（Information Integration Drop）
    3. 注意力熵异常（Attention Entropy）
    """
    
    def __init__(self, model: SimplifiedNCT, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # 阈值（在校准阶段设置）
        self.pe_threshold = None
        self.phi_threshold = None
        self.entropy_threshold = None
        
        # 正常数据统计
        self.stats = {}
    
    def calibrate(self, normal_loader: torch.utils.data.DataLoader, percentile: float = 95.0):
        """
        在正常数据上校准阈值
        
        Args:
            normal_loader: 正常数据的 DataLoader
            percentile: 百分位数（如 95 表示 top 5% 为异常）
        """
        self.model.eval()
        
        all_pes = []
        all_phis = []
        all_entropies = []
        
        with torch.no_grad():
            for data, _ in normal_loader:
                data = data.to(self.device)
                
                # Forward pass
                output_dict = self.model(data)
                
                # Collect metrics
                all_pes.extend(output_dict['prediction_error'].cpu().numpy())
                all_phis.extend(output_dict['phi'].cpu().numpy())
                
                # Compute attention entropy
                attn_weights = output_dict['attention_weights']  # [B, n_cand]
                attn_probs = F.softmax(attn_weights, dim=-1)
                entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1)
                all_entropies.extend(entropy.cpu().numpy())
        
        # Set thresholds
        self.pe_threshold = np.percentile(all_pes, percentile)
        self.phi_threshold = np.percentile(all_phis, 100 - percentile)  # Low Φ is anomalous
        self.entropy_threshold = np.percentile(all_entropies, percentile)
        
        # Save statistics
        self.stats = {
            'pe_mean': np.mean(all_pes),
            'pe_std': np.std(all_pes),
            'phi_mean': np.mean(all_phis),
            'phi_std': np.std(all_phis),
            'entropy_mean': np.mean(all_entropies),
            'entropy_std': np.std(all_entropies)
        }
        
        print(f"Calibration completed:")
        print(f"  PE threshold: {self.pe_threshold:.4f} (>{self.pe_threshold:.4f} → anomaly)")
        print(f"  Φ threshold: {self.phi_threshold:.4f} (<{self.phi_threshold:.4f} → anomaly)")
        print(f"  Entropy threshold: {self.entropy_threshold:.4f} (>{self.entropy_threshold:.4f} → anomaly)")
        
        return self.stats
    
    def detect(self, test_loader: torch.utils.data.DataLoader, ground_truth=None) -> Dict:
        """
        检测异常
        
        Args:
            test_loader: 测试数据 DataLoader
            ground_truth: 可选的真实标签（1=异常，0=正常）
        
        Returns:
            dict: 检测结果和指标
        """
        self.model.eval()
        
        all_predictions = {
            'pe': [],
            'phi': [],
            'entropy': [],
            'combined': []
        }
        
        all_metrics = {
            'pes': [],
            'phis': [],
            'entropies': [],
            'confidences': []
        }
        
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                
                # Forward pass
                output_dict = self.model(data)
                
                # Extract metrics
                pes = output_dict['prediction_error'].cpu().numpy()
                phis = output_dict['phi'].cpu().numpy()
                
                attn_weights = output_dict['attention_weights']
                attn_probs = F.softmax(attn_weights, dim=-1)
                entropies = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1).cpu().numpy()
                
                # Confidence (softmax probability)
                logits = output_dict['output']
                probs = F.softmax(logits, dim=-1)
                confidences = probs.max(dim=-1)[0].cpu().numpy()
                
                # Detect anomalies
                pred_pe = (pes > self.pe_threshold).astype(int)
                pred_phi = (phis < self.phi_threshold).astype(int)
                pred_entropy = (entropies > self.entropy_threshold).astype(int)
                
                # Combined: any signal triggers
                pred_combined = ((pred_pe + pred_phi + pred_entropy) >= 1).astype(int)
                
                # Store
                all_predictions['pe'].extend(pred_pe)
                all_predictions['phi'].extend(pred_phi)
                all_predictions['entropy'].extend(pred_entropy)
                all_predictions['combined'].extend(pred_combined)
                
                all_metrics['pes'].extend(pes)
                all_metrics['phis'].extend(phis)
                all_metrics['entropies'].extend(entropies)
                all_metrics['confidences'].extend(confidences)
        
        # Compute metrics
        results = self._compute_metrics(all_predictions, ground_truth)
        results.update({
            'pe_values': all_metrics['pes'],
            'phi_values': all_metrics['phis'],
            'entropy_values': all_metrics['entropies'],
            'confidence_values': all_metrics['confidences']
        })
        
        return results
    
    def _compute_metrics(self, predictions, ground_truth) -> Dict:
        """计算检测指标"""
        if ground_truth is None:
            return {}
        
        y_true = np.array(ground_truth)
        metrics = {}
        
        for method in ['pe', 'phi', 'entropy', 'combined']:
            y_pred = np.array(predictions[method])
            
            tp = np.sum((y_pred == 1) & (y_true == 1))
            fp = np.sum((y_pred == 1) & (y_true == 0))
            tn = np.sum((y_pred == 0) & (y_true == 0))
            fn = np.sum((y_pred == 0) & (y_true == 1))
            
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            specificity = tn / (tn + fp + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-9)
            
            metrics[f'{method}_precision'] = precision
            metrics[f'{method}_recall'] = recall
            metrics[f'{method}_specificity'] = specificity
            metrics[f'{method}_f1'] = f1
            metrics[f'{method}_accuracy'] = accuracy
            metrics[f'{method}_confusion'] = {
                'tp': int(tp), 'fp': int(fp),
                'tn': int(tn), 'fn': int(fn)
            }
        
        return metrics


if __name__ == '__main__':
    # Test the model
    print("Testing SimplifiedNCT model...")
    
    model = SimplifiedNCT(
        input_shape=(1, 28, 28),
        d_model=384,
        n_heads=6,
        n_layers=3,
        num_classes=10
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    x = torch.randn(8, 1, 28, 28)
    output_dict = model(x)
    
    print(f"\nForward pass successful!")
    print(f"  Output shape: {output_dict['output'].shape}")
    print(f"  Φ shape: {output_dict['phi'].shape}")
    print(f"  Prediction error shape: {output_dict['prediction_error'].shape}")
    print(f"  Attention weights shape: {output_dict['attention_weights'].shape}")
    
    print("\n✓ SimplifiedNCT is ready for anomaly detection experiments!")
