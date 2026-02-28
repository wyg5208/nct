"""
NCT for CIFAR-10 - Color Image Adaptation
==========================================
CIFAR-10 专用的 NCT 架构，适配 RGB 输入和 32x32 图像

架构调整：
1. 输入层：3 通道 RGB
2. Patch size 调整以适应更大图像
3. 数据增强集成
4. 保持与 MNIST 版本相同的核心理论

Author: WENG YONGGANG
Affiliation: NeuroConscious Lab, Universiti Teknologi Malaysia
Date: February 24, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class NCTForCIFAR10(nn.Module):
    """
    NCT CIFAR-10 分类器
    
    关键适配：
    1. RGB 3 通道输入
    2. 32x32 图像尺寸
    3. 更强的数据增强（ColorJitter, RandomCrop）
    4. 更深的 CNN backbone（适应彩色图像复杂度）
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 32, 32),
        d_model: int = 512,  # 更大的 d_model 以适应彩色图像
        n_heads: int = 8,
        n_layers: int = 4,
        dim_ff: int = 1024,
        num_classes: int = 10,
        dropout_rate: float = 0.5,
        n_candidates: int = 20
    ):
        super().__init__()
        
        self.input_shape = input_shape
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_candidates = n_candidates
        
        # 1. RGB 优化的 CNN 编码器
        self.encoder = self._create_rgb_encoder(input_shape)
        
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
            nn.LayerNorm(d_model * n_candidates),
            nn.Dropout(dropout_rate)
        )
        
        # 5. 注意力竞争机制
        self.competition_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )
        
        # 6. 预测编码层次
        self.predictive_hierarchy = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model)
            ) for _ in range(n_layers)
        ])
        
        # 7. 分类头（更强正则化）
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model // 4, num_classes)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _create_rgb_encoder(self, input_shape):
        """创建 RGB 优化的 CNN 编码器"""
        input_channels = input_shape[0]
        
        return nn.Sequential(
            # Block 1: [B, 3, 32, 32] → [B, 64, 16, 16]
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 2: [B, 64, 16, 16] → [B, 128, 8, 8]
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 3: [B, 128, 8, 8] → [B, 256, 4, 4]
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # Block 4: [B, 256, 4, 4] → [B, 512, 2, 2]
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global pooling
            
            # Flatten and project
            nn.Flatten(1),
            nn.Linear(512, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Dropout(0.5)
        )
    
    def _init_weights(self):
        """Xavier/He 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码输入图像"""
        features = self.encoder(x)
        return features
    
    def generate_candidates(self, features: torch.Tensor) -> torch.Tensor:
        """生成多个候选表征"""
        batch_size = features.size(0)
        candidates_flat = self.candidate_generator(features)
        candidates = candidates_flat.view(batch_size, self.n_candidates, self.d_model)
        return candidates
    
    def compete(self, query: torch.Tensor, candidates: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """注意力竞争选择"""
        attended, attention_weights = self.competition_attention(
            query=query,
            key=candidates,
            value=candidates
        )
        
        attention_weights = attention_weights.squeeze(1)
        winner_idx = attention_weights.argmax(dim=1, keepdim=True)
        winner = torch.gather(candidates, 1, winner_idx.unsqueeze(-1).expand(-1, -1, self.d_model))
        winner = winner.squeeze(1)
        
        return winner, attention_weights
    
    def compute_phi(self, candidates: torch.Tensor, attention_weights: torch.Tensor) -> torch.Tensor:
        """计算 Φ 值"""
        batch_size, n_candidates, _ = candidates.shape
        attention_probs = F.softmax(attention_weights, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-9), dim=-1)
        max_entropy = torch.log(torch.tensor(n_candidates, dtype=torch.float))
        integration = 1.0 - (entropy / max_entropy)
        phi = integration * attention_probs.max(dim=-1)[0]
        return phi
    
    def compute_prediction_error(self, features: torch.Tensor) -> torch.Tensor:
        """计算预测误差"""
        prediction_errors = []
        x = features
        for layer in self.predictive_hierarchy:
            predicted = layer(x)
            error = (predicted - x).pow(2).mean(dim=-1)
            prediction_errors.append(error)
            x = predicted
        total_pe = torch.stack(prediction_errors).mean(dim=0)
        return total_pe
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """前向传播"""
        batch_size = x.size(0)
        
        # Encode
        features = self.encode(x)
        
        # Add position
        features_with_pos = features.unsqueeze(1) + self.pos_embed
        
        # Transformer
        transformed = self.transformer(features_with_pos).squeeze(1)
        
        # Generate candidates
        candidates = self.generate_candidates(transformed)
        
        # Compete
        query = transformed.unsqueeze(1)
        winner, attention_weights = self.compete(query, candidates)
        
        # Compute metrics
        phi = self.compute_phi(candidates, attention_weights)
        prediction_error = self.compute_prediction_error(transformed)
        
        # Classify
        output = self.classifier(winner)
        
        return {
            'output': output,
            'phi': phi,
            'prediction_error': prediction_error,
            'attention_weights': attention_weights,
            'features': winner,
            'all_candidates': candidates
        }


if __name__ == '__main__':
    # Test the model
    print("Testing NCTForCIFAR10 model...")
    
    model = NCTForCIFAR10(
        input_shape=(3, 32, 32),
        d_model=512,
        n_heads=8,
        n_layers=4,
        num_classes=10
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    x = torch.randn(4, 3, 32, 32)
    output_dict = model(x)
    
    print(f"\nForward pass successful!")
    print(f"  Output shape: {output_dict['output'].shape}")
    print(f"  Φ shape: {output_dict['phi'].shape}")
    print(f"  Prediction error shape: {output_dict['prediction_error'].shape}")
    print(f"  Attention weights shape: {output_dict['attention_weights'].shape}")
    
    print("\n✓ NCTForCIFAR10 is ready!")
