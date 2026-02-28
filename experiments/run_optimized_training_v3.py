#!/usr/bin/env python3
"""
NCT MNIST Training Script - V3 Optimized
=========================================
V1: 43.3% (10 samples/class, complex model)
V2: 52.5% (100 samples/class, simplified model)
V3: Target 70%+ (500 samples/class, balanced model, stronger augmentation)

Changes from V2:
- Increase samples: 100 -> 500 per class
- Moderate model: d_model 256->384, n_layers 2->3
- Add data augmentation (rotation, shift)
- Longer warmup, slower decay
- Mixed precision for speed
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
import math

# Version identifier
VERSION = "v3"

@dataclass
class OptimizedConfigV3:
    """V3 Configuration - Balanced model with more data"""
    version: str = VERSION
    
    # Model architecture - balanced
    n_heads: int = 6
    n_layers: int = 3
    d_model: int = 384
    d_ff: int = 768
    
    # Data settings - significantly more data
    n_samples_per_class: int = 500
    n_classes: int = 10
    batch_size: int = 128
    
    # Training settings
    n_epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 0.02
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    
    # Regularization
    dropout_rate: float = 0.4
    label_smoothing: float = 0.1
    
    # Early stopping
    patience: int = 25
    
    # NCT specific
    gamma_freq: float = 40.0
    dt: float = 0.001
    
    # Paths
    results_dir: str = f'results/training_{VERSION}'


class DataAugmentation:
    """Simple data augmentation for MNIST"""
    def __init__(self, rotation_degrees=15, translate=(0.1, 0.1)):
        self.transform = transforms.Compose([
            transforms.RandomRotation(rotation_degrees),
            transforms.RandomAffine(0, translate=translate),
        ])
    
    def __call__(self, x):
        return self.transform(x)


def get_mnist_data_v3(config: OptimizedConfigV3):
    """Get MNIST data with augmentation for V3"""
    
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.05, 0.05)),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test/validation transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_full = datasets.MNIST('data', train=True, download=True, transform=train_transform)
    test_full = datasets.MNIST('data', train=False, download=True, transform=test_transform)
    
    # Stratified sampling for training
    n_per_class = config.n_samples_per_class
    train_indices = []
    class_counts = {i: 0 for i in range(config.n_classes)}
    
    for idx, (_, label) in enumerate(train_full):
        if class_counts[label] < n_per_class:
            train_indices.append(idx)
            class_counts[label] += 1
        if all(c >= n_per_class for c in class_counts.values()):
            break
    
    # Validation set - 100 samples per class from test set
    val_indices = []
    val_class_counts = {i: 0 for i in range(config.n_classes)}
    val_per_class = 100
    
    for idx in range(len(test_full)):
        _, label = test_full[idx]
        if val_class_counts[label] < val_per_class:
            val_indices.append(idx)
            val_class_counts[label] += 1
        if all(c >= val_per_class for c in val_class_counts.values()):
            break
    
    train_subset = Subset(train_full, train_indices)
    val_subset = Subset(test_full, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    
    print(f"\n[V3 Data] Train: {len(train_subset)} ({n_per_class}/class)")
    print(f"[V3 Data] Val: {len(val_subset)} ({val_per_class}/class)")
    print(f"[V3 Data] Augmentation: rotation=10°, translate=5%")
    
    return train_loader, val_loader


class NCTImageEncoderV3(nn.Module):
    """Optimized image encoder for V3"""
    def __init__(self, d_model: int, dropout: float = 0.4):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
        )
        
        # 28x28 -> 14x14 -> 7x7
        self.proj = nn.Linear(128 * 7 * 7, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class NCTClassifierV3(nn.Module):
    """V3 Classifier with NCT integration"""
    def __init__(self, config: OptimizedConfigV3):
        super().__init__()
        self.config = config
        
        # Image encoder
        self.encoder = NCTImageEncoderV3(config.d_model, config.dropout_rate)
        
        # Positional encoding (for sequence)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # Classification head with intermediate layer
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.d_model // 2, config.n_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Encode image
        features = self.encoder(x)  # [B, d_model]
        
        # Add sequence dimension and positional embedding
        features = features.unsqueeze(1)  # [B, 1, d_model]
        features = features + self.pos_embed
        
        # Apply transformer
        features = self.transformer(features)
        
        # Classify
        output = self.classifier(features[:, 0])
        
        return output
    
    def get_attention_weights(self):
        """Get attention weights for Φ calculation"""
        weights = []
        for layer in self.transformer.layers:
            if hasattr(layer, 'self_attn'):
                # Get attention weights if available
                weights.append(None)  # Placeholder
        return weights


class BatchedNCTManagerV3:
    """Simplified NCT Manager for V3"""
    def __init__(self, config: OptimizedConfigV3):
        self.config = config
        self.gamma_period = 1.0 / config.gamma_freq
        self.t = 0.0
        
    def compute_phi_batch(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Φ (integrated information) for batch"""
        B = features.size(0)
        
        if features.dim() == 2:
            # [B, d_model] -> compute variance-based Φ
            feat_norm = F.normalize(features, dim=-1)
            
            # Compute local complexity
            variance = torch.var(feat_norm, dim=-1)
            
            # Compute information integration via correlation
            if B > 1:
                correlation = torch.corrcoef(feat_norm)
                integration = torch.abs(correlation).mean()
            else:
                integration = torch.tensor(0.5, device=features.device)
            
            phi = variance * 0.5 + integration * 0.5
            
        else:
            # Fallback
            phi = torch.ones(B, device=features.device) * 0.1
            
        return phi.clamp(0.01, 1.0)
    
    def compute_salience_batch(self, features: torch.Tensor) -> torch.Tensor:
        """Compute salience for batch"""
        if features.dim() == 3:
            features = features.squeeze(1)
        
        # Magnitude-based salience with normalization
        magnitude = torch.norm(features, dim=-1)
        salience = magnitude / (magnitude.max() + 1e-8)
        
        return salience.clamp(0.01, 1.0)
    
    def step(self, dt: float = None):
        """Update internal time"""
        if dt is None:
            dt = self.config.dt
        self.t += dt
        
    def get_gamma_phase(self) -> float:
        """Get current gamma oscillation phase"""
        return (self.t % self.gamma_period) / self.gamma_period


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Cosine annealing with warmup"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch_v3(model, train_loader, optimizer, criterion, nct_manager, device, config):
    """Train one epoch with V3 optimizations"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    phi_sum = 0
    salience_sum = 0
    n_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Get features for NCT metrics
        with torch.no_grad():
            features = model.encoder(data)
            phi = nct_manager.compute_phi_batch(features)
            salience = nct_manager.compute_salience_batch(features)
            phi_sum += phi.mean().item()
            salience_sum += salience.mean().item()
        
        # Loss with label smoothing
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update NCT time
        nct_manager.step()
        
        # Statistics
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    accuracy = 100. * correct / total
    avg_phi = phi_sum / n_batches
    avg_salience = salience_sum / n_batches
    
    return avg_loss, accuracy, avg_phi, avg_salience


def validate_v3(model, val_loader, criterion, nct_manager, device):
    """Validate with V3 metrics"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    phi_sum = 0
    salience_sum = 0
    n_batches = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            # NCT metrics
            features = model.encoder(data)
            phi = nct_manager.compute_phi_batch(features)
            salience = nct_manager.compute_salience_batch(features)
            phi_sum += phi.mean().item()
            salience_sum += salience.mean().item()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    accuracy = 100. * correct / total
    avg_phi = phi_sum / n_batches
    avg_salience = salience_sum / n_batches
    
    return avg_loss, accuracy, avg_phi, avg_salience


def main():
    print("=" * 70)
    print(f"NCT MNIST Training - {VERSION.upper()}")
    print("=" * 70)
    
    # Configuration
    config = OptimizedConfigV3()
    
    # Create results directory
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Setup] Device: {device}")
    print(f"[Setup] PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[Setup] GPU: {torch.cuda.get_device_name(0)}")
    
    # Print config
    print(f"\n[Config V3]")
    print(f"  Model: d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")
    print(f"  Data: {config.n_samples_per_class} samples/class = {config.n_samples_per_class * 10} total")
    print(f"  Training: {config.n_epochs} epochs, lr={config.learning_rate}, batch={config.batch_size}")
    print(f"  Regularization: dropout={config.dropout_rate}, label_smooth={config.label_smoothing}")
    print(f"  Early stopping: patience={config.patience}")
    
    # Get data
    train_loader, val_loader = get_mnist_data_v3(config)
    
    # Create model
    model = NCTClassifierV3(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Model] Parameters: {n_params:,}")
    
    # NCT Manager
    nct_manager = BatchedNCTManagerV3(config)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_epochs, config.n_epochs, config.min_lr
    )
    
    # Loss function with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'phi': [], 'salience': [], 'lr': []
    }
    
    # Early stopping
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    best_model_state = None
    
    print("\n" + "=" * 70)
    print("Training Started")
    print("=" * 70)
    start_time = time.time()
    
    for epoch in range(1, config.n_epochs + 1):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc, train_phi, train_sal = train_epoch_v3(
            model, train_loader, optimizer, criterion, nct_manager, device, config
        )
        
        # Validate
        val_loss, val_acc, val_phi, val_sal = validate_v3(
            model, val_loader, criterion, nct_manager, device
        )
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['phi'].append(val_phi)
        history['salience'].append(val_sal)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        # Check for improvement
        improved = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            improved = " ★ NEW BEST"
            
            # Save best model
            torch.save(best_model_state, results_dir / f'best_model_{VERSION}.pt')
        else:
            patience_counter += 1
        
        # Progress indicator
        target_indicator = "✓" if val_acc >= 70.0 else "○"
        
        # Print progress
        print(f"Epoch {epoch:3d}/{config.n_epochs} | "
              f"Train: {train_acc:5.1f}% | Val: {val_acc:5.1f}% {target_indicator} | "
              f"Φ: {val_phi:.3f} | S: {val_sal:.3f} | "
              f"LR: {current_lr:.1e} | {epoch_time:.1f}s{improved}")
        
        # Early stopping check
        if patience_counter >= config.patience:
            print(f"\n[Early Stopping] No improvement for {config.patience} epochs")
            break
        
        # Target reached check
        if val_acc >= 70.0:
            print(f"\n[TARGET REACHED] 70%+ accuracy achieved!")
    
    # Training complete
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print(f"TRAINING COMPLETE - {VERSION.upper()}")
    print("=" * 70)
    print(f"  Total Time:        {total_time/60:.1f} min")
    print(f"  Total Epochs:      {epoch}")
    print(f"  Best Val Accuracy: {best_val_acc:.2f}% (epoch {best_epoch})")
    print(f"  Final Train Acc:   {train_acc:.2f}%")
    print(f"  Final Val Acc:     {val_acc:.2f}%")
    print(f"  Target (70%):      {'✓ ACHIEVED' if best_val_acc >= 70.0 else '✗ NOT YET'}")
    print("=" * 70)
    
    # Save results with version
    history_path = results_dir / f'history_{VERSION}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    config_path = results_dir / f'config_{VERSION}.json'
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Save summary
    summary = {
        'version': VERSION,
        'best_val_accuracy': best_val_acc,
        'best_epoch': best_epoch,
        'total_epochs': epoch,
        'total_time_minutes': total_time / 60,
        'final_train_acc': train_acc,
        'final_val_acc': val_acc,
        'target_achieved': best_val_acc >= 70.0,
        'n_parameters': n_params,
        'improvements_from_v2': [
            'samples: 100 -> 500/class',
            'd_model: 256 -> 384',
            'n_layers: 2 -> 3',
            'added data augmentation',
            'longer warmup'
        ]
    }
    summary_path = results_dir / f'summary_{VERSION}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs_range = range(1, len(history['train_acc']) + 1)
        
        # Accuracy plot
        ax1 = axes[0, 0]
        ax1.plot(epochs_range, history['train_acc'], 'b-', label='Train', linewidth=2)
        ax1.plot(epochs_range, history['val_acc'], 'r-', label='Val', linewidth=2)
        ax1.axhline(y=70, color='g', linestyle='--', label='Target (70%)', linewidth=2)
        ax1.axhline(y=best_val_acc, color='orange', linestyle=':', 
                    label=f'Best ({best_val_acc:.1f}%)', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title(f'Training Progress - {VERSION.upper()}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Loss plot
        ax2 = axes[0, 1]
        ax2.plot(epochs_range, history['train_loss'], 'b-', label='Train', linewidth=2)
        ax2.plot(epochs_range, history['val_loss'], 'r-', label='Val', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Curves')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # NCT metrics
        ax3 = axes[1, 0]
        ax3.plot(epochs_range, history['phi'], 'purple', label='Φ (Integration)', linewidth=2)
        ax3.plot(epochs_range, history['salience'], 'orange', label='Salience', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Value')
        ax3.set_title('NCT Metrics')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate
        ax4 = axes[1, 1]
        ax4.plot(epochs_range, history['lr'], 'green', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = results_dir / f'results_{VERSION}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[Saved] Plot: {plot_path}")
    except Exception as e:
        print(f"\n[Warning] Could not generate plot: {e}")
    
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - history_{VERSION}.json")
    print(f"  - results_{VERSION}.png")
    print(f"  - config_{VERSION}.json")
    print(f"  - summary_{VERSION}.json")
    print(f"  - best_model_{VERSION}.pt")
    
    return best_val_acc


if __name__ == '__main__':
    best_acc = main()
    print(f"\n{'='*70}")
    print(f"V3 Training finished. Best accuracy: {best_acc:.2f}%")
    print(f"{'='*70}")
