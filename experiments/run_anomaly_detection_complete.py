"""
Phase 2: Complete Anomaly Detection Experiment with NCT
========================================================
使用 SimplifiedNCT 进行完整的异常检测实验

实验设计：
1. 在 MNIST 正常数据上训练模型
2. 校准异常检测阈值
3. 测试四种异常类型：噪声、旋转、遮挡、OOD
4. 对比三种检测方法的效果

Author: WENG YONGGANG
Affiliation: NeuroConscious Lab, Universiti Teknologi Malaysia
Date: February 24, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import NCT modules - 修复路径问题
import sys
import os
# 添加项目根目录到 Python 路径（无论从哪里执行）
project_root = Path(__file__).parent.parent  # experiments/.. = project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nct_modules.nct_anomaly_detector import SimplifiedNCT, NCTAnomalyDetectorV2

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def create_anomalous_samples(normal_samples, anomaly_type, n_anomalies=50):
    """
    生成异常样本
    
    Args:
        normal_samples: 正常样本 Tensor [N, C, H, W]
        anomaly_type: 异常类型 ['noise', 'rotation', 'occlusion', 'ood']
        n_anomalies: 异常样本数量
    
    Returns:
        anomalous_data: 异常样本 Tensor
        labels: 标签列表（1=异常）
    """
    anomalous_data = []
    
    indices = np.random.choice(len(normal_samples), min(n_anomalies, len(normal_samples)), replace=False)
    
    for idx in indices:
        img = normal_samples[idx].clone()
        
        if anomaly_type == 'noise':
            # 高斯噪声
            noisy = img + torch.randn_like(img) * 0.5
            noisy = torch.clamp(noisy, 0, 1)
            anomalous_data.append(noisy)
        
        elif anomaly_type == 'rotation':
            # 90 度旋转
            rotated = img.flip(1).transpose(1, 2)
            anomalous_data.append(rotated)
        
        elif anomaly_type == 'occlusion':
            # 随机遮挡
            occluded = img.clone()
            patch_size = 7
            h_start = np.random.randint(0, 28 - patch_size)
            w_start = np.random.randint(0, 28 - patch_size)
            occluded[:, h_start:h_start+patch_size, w_start:w_start+patch_size] = 0
            anomalous_data.append(occluded)
        
        elif anomaly_type == 'ood':
            # OOD 随机图案
            ood = torch.rand_like(img)
            anomalous_data.append(ood)
    
    return torch.stack(anomalous_data)


def train_model(model, train_loader, val_loader, device, config):
    """训练模型"""
    print("\n" + "=" * 60)
    print("Training SimplifiedNCT Model")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['n_epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'phi_values': [],
        'pe_values': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(config['n_epochs']):
        # Training
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_phis = []
        all_pes = []
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output_dict = model(data)
            output = output_dict['output']
            
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            total += data.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Collect metrics
            all_phis.extend(output_dict['phi'].detach().cpu().numpy())
            all_pes.extend(output_dict['prediction_error'].detach().cpu().numpy())
        
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                output_dict = model(data)
                output = output_dict['output']
                
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                
                _, predicted = output.max(1)
                val_total += data.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['phi_values'].append(np.mean(all_phis))
        history['pe_values'].append(np.mean(all_pes))
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{config['n_epochs']}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"  Avg Φ: {np.mean(all_phis):.4f}, Avg PE: {np.mean(all_pes):.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, config['results_dir'] / 'best_model.pt')
            
            print(f"  ⭐ New best model! Val Acc: {val_acc:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return history, best_val_acc


def main():
    """主实验流程"""
    print("=" * 80)
    print("Phase 2: NCT Anomaly Detection - Complete Experiment")
    print("=" * 80)
    
    # Configuration
    config = {
        'd_model': 384,
        'n_heads': 6,
        'n_layers': 3,
        'dim_ff': 768,
        'dropout_rate': 0.4,
        'n_candidates': 15,
        'batch_size': 128,
        'n_epochs': 50,
        'learning_rate': 0.001,
        'num_classes': 10,
        'input_shape': (1, 28, 28)
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f'results/anomaly_detection_complete_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    config['results_dir'] = results_dir
    
    print(f"\nConfiguration:")
    print(f"  d_model: {config['d_model']}")
    print(f"  n_heads: {config['n_heads']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  n_candidates: {config['n_candidates']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Epochs: {config['n_epochs']}")
    print(f"  Results dir: {results_dir}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load MNIST dataset
    print("\nLoading MNIST dataset...")
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    # Split training data for validation
    n_train = 50000
    n_val = 10000
    indices = torch.randperm(len(train_dataset))
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train+n_val]
    
    train_data = torch.stack([train_dataset[i][0] for i in train_indices])
    train_labels = torch.tensor([train_dataset[i][1] for i in train_indices])
    val_data = torch.stack([train_dataset[i][0] for i in val_indices])
    val_labels = torch.tensor([train_dataset[i][1] for i in val_indices])
    
    train_loader = DataLoader(TensorDataset(train_data, train_labels), batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data, val_labels), batch_size=config['batch_size'], shuffle=False)
    
    print(f"  Train samples: {len(train_data)}")
    print(f"  Val samples: {len(val_data)}")
    
    # Create and train model
    model = SimplifiedNCT(
        input_shape=config['input_shape'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dim_ff=config['dim_ff'],
        num_classes=config['num_classes'],
        dropout_rate=config['dropout_rate'],
        n_candidates=config['n_candidates']
    )
    
    print(f"\nModel created:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Train
    history, best_val_acc = train_model(model, train_loader, val_loader, device, config)
    
    # Phase 2.2: Anomaly Detection
    print("\n" + "=" * 60)
    print("Phase 2.2: Anomaly Detection Experiments")
    print("=" * 60)
    
    # Initialize detector
    detector = NCTAnomalyDetectorV2(model, device)
    
    # Calibrate on normal validation data
    print("\nCalibrating thresholds on normal data...")
    stats = detector.calibrate(val_loader, percentile=90)
    
    # Create anomaly test sets
    anomaly_types = ['noise', 'rotation', 'occlusion', 'ood']
    n_anomalies_per_type = 100
    n_normal_test = 200  # Normal samples from test set
    
    all_results = []
    
    for anomaly_type in anomaly_types:
        print(f"\nTesting {anomaly_type} anomalies...")
        
        # Prepare test data
        # Normal samples
        normal_indices = torch.randperm(len(test_dataset))[:n_normal_test]
        normal_data = torch.stack([test_dataset[i][0] for i in normal_indices])
        
        # Anomalous samples
        anomalous_data = create_anomalous_samples(
            torch.stack([test_dataset[i][0] for i in range(min(n_anomalies_per_type, len(test_dataset)))]),
            anomaly_type,
            n_anomalies=n_anomalies_per_type
        )
        
        # Combine
        combined_data = torch.cat([normal_data, anomalous_data], dim=0)
        labels = torch.cat([torch.zeros(n_normal_test), torch.ones(len(anomalous_data))], dim=0)
        
        # Create loader
        test_loader = DataLoader(TensorDataset(combined_data, torch.zeros_like(labels)), batch_size=config['batch_size'], shuffle=False)
        
        # Detect
        results = detector.detect(test_loader, ground_truth=labels.numpy())
        results['anomaly_type'] = anomaly_type
        results['n_normal'] = n_normal_test
        results['n_anomaly'] = len(anomalous_data)
        all_results.append(results)
        
        # Print results
        print(f"\nResults for {anomaly_type}:")
        print(f"  Sample counts: Normal={n_normal_test}, Anomaly={len(anomalous_data)}")
        for method in ['pe', 'phi', 'entropy', 'combined']:
            if f'{method}_f1' in results:
                print(f"  {method.upper():8s}: Precision={results[f'{method}_precision']:.3f}, Recall={results[f'{method}_recall']:.3f}, F1={results[f'{method}_f1']:.3f}, Acc={results[f'{method}_accuracy']:.3f}")
    
    # Save results
    results_summary = {
        'config': config,
        'training_history': history,
        'calibration_stats': stats,
        'thresholds': {
            'pe': float(detector.pe_threshold),
            'phi': float(detector.phi_threshold),
            'entropy': float(detector.entropy_threshold)
        },
        'anomaly_detection_results': all_results,
        'best_val_acc': best_val_acc
    }
    
    with open(results_dir / 'anomaly_detection_full_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Visualization
    print("\nCreating visualizations...")
    visualize_results(history, stats, all_results, results_dir)
    
    print("\n" + "=" * 80)
    print("Phase 2 Completed Successfully!")
    print("=" * 80)
    
    return results_summary


def visualize_results(history, stats, anomaly_results, save_dir):
    """创建完整可视化"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    
    # Plot 1: Training curves
    ax = axes[0, 0]
    ax.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Loss curves
    ax = axes[0, 1]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Φ value trajectory
    ax = axes[1, 0]
    ax.plot(history['phi_values'], color='green', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Φ Value')
    ax.set_title('Information Integration (Φ) During Training')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Prediction Error trajectory
    ax = axes[1, 1]
    ax.plot(history['pe_values'], color='red', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Prediction Error')
    ax.set_title('Free Energy (Prediction Error) During Training')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Anomaly detection F1 scores
    ax = axes[2, 0]
    anomaly_types = [r['anomaly_type'] for r in anomaly_results]
    f1_scores = {
        'PE': [r.get('pe_f1', 0) for r in anomaly_results],
        'Φ': [r.get('phi_f1', 0) for r in anomaly_results],
        'Entropy': [r.get('entropy_f1', 0) for r in anomaly_results],
        'Combined': [r.get('combined_f1', 0) for r in anomaly_results]
    }
    
    x = np.arange(len(anomaly_types))
    width = 0.2
    
    for i, (method, scores) in enumerate(f1_scores.items()):
        ax.bar(x + i*width, scores, width, label=method, alpha=0.8)
    
    ax.set_xlabel('Anomaly Type')
    ax.set_ylabel('F1 Score')
    ax.set_title('Anomaly Detection Performance by Method')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(anomaly_types, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Confusion matrix (combined, average)
    ax = axes[2, 1]
    avg_tp = np.mean([r['combined_confusion']['tp'] for r in anomaly_results])
    avg_fp = np.mean([r['combined_confusion']['fp'] for r in anomaly_results])
    avg_tn = np.mean([r['combined_confusion']['tn'] for r in anomaly_results])
    avg_fn = np.mean([r['combined_confusion']['fn'] for r in anomaly_results])
    
    confusion_matrix = np.array([[avg_tp, avg_fp], [avg_fn, avg_tn]])
    
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred Normal', 'Pred Anomaly'])
    ax.set_yticklabels(['Actual Normal', 'Actual Anomaly'])
    ax.set_title('Average Confusion Matrix (Combined Method)')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{confusion_matrix[i, j]:.1f}', ha='center', va='center',
                   fontsize=12, color='white' if confusion_matrix[i, j] > 50 else 'black')
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'anomaly_detection_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_dir / 'anomaly_detection_results.png'}")


if __name__ == '__main__':
    results = main()
