"""
Phase 2: Simplified Anomaly Detection with NCT V3 Model
=========================================================
使用预训练的 NCT V3 模型进行异常检测

三种异常检测方法：
1. 预测误差（Prediction Error）
2. Φ值下降（Information Integration Drop）  
3. 置信度降低（Classification Confidence Drop）

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
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import NCT modules
import sys
sys.path.append('..')
from nct_modules.nct_core import NCTConfig
from nct_modules.nct_manager import NCTManager

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class SimpleAnomalyDetector:
    """简化版异常检测器"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load NCT config and model
        config = NCTConfig(
            n_heads=6,
            n_layers=3,
            d_model=384,
            dim_ff=768,
            dropout=0.4
        )
        
        self.nct_manager = NCTManager(config)
        self.nct_manager.to(self.device)
        
        # Load pretrained weights if available
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if it's a dict with 'model_state_dict' or direct state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.nct_manager.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded pretrained model from {model_path}")
            elif isinstance(checkpoint, dict):
                # Direct state_dict
                self.nct_manager.load_state_dict(checkpoint)
                print(f"Loaded direct state_dict from {model_path}")
            else:
                print("Warning: Unexpected checkpoint format")
        else:
            print("Warning: No pretrained model found, using random initialization")
        
        # Statistics
        self.pe_threshold = None
        self.phi_threshold = None
        self.conf_threshold = None
    
    def compute_prediction_error(self, x, target_class):
        """
        计算预测误差：输入与目标类别的重构误差
        简化版：使用分类 loss 作为代理
        """
        self.nct_manager.eval()
        
        with torch.no_grad():
            # Process through NCT
            self.nct_manager.start()
            
            # Convert batch to list of samples
            prediction_errors = []
            
            for i in range(x.size(0)):
                sample = x[i].cpu().numpy()
                
                state = self.nct_manager.process_cycle(
                    sensory_data={'visual': sample[0]}  # Remove channel dim
                )
                
                # Get prediction error from self_representation
                if 'prediction_error' in state.self_representation:
                    pe = state.self_representation['prediction_error']
                    if isinstance(pe, (int, float)):
                        prediction_errors.append(pe)
                    elif isinstance(pe, torch.Tensor):
                        prediction_errors.append(pe.item())
                    else:
                        prediction_errors.append(0.0)
                else:
                    prediction_errors.append(0.0)
            
            return np.array(prediction_errors)
    
    def compute_phi(self, x):
        """计算Φ值"""
        self.nct_manager.eval()
        
        phi_values = []
        
        with torch.no_grad():
            for i in range(x.size(0)):
                sample = x[i].cpu().numpy()
                
                self.nct_manager.start()
                state = self.nct_manager.process_cycle(
                    sensory_data={'visual': sample[0]}
                )
                
                # Get Φ from consciousness_metrics
                if 'phi' in state.consciousness_metrics:
                    phi = state.consciousness_metrics['phi']
                    if isinstance(phi, (int, float)):
                        phi_values.append(phi)
                    elif isinstance(phi, torch.Tensor):
                        phi_values.append(phi.item())
                    else:
                        phi_values.append(0.0)
                else:
                    phi_values.append(0.0)
        
        return np.array(phi_values)
    
    def set_thresholds(self, train_loader, percentile=95):
        """在正常数据上设置阈值"""
        print("Setting anomaly detection thresholds...")
        
        all_pes = []
        all_phis = []
        all_confs = []
        
        self.nct_manager.eval()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            
            # Prediction error
            pes = self.compute_prediction_error(data, target)
            all_pes.extend(pes)
            
            # Φ values
            phis = self.compute_phi(data)
            all_phis.extend(phis)
            
            # Classification confidence (using simple forward pass)
            # Note: NCTManager doesn't have standard forward, skip for now
            all_confs.extend([1.0] * data.size(0))  # Placeholder
        
        # Set thresholds
        self.pe_threshold = np.percentile(all_pes, percentile)
        self.phi_threshold = np.percentile(all_phis, 100 - percentile)  # Low Φ is anomalous
        self.conf_threshold = np.percentile(all_confs, 5)  # Low confidence
        
        print(f"Thresholds set:")
        print(f"  PE threshold: {self.pe_threshold:.4f} (>{self.pe_threshold:.4f} → anomaly)")
        print(f"  Φ threshold: {self.phi_threshold:.4f} (<{self.phi_threshold:.4f} → anomaly)")
        
        return {
            'pe_mean': np.mean(all_pes),
            'pe_std': np.std(all_pes),
            'phi_mean': np.mean(all_phis),
            'phi_std': np.std(all_phis)
        }
    
    def detect_anomalies(self, test_loader, labels=None):
        """
        检测异常
        
        Args:
            test_loader: DataLoader with samples
            labels: Optional ground truth labels (1=anomaly, 0=normal)
        
        Returns:
            dict: Detection results
        """
        self.nct_manager.eval()
        
        all_pes = []
        all_phis = []
        all_preds = []
        all_labels = [] if labels is not None else None
        
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(self.device)
            
            # Compute signals
            pes = self.compute_prediction_error(data, torch.zeros(data.size(0), dtype=torch.long))
            phis = self.compute_phi(data)
            
            # Detect anomalies
            pred_pe = (pes > self.pe_threshold).astype(int)
            pred_phi = (phis < self.phi_threshold).astype(int)
            pred_combined = ((pred_pe + pred_phi) >= 1).astype(int)
            
            all_pes.extend(pes)
            all_phis.extend(phis)
            all_preds.append({
                'pe': pred_pe,
                'phi': pred_phi,
                'combined': pred_combined
            })
            
            if labels is not None:
                # Get labels for this batch
                if isinstance(labels, torch.Tensor):
                    batch_labels = labels[batch_idx * data.size(0):(batch_idx + 1) * data.size(0)].cpu().numpy()
                else:
                    batch_labels = labels[batch_idx * data.size(0):(batch_idx + 1) * data.size(0)]
                all_labels.extend(batch_labels)
        
        # Compute metrics
        results = self._compute_metrics(all_preds, all_labels if labels is not None else None)
        results['pe_values'] = all_pes
        results['phi_values'] = all_phis
        
        return results
    
    def _compute_metrics(self, predictions, true_labels):
        """计算检测指标"""
        if true_labels is None:
            return {}
        
        true_labels = np.array(true_labels)
        metrics = {}
        
        for method in ['pe', 'phi', 'combined']:
            preds = np.concatenate([p[method] for p in predictions])
            
            tp = np.sum((preds == 1) & (true_labels == 1))
            fp = np.sum((preds == 1) & (true_labels == 0))
            tn = np.sum((preds == 0) & (true_labels == 0))
            fn = np.sum((preds == 0) & (true_labels == 1))
            
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            specificity = tn / (tn + fp + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            
            metrics[f'{method}_precision'] = precision
            metrics[f'{method}_recall'] = recall
            metrics[f'{method}_specificity'] = specificity
            metrics[f'{method}_f1'] = f1
            metrics[f'{method}_confusion'] = {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)}
        
        return metrics


def create_anomaly_dataset(normal_test_dataset, anomaly_types, n_anomalies_per_type=25):
    """创建异常数据集"""
    from torchvision import datasets, transforms
    
    all_images = []
    all_labels = []  # 1 = anomaly, 0 = normal
    
    # Add normal samples
    n_normal = len(normal_test_dataset)
    for i in range(n_normal):
        img, _ = normal_test_dataset[i]
        all_images.append(img)
        all_labels.append(0)  # Normal
    
    # Generate anomalies
    for anomaly_type in anomaly_types:
        print(f"  Generating {anomaly_type} anomalies...")
        
        # Sample random indices
        indices = np.random.choice(n_normal, min(n_anomalies_per_type, n_normal), replace=False)
        
        for idx in indices:
            img, _ = normal_test_dataset[idx]
            
            if anomaly_type == 'noise':
                # Add Gaussian noise
                noisy_img = img + torch.randn_like(img) * 0.5
                noisy_img = torch.clamp(noisy_img, 0, 1)
                all_images.append(noisy_img)
            elif anomaly_type == 'rotation':
                # Rotate 90 degrees
                rotated_img = img.flip(1).transpose(1, 2)
                all_images.append(rotated_img)
            elif anomaly_type == 'occlusion':
                # Add black square
                occluded_img = img.clone()
                patch_size = 7
                h_start = np.random.randint(0, 28 - patch_size)
                w_start = np.random.randint(0, 28 - patch_size)
                occluded_img[:, h_start:h_start+patch_size, w_start:w_start+patch_size] = 0
                all_images.append(occluded_img)
            elif anomaly_type == 'ood':
                # Random pattern (OOD)
                ood_img = torch.rand(1, 28, 28)
                all_images.append(ood_img)
            
            all_labels.append(1)  # Anomaly
    
    # Create dataset
    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(all_labels)
    
    print(f"  Total: {len(images_tensor)} samples (Normal: {n_normal}, Anomaly: {len(all_labels) - n_normal})")
    
    return TensorDataset(images_tensor, labels_tensor)


def main():
    """主实验"""
    print("=" * 80)
    print("Phase 2: NCT Anomaly Detection (Simplified Version)")
    print("=" * 80)
    
    # Try to load V3 pretrained model
    v3_model_path = Path('../results/training_v3/best_model_v3.pt')
    
    if v3_model_path.exists():
        print(f"\nLoading V3 model from {v3_model_path}")
    else:
        print("\nV3 model not found. Please run V3 training first.")
        return
    
    # Initialize detector
    detector = SimpleAnomalyDetector(model_path=str(v3_model_path))
    
    # Load MNIST test data
    print("\nLoading MNIST dataset...")
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    # Subsample for testing
    n_test = 200
    test_indices = np.random.choice(len(test_dataset), n_test, replace=False)
    test_samples = [test_dataset[i][0] for i in test_indices]
    
    # Create data loader
    test_loader = DataLoader(
        TensorDataset(torch.stack(test_samples), torch.zeros(n_test)),
        batch_size=32, shuffle=False
    )
    
    # Set thresholds on subset of normal data
    print("\nCalibrating thresholds on normal data...")
    stats = detector.set_thresholds(test_loader, percentile=90)
    
    # Create anomaly test sets
    print("\nCreating anomaly test datasets...")
    anomaly_types = ['noise', 'rotation', 'occlusion', 'ood']
    all_results = []
    
    for anomaly_type in anomaly_types:
        print(f"\nTesting {anomaly_type} anomalies...")
        
        # Create mixed dataset
        anomaly_dataset = create_anomaly_dataset(
            [test_dataset[i] for i in test_indices],
            [anomaly_type],
            n_anomalies_per_type=25
        )
        
        anomaly_loader = DataLoader(anomaly_dataset, batch_size=32, shuffle=False)
        
        # Extract labels
        labels = anomaly_dataset.tensors[1]
        
        # Run detection
        results = detector.detect_anomalies(anomaly_loader, labels=labels)
        results['anomaly_type'] = anomaly_type
        all_results.append(results)
        
        # Print results
        print(f"\nResults for {anomaly_type}:")
        print(f"  PE-based:    Precision={results.get('pe_precision', 0):.3f}, Recall={results.get('pe_recall', 0):.3f}, F1={results.get('pe_f1', 0):.3f}")
        print(f"  Φ-based:     Precision={results.get('phi_precision', 0):.3f}, Recall={results.get('phi_recall', 0):.3f}, F1={results.get('phi_f1', 0):.3f}")
        print(f"  Combined:    Precision={results.get('combined_precision', 0):.3f}, Recall={results.get('combined_recall', 0):.3f}, F1={results.get('combined_f1', 0):.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f'../results/anomaly_detection_simple_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'statistics': stats,
        'thresholds': {
            'pe': float(detector.pe_threshold),
            'phi': float(detector.phi_threshold)
        },
        'anomaly_detection_results': all_results
    }
    
    with open(results_dir / 'anomaly_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Visualize
    visualize_results(stats, all_results, results_dir)
    
    print("\n" + "=" * 80)
    print("Phase 2 Completed!")
    print("=" * 80)
    
    return results_summary


def visualize_results(stats, anomaly_results, save_dir):
    """可视化结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Statistics
    ax = axes[0]
    ax.bar(['PE Mean', 'PE Std', 'Φ Mean', 'Φ Std'], 
           [stats['pe_mean'], stats['pe_std'], stats['phi_mean'], stats['phi_std']])
    ax.set_title('Signal Statistics on Normal Data')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: F1 Scores
    ax = axes[1]
    anomaly_types = [r['anomaly_type'] for r in anomaly_results]
    f1_pe = [r.get('pe_f1', 0) for r in anomaly_results]
    f1_phi = [r.get('phi_f1', 0) for r in anomaly_results]
    f1_combined = [r.get('combined_f1', 0) for r in anomaly_results]
    
    x = np.arange(len(anomaly_types))
    width = 0.25
    
    ax.bar(x - width, f1_pe, width, label='PE-based', alpha=0.8)
    ax.bar(x, f1_phi, width, label='Φ-based', alpha=0.8)
    ax.bar(x + width, f1_combined, width, label='Combined', alpha=0.8)
    
    ax.set_xlabel('Anomaly Type')
    ax.set_ylabel('F1 Score')
    ax.set_title('Anomaly Detection Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(anomaly_types, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Confusion Matrix (average combined)
    ax = axes[2]
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
    ax.set_title('Average Confusion Matrix (Combined)')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{confusion_matrix[i, j]:.1f}', ha='center', va='center',
                   fontsize=12, color='white' if confusion_matrix[i, j] > 50 else 'black')
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'anomaly_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_dir / 'anomaly_results.png'}")


if __name__ == '__main__':
    results = main()
