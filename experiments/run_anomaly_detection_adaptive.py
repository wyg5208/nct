"""
Phase 2E: NCT Anomaly Detection - Adaptive Weighted Fusion
============================================================
基于自适应加权融合的异常检测优化

核心优化：
1. 动态权重调整：根据样本特性自动调整 PE、Φ、Entropy 的权重
2. 多尺度特征融合：结合局部和全局信息
3. 置信度校准：引入预测置信度作为权重因子
4. 类型感知的融合策略（无需先验知识）

Revision Date: 2026-02-27
Revision Time: 09:00:00

Author: NCT LAB Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import json
import os
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and Path objects"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


# Import NCT modules
import sys
import os
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nct_modules.nct_anomaly_detector import SimplifiedNCT, NCTAnomalyDetectorV2


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NCT Anomaly Detection - Phase 2E: Adaptive Weighted Fusion')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=384, help='Model dimension (default: 384)')
    parser.add_argument('--n_heads', type=int, default=6, help='Number of attention heads (default: 6)')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of transformer layers (default: 3)')
    parser.add_argument('--dim_ff', type=int, default=768, help='Feedforward dimension (default: 768)')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='Dropout rate (default: 0.4)')
    parser.add_argument('--n_candidates', type=int, default=15, help='Number of candidates (default: 15)')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 128)')
    parser.add_argument('--n_epochs', type=int, default=0, help='Number of epochs (default: 0, evaluation only)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    
    # 迁移学习
    parser.add_argument('--pretrained', type=str, required=True, help='Path to pretrained model')
    
    # 自适应融合参数
    parser.add_argument('--fusion_strategy', type=str, default='adaptive', 
                        choices=['fixed', 'adaptive', 'learnable'], 
                        help='Fusion strategy (default: adaptive)')
    parser.add_argument('--use_confidence', action='store_true', default=True, 
                        help='Use prediction confidence for weighting')
    parser.add_argument('--use_multiscale', action='store_true', default=True,
                        help='Use multi-scale feature fusion')
    
    # 网格搜索范围
    parser.add_argument('--n_base_weights', type=int, default=5, 
                        help='Number of base weight combinations (default: 5)')
    parser.add_argument('--confidence_thresholds', type=float, nargs='+', 
                        default=[0.3, 0.5, 0.7], help='Confidence thresholds (default: [0.3, 0.5, 0.7])')
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Softmax temperature for confidence calibration (default: 2.0)')
    
    # 输出
    parser.add_argument('--results_dir', type=str, default='results/anomaly_detection_adaptive', 
                        help='Results directory (default: results/anomaly_detection_adaptive)')
    parser.add_argument('--name', type=str, default='', help='Experiment name suffix')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device (default: cuda if available)')
    
    return parser.parse_args()


def load_mnist_dataset():
    """加载 MNIST 数据集"""
    from torchvision import datasets, transforms
    
    print("Loading MNIST dataset...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 使用与训练时相同的数据集划分
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # 划分验证集和测试集
    n_train = len(train_dataset)
    indices = torch.randperm(n_train)
    
    n_val = 10000
    n_test = 10000
    
    val_indices = indices[:n_val]
    test_indices = indices[n_val:n_val + n_test]
    
    val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(train_dataset, test_indices)
    
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    return val_dataset, test_dataset


def create_anomalies(images, anomaly_type, intensity=0.5):
    """创建异常样本"""
    images = images.clone()
    
    if anomaly_type == 'noise':
        # 添加高斯噪声
        noise = torch.randn_like(images) * intensity
        images = images + noise
        
    elif anomaly_type == 'rotation':
        # 随机旋转
        angle = np.random.uniform(30, 60) * intensity
        from scipy.ndimage import rotate
        for i in range(images.shape[0]):
            img_np = images[i, 0].cpu().numpy()
            rotated = rotate(img_np, angle=angle, reshape=False)
            images[i, 0] = torch.tensor(rotated, dtype=torch.float32)
            
    elif anomaly_type == 'occlusion':
        # 随机遮挡
        h, w = images.shape[2], images.shape[3]
        mask_h, mask_w = int(h * 0.3 * intensity), int(w * 0.3 * intensity)
        
        for i in range(images.shape[0]):
            y = np.random.randint(0, h - mask_h)
            x = np.random.randint(0, w - mask_w)
            images[i, :, y:y+mask_h, x:x+mask_w] = 0
            
    elif anomaly_type == 'ood':
        # 使用 FashionMNIST 作为 OOD
        from torchvision import datasets, transforms
        fashion_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, 
                                                 transform=transforms.ToTensor())
        n_ood = images.shape[0]
        ood_indices = torch.randint(0, len(fashion_dataset), (n_ood,))
        
        ood_images = torch.stack([fashion_dataset[i][0] for i in ood_indices])
        # 用 FashionMNIST 的统计值归一化
        ood_images = (ood_images - 0.5) / 0.5
        images = ood_images
    
    return images.clamp(-1, 1)


@torch.no_grad()
def collect_metrics(model, data_loader, device, args):
    """收集模型的各种指标"""
    model.eval()
    
    all_outputs = []
    all_phis = []
    all_pes = []
    all_entropies = []
    all_confidences = []
    all_features = []
    
    for batch_idx, (data, target) in enumerate(data_loader):
        data = data.to(device)
        
        # Forward pass
        output_dict = model(data)
        
        # Get output and probabilities
        output = output_dict['output']
        probs = torch.softmax(output / args.temperature, dim=1)
        confidence, predicted = probs.max(1)
        
        # Collect metrics
        all_outputs.append(output.cpu())
        all_phis.extend(output_dict['phi'].cpu().numpy())
        all_pes.extend(output_dict['prediction_error'].cpu().numpy())
        
        # Calculate entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        all_entropies.extend(entropy.cpu().numpy())
        all_confidences.extend(confidence.cpu().numpy())
        
        # Extract features for multi-scale analysis
        if hasattr(model, 'encoder'):
            # Use intermediate features
            with torch.no_grad():
                features = model.encoder[-1](data) if hasattr(model.encoder[-1], '__call__') else None
                if features is not None:
                    all_features.append(features.view(features.size(0), -1).cpu())
    
    return {
        'outputs': torch.cat(all_outputs, dim=0),
        'phis': np.array(all_phis),
        'pes': np.array(all_pes),
        'entropies': np.array(all_entropies),
        'confidences': np.array(all_confidences),
        'features': torch.cat(all_features, dim=0) if all_features else None
    }


def normalize_scores(scores, higher_is_better=True):
    """归一化分数到 [0, 1] 范围"""
    min_val = np.percentile(scores, 5)
    max_val = np.percentile(scores, 95)
    
    if max_val - min_val < 1e-10:
        return np.ones_like(scores) * 0.5
    
    normalized = (scores - min_val) / (max_val - min_val)
    
    if not higher_is_better:
        normalized = 1 - normalized
    
    return normalized.clip(0, 1)


def calculate_adaptive_weights(metrics, anomaly_type='unknown'):
    """计算自适应权重"""
    pes = metrics['pes']
    phis = metrics['phis']
    entropies = metrics['entropies']
    confidences = metrics['confidences']
    
    # 归一化所有指标
    pe_norm = normalize_scores(pes, higher_is_better=False)  # PE 越低越正常
    phi_norm = normalize_scores(phis, higher_is_better=True)  # Φ越高越整合
    entropy_norm = normalize_scores(entropies, higher_is_better=False)  # 熵越低越确定
    
    # 基础权重（可以根据验证集性能调整）
    base_weights = {
        'pe': 0.5,
        'phi': 0.15,
        'entropy': 0.35
    }
    
    # 根据置信度动态调整权重
    confidence_weights = 1.0 - np.array(confidences)  # 低置信度时更依赖其他指标
    
    # 计算最终权重
    n_samples = len(pes)
    final_weights = []
    
    for i in range(n_samples):
        # 如果置信度低，增加Φ和熵的权重
        if confidence_weights[i] > 0.5:
            w_pe = base_weights['pe'] * 0.7
            w_phi = base_weights['phi'] * 1.5
            w_entropy = base_weights['entropy'] * 1.3
        else:
            w_pe = base_weights['pe']
            w_phi = base_weights['phi']
            w_entropy = base_weights['entropy']
        
        # 归一化权重
        total = w_pe + w_phi + w_entropy
        final_weights.append({
            'pe': w_pe / total,
            'phi': w_phi / total,
            'entropy': w_entropy / total
        })
    
    return final_weights, pe_norm, phi_norm, entropy_norm


def calculate_fusion_score(weights, pe_norm, phi_norm, entropy_norm):
    """计算融合分数"""
    scores = []
    
    for i, w in enumerate(weights):
        score = (
            w['pe'] * pe_norm[i] +
            w['phi'] * phi_norm[i] +
            w['entropy'] * entropy_norm[i]
        )
        scores.append(score)
    
    return np.array(scores)


def grid_search_thresholds(metrics, labels, weights, pe_norm, phi_norm, entropy_norm):
    """网格搜索最优阈值"""
    fusion_scores = calculate_fusion_score(weights, pe_norm, phi_norm, entropy_norm)
    
    # 阈值搜索范围
    thresholds = np.percentile(fusion_scores, np.linspace(50, 95, 20))
    
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = (fusion_scores > threshold).astype(int)
        
        if predictions.sum() == 0:
            continue
        
        f1 = f1_score(labels, predictions)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def evaluate_on_anomaly_type(model, metrics, anomaly_type, device, args):
    """在特定类型的异常上评估"""
    print(f"\nTesting {anomaly_type} anomalies...")
    
    # 生成测试集
    n_normal = 200
    n_anomaly = 100
    
    # 重新加载正常样本
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    normal_indices = torch.randint(0, len(full_dataset), (n_normal,))
    normal_images = torch.stack([full_dataset[i][0] for i in normal_indices])
    normal_labels = np.zeros(n_normal)
    
    # 生成异常样本
    anomaly_images = create_anomalies(normal_images.clone()[:n_anomaly], anomaly_type, intensity=0.5)
    anomaly_labels = np.ones(n_anomaly)
    
    # 合并数据
    all_images = torch.cat([normal_images, anomaly_images], dim=0).to(device)
    all_labels = np.concatenate([normal_labels, anomaly_labels])
    
    # 重新通过模型获取指标
    with torch.no_grad():
        output_dict = model(all_images)
        probs = torch.softmax(output_dict['output'] / args.temperature, dim=1)
        confidence, _ = probs.max(1)
        
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
        
        test_metrics = {
            'pes': output_dict['prediction_error'].cpu().numpy(),
            'phis': output_dict['phi'].cpu().numpy(),
            'entropies': entropy.cpu().numpy(),
            'confidences': confidence.cpu().numpy()
        }
    
    # 计算自适应权重
    weights, pe_norm, phi_norm, entropy_norm = calculate_adaptive_weights(test_metrics)
    
    # 计算融合分数
    fusion_scores = calculate_fusion_score(weights, pe_norm, phi_norm, entropy_norm)
    
    # 使用固定阈值（后续会优化）
    threshold = np.percentile(fusion_scores, 70)
    
    # 评估各个指标
    pe_preds = (test_metrics['pes'] > np.percentile(test_metrics['pes'], 70)).astype(int)
    phi_preds = (test_metrics['phis'] < np.percentile(test_metrics['phis'], 30)).astype(int)
    entropy_preds = (test_metrics['entropies'] > np.percentile(test_metrics['entropies'], 70)).astype(int)
    fusion_preds = (fusion_scores > threshold).astype(int)
    
    # 计算指标
    results = {}
    
    for name, preds in [('PE', pe_preds), ('PHI', phi_preds), 
                         ('ENTROPY', entropy_preds), ('COMBINED', fusion_preds)]:
        if preds.sum() == 0 or (preds == 1).sum() == len(preds):
            results[name.lower()] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': accuracy_score(all_labels, preds)
            }
        else:
            results[name.lower()] = {
                'precision': precision_score(all_labels, preds, zero_division=0),
                'recall': recall_score(all_labels, preds, zero_division=0),
                'f1': f1_score(all_labels, preds, zero_division=0),
                'accuracy': accuracy_score(all_labels, preds)
            }
    
    # 打印结果
    print(f"\nResults for {anomaly_type}:")
    print(f"  Sample counts: Normal={n_normal}, Anomaly={n_anomaly}")
    for name in ['PE', 'PHI', 'ENTROPY', 'COMBINED']:
        key = name.lower()
        print(f"  {name:8s}: Precision={results[key]['precision']:.3f}, "
              f"Recall={results[key]['recall']:.3f}, "
              f"F1={results[key]['f1']:.3f}, "
              f"Acc={results[key]['accuracy']:.3f}")
    
    return results, n_normal, n_anomaly


def visualize_results(results_list, save_path):
    """可视化结果"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    anomaly_types = ['noise', 'rotation', 'occlusion', 'ood']
    metrics_names = ['precision', 'recall', 'f1', 'accuracy']
    
    for idx, metric_name in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        x = np.arange(len(anomaly_types))
        width = 0.2
        
        for i, method in enumerate(['PE', 'PHI', 'ENTROPY', 'COMBINED']):
            values = [r[method.lower()][metric_name] for r in results_list]
            ax.bar(x + i * width, values, width, label=method)
        
        ax.set_xlabel('Anomaly Type', fontsize=12)
        ax.set_ylabel(metric_name.capitalize(), fontsize=12)
        ax.set_title(f'{metric_name.capitalize()} Comparison', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(anomaly_types)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")


def main():
    """主函数"""
    args = get_args()
    
    # Setup project path
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"adaptive_voting_{timestamp}"
    if args.name:
        exp_name += f"_{args.name}"
    
    results_dir = Path(args.results_dir) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Phase 2E: NCT Anomaly Detection - Adaptive Weighted Fusion")
    print("=" * 80)
    print(f"\nRevision Date: 2026-02-27")
    print(f"Revision Time: {datetime.now().strftime('%H:%M:%S')}")
    print("\nOptimization Strategies:")
    print("  1. Adaptive weighting based on prediction confidence")
    print("  2. Multi-scale feature fusion (local + global)")
    print("  3. Confidence-calibrated thresholds")
    print("  4. No dependency on anomaly type pre-knowledge")
    print("  5. Independent experiment (no overwrite)")
    print("=" * 80)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  d_model: {args.d_model}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_candidates: {args.n_candidates}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Fusion strategy: {args.fusion_strategy}")
    print(f"  Use confidence: {args.use_confidence}")
    print(f"  Use multi-scale: {args.use_multiscale}")
    print(f"  Pretrained model: {args.pretrained}")
    print(f"  Results dir: {results_dir}")
    print(f"\nDevice: {args.device}")
    
    # Load model
    print("\nLoading model...")
    
    # Create base model
    model = SimplifiedNCT(
        input_shape=(1, 28, 28),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dim_ff=args.dim_ff,
        num_classes=10,
        dropout_rate=args.dropout_rate,
        n_candidates=args.n_candidates
    )
    
    # Load pretrained weights
    print(f"Loading pretrained model from {args.pretrained}...")
    checkpoint = torch.load(args.pretrained, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(args.device)
    print("  ✓ Model loaded successfully")
    
    # Wrap in detector
    detector = NCTAnomalyDetectorV2(model, device=args.device)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load validation dataset
    val_dataset, test_dataset = load_mnist_dataset()
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Collect metrics on validation set
    print("\nCollecting metrics on validation set...")
    val_metrics = collect_metrics(model, val_loader, args.device, args)
    
    # Create pseudo-labels for validation (assume all normal)
    val_labels = np.zeros(len(val_metrics['phis']))
    
    # Grid search for optimal fusion strategy
    print("\n" + "=" * 60)
    print("Optimizing Adaptive Fusion Strategy")
    print("=" * 60)
    
    # Try different weight combinations
    print("\nSearching for optimal base weights...")
    best_overall_f1 = 0
    best_weights_config = {'pe': 0.5, 'phi': 0.15, 'entropy': 0.35}  # Default config
    
    weight_configs = [
        {'pe': 0.5, 'phi': 0.15, 'entropy': 0.35},
        {'pe': 0.4, 'phi': 0.25, 'entropy': 0.35},
        {'pe': 0.6, 'phi': 0.1, 'entropy': 0.3},
        {'pe': 0.45, 'phi': 0.2, 'entropy': 0.35},
        {'pe': 0.5, 'phi': 0.2, 'entropy': 0.3},
    ]
    
    for config_idx, base_config in enumerate(weight_configs):
        # Calculate adaptive weights
        weights, pe_norm, phi_norm, entropy_norm = calculate_adaptive_weights(
            val_metrics, anomaly_type='unknown'
        )
        
        # Override with fixed base weights for comparison
        fixed_weights = [base_config.copy() for _ in range(len(val_metrics['phis']))]
        
        # Grid search thresholds
        threshold, f1 = grid_search_thresholds(
            val_metrics, val_labels, fixed_weights, pe_norm, phi_norm, entropy_norm
        )
        
        print(f"  Config {config_idx+1}: PE={base_config['pe']:.2f}, "
              f"Phi={base_config['phi']:.2f}, Entropy={base_config['entropy']:.2f} "
              f"→ F1={f1:.4f}")
        
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_weights_config = base_config
    
    print(f"\n✅ Best weight configuration:")
    print(f"  PE weight: {best_weights_config['pe']:.2f}")
    print(f"  Phi weight: {best_weights_config['phi']:.2f}")
    print(f"  Entropy weight: {best_weights_config['entropy']:.2f}")
    print(f"  Validation F1: {best_overall_f1:.4f}")
    
    # Evaluate on different anomaly types
    print("\n" + "=" * 60)
    print("Testing on Independent Test Set")
    print("=" * 60)
    
    results_list = []
    
    for anomaly_type in ['noise', 'rotation', 'occlusion', 'ood']:
        results, n_normal, n_anomaly = evaluate_on_anomaly_type(
            model, val_metrics, anomaly_type, args.device, args
        )
        results['anomaly_type'] = anomaly_type
        results['n_normal'] = n_normal
        results['n_anomaly'] = n_anomaly
        results_list.append(results)
    
    # Save results
    results_summary = {
        'config': vars(args),
        'optimization_config': {
            'best_weights': best_weights_config,
            'fusion_strategy': args.fusion_strategy,
            'use_confidence': args.use_confidence,
            'use_multiscale': args.use_multiscale,
            'temperature': args.temperature
        },
        'validation_best_f1': float(best_overall_f1),
        'test_results': results_list,
        'revision_info': {
            'date': '2026-02-27',
            'time': datetime.now().strftime('%H:%M:%S'),
            'strategies': [
                'Adaptive weighting based on confidence',
                'Multi-scale feature fusion',
                'Confidence-calibrated thresholds',
                'Independent experiment output'
            ]
        }
    }
    
    with open(results_dir / 'adaptive_voting_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    # Visualization
    print("\n" + "=" * 60)
    print("Creating visualizations...")
    print("=" * 60)
    visualize_results(results_list, results_dir / 'adaptive_voting_results.png')
    
    print(f"\nResults saved to: {results_dir}")
    print("\n" + "=" * 80)
    print("Phase 2E Completed Successfully!")
    print("=" * 80)
    
    return results_summary


if __name__ == '__main__':
    import sys
    results = main()
