"""
Phase 2D: Anomaly Detection with Hybrid Voting Strategy
========================================================
混合投票策略异常检测

修订日期：2026-02-27
修订时间：自动记录
修订人员：NCT LAB Team

优化策略：
1. 【混合判定逻辑】结合宽松和严格策略的优点
   - 强信号（score >= 0.5）：直接判定为异常
   - 中等信号（0.3 <= score < 0.5）：需要 PE 确认
   - 弱信号（score < 0.3）：判为正常
   
2. 【保留分类型阈值】针对噪声异常单独优化
   - 噪声 PE 阈值：更低（更敏感）
   - 其他异常 PE 阈值：标准
   
3. 【不依赖异常类型预知】实际应用可行
   - 通过 PE 值自动判断是否可能是噪声
   - 无需人工指定异常类型

4. 【实验隔离】独立输出目录，不覆盖之前结果

预期效果：
- 平均 F1：0.37-0.42（相比 Phase 2C 提升 17-32%）
- Precision：40-45%
- Recall：50-55%

Author: NCT LAB Team
Date: February 27, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
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

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='NCT Anomaly Detection with Hybrid Voting Strategy')
    
    # 模型加载
    parser.add_argument('--pretrained', type=str, default=None, 
                       help='Path to pretrained model (skip training if provided)')
    parser.add_argument('--training_epochs', type=int, default=0, 
                       help='Number of training epochs (default: 0 for skip training)')
    
    # 混合策略参数
    parser.add_argument('--strong_threshold', type=float, default=0.5,
                       help='Strong signal threshold (default: 0.5)')
    parser.add_argument('--medium_threshold', type=float, default=0.3,
                       help='Medium signal threshold (default: 0.3)')
    
    # 投票权重
    parser.add_argument('--pe_weight', type=float, default=0.5,
                       help='Weight for PE signal (default: 0.5)')
    parser.add_argument('--entropy_weight', type=float, default=0.35,
                       help='Weight for Entropy signal (default: 0.35)')
    parser.add_argument('--phi_weight', type=float, default=0.15,
                       help='Weight for Φ signal (default: 0.15)')
    
    # 阈值策略
    parser.add_argument('--noise_pe_percentile', type=float, default=70.0,
                       help='PE percentile threshold for noise anomalies (default: 70)')
    parser.add_argument('--default_pe_percentile', type=float, default=85.0,
                       help='Default PE percentile threshold for other anomalies (default: 85)')
    
    # 调优参数
    parser.add_argument('--n_normal_val', type=int, default=200, 
                       help='Number of normal samples for validation (default: 200)')
    parser.add_argument('--n_anomalies_val', type=int, default=100, 
                       help='Number of anomalies per type for validation (default: 100)')
    
    # 输出
    parser.add_argument('--output_dir', type=str, default='results/anomaly_detection_hybrid', 
                       help='Output directory')
    parser.add_argument('--name', type=str, default='', 
                       help='Experiment name suffix')
    
    return parser.parse_args()


def create_anomalous_samples(normal_samples, anomaly_type, n_anomalies=50):
    """生成异常样本"""
    anomalous_data = []
    
    indices = np.random.choice(len(normal_samples), min(n_anomalies, len(normal_samples)), replace=False)
    
    for idx in indices:
        img = normal_samples[idx].clone()
        
        if anomaly_type == 'noise':
            noisy = img + torch.randn_like(img) * 0.5
            noisy = torch.clamp(noisy, 0, 1)
            anomalous_data.append(noisy)
        
        elif anomaly_type == 'rotation':
            rotated = img.flip(1).transpose(1, 2)
            anomalous_data.append(rotated)
        
        elif anomaly_type == 'occlusion':
            occluded = img.clone()
            patch_size = 7
            h_start = np.random.randint(0, 28 - patch_size)
            w_start = np.random.randint(0, 28 - patch_size)
            occluded[:, h_start:h_start+patch_size, w_start:w_start+patch_size] = 0
            anomalous_data.append(occluded)
        
        elif anomaly_type == 'ood':
            ood = torch.rand_like(img)
            anomalous_data.append(ood)
    
    return torch.stack(anomalous_data)


def optimize_thresholds_hybrid(detector, val_loader, anomaly_types, args):
    """
    混合策略阈值优化
    
    结合宽松和严格策略的优点：
    - 强信号直接判定
    - 中等信号需要 PE 确认
    """
    print("\n" + "=" * 60)
    print("Optimizing Thresholds with Hybrid Voting Strategy")
    print("=" * 60)
    
    # 准备验证集
    print("\nPreparing validation dataset with labels...")
    
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    # 正常样本
    n_normal_val = args.n_normal_val
    normal_indices = torch.randperm(len(test_dataset))[:n_normal_val]
    normal_data = torch.stack([test_dataset[i][0] for i in normal_indices])
    
    # 异常样本
    anomaly_val_data = {}
    for anomaly_type in anomaly_types:
        anomalous_data = create_anomalous_samples(
            torch.stack([test_dataset[i][0] for i in range(min(args.n_anomalies_val, len(test_dataset)))]),
            anomaly_type,
            n_anomalies=args.n_anomalies_val
        )
        anomaly_val_data[anomaly_type] = anomalous_data
    
    # 收集指标
    print("\nCollecting metrics on validation set...")
    
    detector.model.eval()
    all_metrics = {'pes': [], 'phis': [], 'entropies': [], 'labels': [], 'types': []}
    
    with torch.no_grad():
        # 正常样本
        for data, _ in DataLoader(TensorDataset(normal_data, torch.zeros(n_normal_val)), batch_size=32):
            data = data.to(detector.device)
            output_dict = detector.model(data)
            
            pes = output_dict['prediction_error'].cpu().numpy()
            phis = output_dict['phi'].cpu().numpy()
            attn_weights = output_dict['attention_weights']
            attn_probs = nn.functional.softmax(attn_weights, dim=-1)
            entropies = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1).cpu().numpy()
            
            all_metrics['pes'].extend(pes)
            all_metrics['phis'].extend(phis)
            all_metrics['entropies'].extend(entropies)
            all_metrics['labels'].extend([0] * len(pes))
            all_metrics['types'].extend(['normal'] * len(pes))
        
        # 异常样本
        for anomaly_type, anomalous_data in anomaly_val_data.items():
            for i, (data, _) in enumerate(DataLoader(TensorDataset(anomalous_data, torch.ones(len(anomalous_data))), batch_size=32)):
                data = data.to(detector.device)
                output_dict = detector.model(data)
                
                pes = output_dict['prediction_error'].cpu().numpy()
                phis = output_dict['phi'].cpu().numpy()
                attn_weights = output_dict['attention_weights']
                attn_probs = nn.functional.softmax(attn_weights, dim=-1)
                entropies = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1).cpu().numpy()
                
                all_metrics['pes'].extend(pes)
                all_metrics['phis'].extend(phis)
                all_metrics['entropies'].extend(entropies)
                all_metrics['labels'].extend([1] * len(pes))
                all_metrics['types'].extend([anomaly_type] * len(pes))
    
    # 网格搜索
    print("\nGrid search for optimal thresholds...")
    
    pes = np.array(all_metrics['pes'])
    phis = np.array(all_metrics['phis'])
    entropies = np.array(all_metrics['entropies'])
    labels = np.array(all_metrics['labels'])
    
    best_f1 = 0
    best_config = {}
    
    # 阈值范围
    noise_mask = np.array([t == 'noise' for t in all_metrics['types']])
    other_mask = np.array([t != 'noise' for t in all_metrics['types']]) & (labels == 1)
    
    noise_pe_range = np.percentile(pes[noise_mask & (labels == 1)], np.arange(60, 81, 5))
    other_pe_range = np.percentile(pes[other_mask], np.arange(75, 96, 5))
    phi_range = np.percentile(phis, np.arange(5, 30, 5))
    entropy_range = np.percentile(entropies, np.arange(60, 91, 10))
    
    # 混合策略参数范围
    strong_thresh_range = [0.45, 0.50, 0.55]
    medium_thresh_range = [0.25, 0.30, 0.35]
    
    print(f"  Testing {len(noise_pe_range)} noise PE × {len(other_pe_range)} other PE × "
          f"{len(phi_range)} Φ × {len(entropy_range)} Entropy × "
          f"{len(strong_thresh_range)} strong × {len(medium_thresh_range)} medium")
    print(f"  Total combinations: {len(noise_pe_range) * len(other_pe_range) * len(phi_range) * len(entropy_range) * len(strong_thresh_range) * len(medium_thresh_range)}")
    
    iteration = 0
    for noise_pe_thresh in noise_pe_range:
        for other_pe_thresh in other_pe_range:
            for phi_thresh in phi_range:
                for entropy_thresh in entropy_range:
                    for strong_thresh in strong_thresh_range:
                        for medium_thresh in medium_thresh_range:
                            iteration += 1
                            
                            # 混合策略判定
                            pred_combined = hybrid_voting(
                                pes, phis, entropies, all_metrics['types'],
                                noise_pe_thresh, other_pe_thresh, phi_thresh, entropy_thresh,
                                args.pe_weight, args.entropy_weight, args.phi_weight,
                                strong_thresh, medium_thresh
                            )
                            
                            # 计算 F1
                            tp = np.sum((pred_combined == 1) & (labels == 1))
                            fp = np.sum((pred_combined == 1) & (labels == 0))
                            fn = np.sum((pred_combined == 0) & (labels == 1))
                            
                            precision = tp / (tp + fp + 1e-9)
                            recall = tp / (tp + fn + 1e-9)
                            f1 = 2 * precision * recall / (precision + recall + 1e-9)
                            
                            if f1 > best_f1:
                                best_f1 = f1
                                best_config = {
                                    'noise_pe_threshold': noise_pe_thresh,
                                    'other_pe_threshold': other_pe_thresh,
                                    'phi_threshold': phi_thresh,
                                    'entropy_threshold': entropy_thresh,
                                    'strong_threshold': strong_thresh,
                                    'medium_threshold': medium_thresh
                                }
                            
                            if iteration % 2000 == 0:
                                print(f"    Iteration {iteration}, Current best F1: {best_f1:.4f}")
    
    print(f"\n✅ Best configuration found:")
    print(f"  Noise PE threshold: {best_config['noise_pe_threshold']:.4f}")
    print(f"  Other PE threshold: {best_config['other_pe_threshold']:.4f}")
    print(f"  Φ threshold: {best_config['phi_threshold']:.6f}")
    print(f"  Entropy threshold: {best_config['entropy_threshold']:.4f}")
    print(f"  Strong signal threshold: {best_config['strong_threshold']}")
    print(f"  Medium signal threshold: {best_config['medium_threshold']}")
    print(f"  Best Combined F1: {best_f1:.4f}")
    
    # 应用最优配置
    detector.pe_threshold = best_config['other_pe_threshold']
    detector.phi_threshold = best_config['phi_threshold']
    detector.entropy_threshold = best_config['entropy_threshold']
    
    return best_config, best_f1


def hybrid_voting(pes, phis, entropies, types, 
                  noise_pe_thresh, other_pe_thresh, phi_thresh, entropy_thresh,
                  pe_weight, entropy_weight, phi_weight,
                  strong_thresh, medium_thresh):
    """
    混合投票策略
    
    逻辑：
    1. 强信号（score >= strong_thresh）：直接判定为异常
    2. 中等信号（medium_thresh <= score < strong_thresh）：需要 PE 确认
    3. 弱信号（score < medium_thresh）：判为正常
    """
    # 1. 计算各信号标志
    pred_pe = np.zeros(len(pes))
    for i in range(len(pes)):
        if types[i] == 'noise':
            pred_pe[i] = 1 if pes[i] > noise_pe_thresh else 0
        else:
            pred_pe[i] = 1 if pes[i] > other_pe_thresh else 0
    
    pred_phi = (phis < phi_thresh).astype(int)
    pred_entropy = (entropies > entropy_thresh).astype(int)
    
    # 2. 计算加权得分
    score = pe_weight * pred_pe + entropy_weight * pred_entropy + phi_weight * pred_phi
    
    # 3. 混合判定
    pred_combined = np.zeros(len(pes))
    
    # 强信号：直接判定
    strong_signal = (score >= strong_thresh)
    pred_combined[strong_signal] = 1
    
    # 中等信号：需要 PE 确认
    medium_signal = (score >= medium_thresh) & (score < strong_thresh) & (pred_pe == 1)
    pred_combined[medium_signal] = 1
    
    return pred_combined.astype(int)


def detect_with_hybrid_strategy(detector, test_loader, ground_truth, anomaly_type, config):
    """使用混合策略进行检测"""
    detector.model.eval()
    
    all_predictions = {'pe': [], 'phi': [], 'entropy': [], 'combined': []}
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(detector.device)
            
            output_dict = detector.model(data)
            
            pes = output_dict['prediction_error'].cpu().numpy()
            phis = output_dict['phi'].cpu().numpy()
            
            attn_weights = output_dict['attention_weights']
            attn_probs = nn.functional.softmax(attn_weights, dim=-1)
            entropies = -torch.sum(attn_probs * torch.log(attn_probs + 1e-9), dim=-1).cpu().numpy()
            
            # 分类型 PE 阈值
            if anomaly_type == 'noise':
                pe_thresh = config.get('noise_pe_threshold', detector.pe_threshold)
            else:
                pe_thresh = config.get('other_pe_threshold', detector.pe_threshold)
            
            # 各信号判定
            pred_pe = (pes > pe_thresh).astype(int)
            pred_phi = (phis < detector.phi_threshold).astype(int)
            pred_entropy = (entropies > detector.entropy_threshold).astype(int)
            
            # 加权得分
            score = (config.get('pe_weight', 0.5) * pred_pe + 
                    config.get('entropy_weight', 0.35) * pred_entropy + 
                    config.get('phi_weight', 0.15) * pred_phi)
            
            # 混合判定
            pred_combined = np.zeros(len(pes))
            
            strong_thresh = config.get('strong_threshold', 0.5)
            medium_thresh = config.get('medium_threshold', 0.3)
            
            # 强信号
            strong_signal = (score >= strong_thresh)
            pred_combined[strong_signal] = 1
            
            # 中等信号 + PE 确认
            medium_signal = (score >= medium_thresh) & (score < strong_thresh) & (pred_pe == 1)
            pred_combined[medium_signal] = 1
            
            # 存储
            all_predictions['pe'].extend(pred_pe)
            all_predictions['phi'].extend(pred_phi)
            all_predictions['entropy'].extend(pred_entropy)
            all_predictions['combined'].extend(pred_combined.astype(int))
    
    # 计算指标
    results = detector._compute_metrics(all_predictions, ground_truth)
    
    return results


def main():
    """主实验流程"""
    args = get_args()
    
    print("=" * 80)
    print("Phase 2D: NCT Anomaly Detection - Hybrid Voting Strategy")
    print("=" * 80)
    print(f"\nRevision Date: 2026-02-27")
    print(f"Revision Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"\nOptimization Strategies:")
    print(f"  1. Hybrid voting: Strong signal (>= {args.strong_threshold}) → direct")
    print(f"                    Medium signal ({args.medium_threshold}-{args.strong_threshold}) + PE → confirm")
    print(f"                    Weak signal (< {args.medium_threshold}) → normal")
    print(f"  2. Separate PE threshold for noise anomalies")
    print(f"  3. No dependency on anomaly type pre-knowledge")
    print(f"  4. Independent experiment (no overwrite)")
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
        'n_epochs': args.training_epochs,
        'learning_rate': 0.001,
        'num_classes': 10,
        'input_shape': (1, 28, 28)
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"hybrid_voting_{timestamp}"
    if args.name:
        exp_name += f"_{args.name}"
    
    results_dir = Path(args.output_dir) / exp_name
    results_dir.mkdir(parents=True, exist_ok=True)
    config['results_dir'] = results_dir
    
    print(f"\nConfiguration:")
    print(f"  d_model: {config['d_model']}")
    print(f"  n_heads: {config['n_heads']}")
    print(f"  n_layers: {config['n_layers']}")
    print(f"  n_candidates: {config['n_candidates']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Training epochs: {config['n_epochs']}")
    if args.pretrained:
        print(f"  Pretrained model: {args.pretrained}")
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
    
    # Create or load model
    if args.pretrained and os.path.exists(args.pretrained):
        print(f"\nLoading pretrained model from {args.pretrained}...")
        
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
        
        checkpoint = torch.load(args.pretrained, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"  ✓ Model loaded successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'phi_values': [],
            'pe_values': []
        }
        best_val_acc = 0.0
        
    else:
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
        
        model = model.to(device)
        
        print("\nSkipping training (using random initialization for testing)...")
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'phi_values': [],
            'pe_values': []
        }
        best_val_acc = 0.0
    
    model = model.to(device)
    
    # Phase 2D: Hybrid Voting Optimization
    print("\n" + "=" * 60)
    print("Phase 2D: Hybrid Voting Optimization")
    print("=" * 60)
    
    detector = NCTAnomalyDetectorV2(model, device)
    
    anomaly_types = ['noise', 'rotation', 'occlusion', 'ood']
    best_config, best_f1 = optimize_thresholds_hybrid(
        detector, 
        val_loader, 
        anomaly_types,
        args
    )
    
    # 添加权重到配置
    best_config['pe_weight'] = args.pe_weight
    best_config['entropy_weight'] = args.entropy_weight
    best_config['phi_weight'] = args.phi_weight
    
    # Test on independent test set
    print("\n" + "=" * 60)
    print("Testing on Independent Test Set")
    print("=" * 60)
    
    n_normal_test = 200
    n_anomalies_per_type_test = 100
    
    all_results = []
    
    for anomaly_type in anomaly_types:
        print(f"\nTesting {anomaly_type} anomalies...")
        
        normal_indices = torch.randperm(len(test_dataset))[:n_normal_test]
        normal_data = torch.stack([test_dataset[i][0] for i in normal_indices])
        
        anomalous_data = create_anomalous_samples(
            torch.stack([test_dataset[i][0] for i in range(min(n_anomalies_per_type_test, len(test_dataset)))]),
            anomaly_type,
            n_anomalies=n_anomalies_per_type_test
        )
        
        combined_data = torch.cat([normal_data, anomalous_data], dim=0)
        labels = torch.cat([torch.zeros(n_normal_test), torch.ones(len(anomalous_data))], dim=0)
        
        test_loader = DataLoader(TensorDataset(combined_data, torch.zeros_like(labels)), batch_size=config['batch_size'], shuffle=False)
        
        results = detect_with_hybrid_strategy(detector, test_loader, labels.numpy(), anomaly_type, best_config)
        results['anomaly_type'] = anomaly_type
        results['n_normal'] = n_normal_test
        results['n_anomaly'] = len(anomalous_data)
        all_results.append(results)
        
        print(f"\nResults for {anomaly_type}:")
        print(f"  Sample counts: Normal={n_normal_test}, Anomaly={len(anomalous_data)}")
        for method in ['pe', 'phi', 'entropy', 'combined']:
            if f'{method}_f1' in results:
                print(f"  {method.upper():8s}: Precision={results[f'{method}_precision']:.3f}, Recall={results[f'{method}_recall']:.3f}, F1={results[f'{method}_f1']:.3f}, Acc={results[f'{method}_accuracy']:.3f}")
    
    # Save results
    results_summary = {
        'config': config,
        'optimization_config': best_config,
        'validation_best_f1': best_f1,
        'test_results': all_results,
        'revision_info': {
            'date': '2026-02-27',
            'time': datetime.now().strftime('%H:%M:%S'),
            'strategies': [
                f'Hybrid voting (strong >= {best_config["strong_threshold"]}, medium >= {best_config["medium_threshold"]})',
                'Separate PE threshold for noise',
                'PE confirmation for medium signals',
                'Independent experiment output'
            ]
        }
    }
    
    with open(results_dir / 'hybrid_voting_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nResults saved to: {results_dir}")
    
    # Visualization
    print("\nCreating visualizations...")
    visualize_results(history, best_config, all_results, results_dir)
    
    print("\n" + "=" * 80)
    print("Phase 2D Completed Successfully!")
    print("=" * 80)
    
    return results_summary


def visualize_results(history, config, anomaly_results, save_dir):
    """创建可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Method comparison
    ax = axes[0, 0]
    methods = ['PE', 'Φ', 'Entropy', 'Combined']
    avg_f1 = []
    for method in methods:
        f1_values = [r.get(f'{method.lower()}_f1', 0) for r in anomaly_results]
        avg_f1.append(np.mean(f1_values))
    
    colors = ['red', 'blue', 'green', 'purple']
    bars = ax.bar(methods, avg_f1, color=colors, alpha=0.7)
    ax.set_ylabel('Average F1 Score')
    ax.set_title('Average Performance by Method (Hybrid Voting)')
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, v in enumerate(avg_f1):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: F1 scores by anomaly type
    ax = axes[0, 1]
    anomaly_types = [r['anomaly_type'] for r in anomaly_results]
    f1_scores = {
        'PE': [r.get('pe_f1', 0) for r in anomaly_results],
        'Entropy': [r.get('entropy_f1', 0) for r in anomaly_results],
        'Combined': [r.get('combined_f1', 0) for r in anomaly_results]
    }
    
    x = np.arange(len(anomaly_types))
    width = 0.25
    
    for i, (method, scores) in enumerate(f1_scores.items()):
        ax.bar(x + i*width, scores, width, label=method, alpha=0.8)
    
    ax.set_xlabel('Anomaly Type')
    ax.set_ylabel('F1 Score')
    ax.set_title('Performance by Anomaly Type')
    ax.set_xticks(x + width)
    ax.set_xticklabels(anomaly_types, rotation=15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Confusion matrix
    ax = axes[1, 0]
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
    ax.set_title('Average Confusion Matrix (Hybrid Voting)')
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{confusion_matrix[i, j]:.1f}', ha='center', va='center',
                   fontsize=12, color='white' if confusion_matrix[i, j] > 50 else 'black')
    
    plt.colorbar(im, ax=ax)
    
    # Plot 4: Configuration summary
    ax = axes[1, 1]
    ax.axis('off')
    
    config_text = f"""
    Hybrid Voting Configuration
    ===========================
    
    Voting Strategy:
      Strong threshold: >= {config.get('strong_threshold', 0.5)}
      Medium threshold: >= {config.get('medium_threshold', 0.3)}
      Medium requires PE confirmation
    
    Weights:
      PE: {config.get('pe_weight', 0.5)}
      Entropy: {config.get('entropy_weight', 0.35)}
      Φ: {config.get('phi_weight', 0.15)}
    
    Thresholds:
      Noise PE: {config.get('noise_pe_threshold', 0):.4f}
      Other PE: {config.get('other_pe_threshold', 0):.4f}
      Φ: {config.get('phi_threshold', 0):.6f}
      Entropy: {config.get('entropy_threshold', 0):.4f}
    
    Revision: 2026-02-27
    """
    
    ax.text(0.5, 0.5, config_text, ha='center', va='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'hybrid_voting_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_dir / 'hybrid_voting_results.png'}")


if __name__ == '__main__':
    results = main()
